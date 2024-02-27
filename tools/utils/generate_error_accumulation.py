# import common
# import header
import argparse
# import algo
# import tunstall
import copy
from functools import partial
from huffman import *
from common import *
from header import *
from algo import *
from tunstall import *
import time
import scipy.io as io
import math
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


parser = argparse.ArgumentParser()
parser.add_argument('--archname', default='resnet34py', type=str,
                    help='name of network architecture: resnet18, resnet34, resnet50, densenet, etc')
parser.add_argument('--pathrdcurve', default='./rd_curves', \
                    type=str,
                    help='path of rate distortion curves')
parser.add_argument('--maxdeadzones', default=10, type=int,
                    help='number of sizes of dead zones')
parser.add_argument('--maxrates', default=11, type=int,
                    help='number of bit rates')
parser.add_argument('--gpuid', default=0, type=int,
                    help='gpu id')
parser.add_argument('--datapath', default='./ImageNet2012/', type=str,
                    help='imagenet dateset path')
parser.add_argument('--val_testsize', default=63, type=int,
                    help='number of images to evaluate')
parser.add_argument('--batchsize', default=64, type=int,
                    help='batch size')
parser.add_argument('--nchannelbatch', default=128, type=int,
                    help='number of channels for each quantization batch')
parser.add_argument('--bitrangemin', default=0, type=int,
                    help='0 <= bitrangemin <= 10')
parser.add_argument('--bitrangemax', default=10, type=int,
                    help='0 <= bitrangemax <= 10')
parser.add_argument('--w_quant', action="store_true")
parser.add_argument('--a_quant', action="store_true")
parser.add_argument('--layer_id_toquant', '-l', type=int, nargs="+")
parser.add_argument('--mode', type=str, default="first", choices=["first", "specific", "consecutive_2", "fix_first", "consecutive_1"])
args = parser.parse_args()

if args.mode == "first":
    args.layer_id_toquant = 0

tranname = "idt"
trantype = "exter"

srcnet = loadnetwork(args.archname, args.gpuid)
tarnet = copy.deepcopy(srcnet)
images, labels = loadvaldata(args.datapath, args.gpuid, testsize=args.val_testsize)
tarnet, tarlayers = convert_qconv(tarnet, stats=False)
srcnet, srclayers = convert_qconv(srcnet, stats=False)
tarnet.eval()
srcnet.eval()
srchooks = hooklayers(srclayers)
if "vit" in args.archname:
    args.mean = [0.5,] * 3
    args.std = [0.5,] * 3
else:
    args.mean = IMAGENET_DEFAULT_MEAN
    args.std = IMAGENET_DEFAULT_STD
loader = get_val_imagenet_dali_loader(args)
Y, labels = predict_dali_withgt(srcnet, loader)
top_1, top_5 = accuracy(Y, labels, topk=(1,5))
print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.archname, top_1, top_5))
Y_norm = Y / ((Y**2).sum().sqrt())
fp_acts = [h.input_tensor for h in srchooks]

tardimens = hooklayers_with_fp_act(tarlayers, fp_acts)
dimens = [h.input for h in srchooks]

rd_rate, rd_rate_entropy, rd_dist, rd_phi, rd_delta, rd_delta_mse, rd_dist_mse = \
        load_rd_curve_batch(args.archname, srclayers, args.maxdeadzones, args.maxrates, args.pathrdcurve, args.nchannelbatch)



path_output = ('./%s_nr_%04d_ns_%04d_nf_%04d_accum_errors_%s' % (args.archname, args.maxrates, args.val_testsize, args.nchannelbatch, args.mode))
isExists=os.path.exists(path_output)
if not isExists:
    os.makedirs(path_output)


def quantize_layer(b, l):
    if args.a_quant:
        tarlayers[l].quantized = True
        acti_Y_sse, acti_delta, acti_coded = loadrdcurves(args.archname, l, 0, 'acti')
        # TODO: support grouped quantize for act
        best_j = np.argmin(acti_Y_sse[b, :, 0])
        acti_delta = [acti_delta[b, best_j, 0]]
        tarlayers[l].coded, tarlayers[l].delta = [(b-1) * dimens[l].prod()], acti_delta

    elif args.w_quant:
        layer_weights = srclayers[l].weight.clone()
        # import pdb; pdb.set_trace()
        pc_phi = [rd_phi[l][c][0, b] for c in range(len(rd_phi[l]))]
        pc_delta = [rd_delta[l][c][0, b] for c in range(len(rd_delta[l]))]
        pc_bits = [rd_rate[l][c][0, b] for c in range(len(rd_rate[l]))]
        
        nchannels = get_num_output_channels(layer_weights)
        n_channel_elements = get_ele_per_output_channel(layer_weights)
        quant_weights = tarlayers[l].weight.clone()

        cnt = 0
        for f in range(0, nchannels, args.nchannelbatch):
            st_layer = f
            ed_layer = f + args.nchannelbatch
            if f + args.nchannelbatch > nchannels:
                ed_layer = nchannels

            output_channels = get_output_channels(layer_weights, st_layer, ed_layer)
            quant_output_channels = deadzone_quantize(output_channels, pc_phi[cnt], 2**(pc_delta[cnt]), pc_bits[cnt] / output_channels.numel())
            quant_index_output_channels = deadzone_quantize_idx(output_channels, pc_phi[cnt], 2**(pc_delta[cnt]), pc_bits[cnt] / output_channels.numel())

            assign_output_channels(tarlayers[l].weight, st_layer, ed_layer, quant_output_channels)
            assign_output_channels(quant_weights, st_layer, ed_layer, quant_index_output_channels)
            cnt += 1

if args.mode in ["consecutive_1"]:
    hist_accum_errors = torch.ones(len(tarlayers), args.bitrangemax - args.bitrangemin, len(tarlayers), device=getdevice())
elif args.mode not in ["consecutive_2", "fix_first"]:
    hist_accum_errors = torch.ones(args.bitrangemax - args.bitrangemin, len(tarlayers), device=getdevice())
else:
    hist_accum_errors = torch.ones(len(tarlayers)-1, args.bitrangemax - args.bitrangemin, len(tarlayers), device=getdevice())

with torch.no_grad():
    for b in range(args.bitrangemin + 1, args.bitrangemax + 1):
        sec = time.time()
        if args.mode == "first":
            quantize_layer(b, args.layer_id_toquant)
            Y = predict_dali(tarnet, loader)
            top_1, top_5 = accuracy(Y, labels, topk=(1,5))
            print('rate: %d top1 %5.2f top5 %5.2f \t%2.2f s' % (b, top_1, top_5, time.time() - sec))
            
            for l_ in range(len(tardimens)):
                hist_accum_errors[b - 1, l_] = tardimens[l_].accum_err_act
        elif args.mode == "fix_first":
            quantize_layer(b, 0)
            for l in range(1, len(tarlayers)):
                quantize_layer(b, l)

                Y = predict_dali(tarnet, loader)
                top_1, top_5 = accuracy(Y, labels, topk=(1,5))
                print('rate: %d top1 %5.2f top5 %5.2f \t%2.2f s' % (b, top_1, top_5, time.time() - sec))
                
                for l_ in range(len(tardimens)):
                    hist_accum_errors[l - 1, b - 1, l_] = tardimens[l_].accum_err_act
                
                tarlayers[l].weight.data = srclayers[l].weight.clone()
                tarlayers[l].quantized = False
        elif args.mode == "consecutive_1":
            for l in range(0, len(tarlayers)):
                quantize_layer(b, l)

                Y = predict_dali(tarnet, loader)
                top_1, top_5 = accuracy(Y, labels, topk=(1,5))
                print('rate: %d top1 %5.2f top5 %5.2f \t%2.2f s' % (b, top_1, top_5, time.time() - sec))
                
                for l_ in range(len(tardimens)):
                    hist_accum_errors[l, b - 1, l_] = tardimens[l_].accum_err_act
                
                tarlayers[l].weight.data = srclayers[l].weight.clone()
                tarlayers[l].quantized = False
        else:
            for l in range(len(tarlayers) - 1):
                quantize_layer(b, l)
                quantize_layer(b, l+1)
                
                Y = predict_dali(tarnet, loader)
                top_1, top_5 = accuracy(Y, labels, topk=(1,5))
                print('rate: %d top1 %5.2f top5 %5.2f \t%2.2f s' % (b, top_1, top_5, time.time() - sec))
                
                for l_ in range(l, len(tardimens)):
                    hist_accum_errors[l, b - 1, l_] = tardimens[l_].accum_err_act
                
                tarlayers[l].weight.data = srclayers[l].weight.clone()
                tarlayers[l+1].weight.data = srclayers[l+1].weight.clone()
                tarlayers[l].quantized = False
                tarlayers[l+1].quantized = False



io.savemat('%s/%s_%s.mat' % (path_output, "w_quant" if args.w_quant else "a_quant", args.layer_id_toquant), {'hist_accum_errors':hist_accum_errors.cpu().numpy()})