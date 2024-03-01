import torch

# import resnetpy
from utils.models import resnet_cifar, efficientnet

# import vggpy
# import alexnetpy
# import densenetpy
# import mobilenetv2py
# import wideresnetpy
from . import algo
import torchvision.models as models
import scipy.io as io
import numpy as np
import time
import timm.models.vision_transformer as vit
import warnings
from pcdet.models.detectors.voxel_rcnn import VoxelRCNN
import spconv.pytorch as spconv
import torch.nn as nn
from utils.pruners_whole import get_weights, get_modules,get_all_modules,get_weights_2d
from pcdet.models.model_utils.flops_utils import calculate_gemm_flops
try:
    import kornia
except:
    pass
warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda:0")

rgb_avg = [0.5, 0.5, 0.5]  # [0.485, 0.456, 0.406]
rgb_std = [0.5, 0.5, 0.5]  # [0.229, 0.224, 0.225]
PARAMETRIZED_MODULE_TYPES = (
    torch.nn.Linear,
    torch.nn.Conv2d,
)
NORM_MODULE_TYPES = (torch.nn.BatchNorm2d, torch.nn.LayerNorm)


def loadnetwork(archname, gpuid, act_bitwidth=-1):
    global device
    device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")
    if archname in resnet_cifar.__all__:
        net = resnet_cifar.__dict__[archname](pretrained=True)

    return net.to(device)


def replaceconv(net, layers, includenorm=True):
    pushconv([layers], net, includenorm, direction=1)
    return net


def findconv(net, includenorm=True,compare=False):
    layers = pushconv([[]], net, includenorm,compare=compare)
    return layers


def getdevice():
    global device
    return device


def pushattr(layers, container, attr, includenorm, direction, prefix=""):
    if isinstance(getattr(container, attr, None), PARAMETRIZED_MODULE_TYPES) or (
        isinstance(getattr(container, attr, None), NORM_MODULE_TYPES) and includenorm
    ):
        # setattr(container,attr,TimeWrapper(getattr(container,attr), prefix))

        if direction == 0:
            layers[0].append(getattr(container, attr))
        else:
            setattr(container, attr, layers[0][0])
            layers[0] = layers[0][1 : len(layers[0])]
    # print(container.__class__.__name__, attr)


def pushlist(layers, container, attr, includenorm, direction, prefix=""):
    if isinstance(container[attr], PARAMETRIZED_MODULE_TYPES) or (
        isinstance(container[attr], NORM_MODULE_TYPES) and includenorm
    ):
        # container[attr] = TimeWrapper(container[attr], prefix)
        if direction == 0:
            layers[0].append(container[attr])
        else:
            container[attr] = layers[0][0]
            layers[0] = layers[0][1 : len(layers[0])]
    else:
        pushconv(layers, container[attr], includenorm, direction, prefix=prefix)


def pushconv(layers, container, includenorm=True, direction=0, prefix="model",compare=False):
    if compare:
        return get_all_modules(container)
    else:
        return get_modules(container)
    # if isinstance(container, models.densenet.DenseNet):
    #     pushconv(layers, container.features, includenorm, direction)
    #     pushattr(layers, container, "classifier", includenorm, direction)
    # elif isinstance(container, models.densenet._DenseBlock):
    #     for l in range(0, 25):
    #         if hasattr(container, "denselayer%d" % l):
    #             pushconv(layers, getattr(container, "denselayer%d" % l), includenorm, direction)
    # elif isinstance(container, models.densenet._DenseLayer):
    #     pushattr(layers, container, "conv1", includenorm, direction)
    #     pushattr(layers, container, "norm2", includenorm, direction)
    #     pushattr(layers, container, "conv2", includenorm, direction)
    # elif isinstance(container, models.densenet._Transition):
    #     pushattr(layers, container, "norm", includenorm, direction)
    #     pushattr(layers, container, "conv", includenorm, direction)
    # elif isinstance(container, (torch.nn.Sequential, torch.nn.ModuleList)):
    #     for attr in range(0, len(container)):
    #         pushlist(layers, container, attr, includenorm, direction, prefix=prefix + f".{attr}")

    # elif isinstance(container, resnet_cifar.ResNetCifar):
    #     pushattr(layers, container, "conv1", includenorm, direction)
    #     pushattr(layers, container, "bn1", includenorm, direction)
    #     pushconv(layers, container.layer1, includenorm, direction)
    #     pushconv(layers, container.layer2, includenorm, direction)
    #     pushconv(layers, container.layer3, includenorm, direction)
    #     pushattr(layers, container, "fc", includenorm, direction)
    # elif isinstance(container, resnet_cifar.BasicBlock):
    #     pushattr(layers, container, "conv1", includenorm, direction)
    #     pushattr(layers, container, "bn1", includenorm, direction)
    #     pushattr(layers, container, "conv2", includenorm, direction)
    #     pushattr(layers, container, "bn2", includenorm, direction)
    #     if container.downsample is not None:
    #         pushattr(layers, container.downsample, "0", includenorm, direction)
    #         pushattr(layers, container.downsample, "1", includenorm, direction)
    # # # Added for MAE
    # # elif isinstance(container, models_mae.PatchEmbed):
    # #     pushattr(layers, container, 'proj', includenorm, direction)
    # #     pushattr(layers, container, 'norm', includenorm, direction)
    # # elif isinstance(container, models_mae.Mlp):
    # #     pushattr(layers, container, 'fc1', includenorm, direction)
    # #     pushattr(layers, container, 'fc2', includenorm, direction)
    # # elif isinstance(container, models_mae.Block):
    # #     pushattr(layers, container, 'norm1', includenorm, direction)
    # #     pushattr(layers, container, 'attn', includenorm, direction)
    # #     pushattr(layers, container, 'norm2', includenorm, direction)
    # #     pushattr(layers, container, 'mlp', includenorm, direction)
    # # elif isinstance(container, models_mae.Attention):
    # #     pushattr(layers, container, 'qkv', includenorm, direction)
    # #     pushattr(layers, container, 'proj', includenorm, direction)
    # # elif isinstance(container, models_mae.MaskedAutoencoderViT):
    # #     pushlist(layers, container, 'blocks', includenorm, direction, prefix=prefix+f".blocks")
    # #     pushattr(layers, container, 'norm', includenorm, direction)
    # #     pushattr(layers, container, 'decoder_embed', includenorm, direction)
    # #     pushlist(layers, container, 'decoder_blocks', includenorm, direction, prefix=prefix+f".decoder_blocks")
    # #     pushattr(layers, container, 'decoder_norm', includenorm, direction)
    # #     pushattr(layers, container, 'decoder_pred', includenorm, direction)
    # elif isinstance(container, vit.PatchEmbed):
    #     pushattr(layers, container, "proj", includenorm, direction, prefix=prefix + ".proj")

    # elif isinstance(container, vit.Block):
    #     # pushattr(layers, container, "norm1", includenorm, direction)
    #     pushconv(layers, container.attn, includenorm, direction, prefix=prefix + ".attn")
    #     # pushattr(layers, container, "norm2", includenorm, direction)
    #     pushconv(layers, container.mlp, includenorm, direction, prefix=prefix + ".mlp")

    # elif isinstance(container, vit.Attention):
    #     pushattr(layers, container, "qkv", includenorm, direction, prefix=prefix + ".qkv")
    #     pushattr(layers, container, "proj", includenorm, direction, prefix=prefix + ".proj")

    # elif isinstance(container, vit.Mlp):
    #     pushattr(layers, container, "fc1", includenorm, direction, prefix=prefix + ".fc1")
    #     pushattr(layers, container, "fc2", includenorm, direction, prefix=prefix + ".fc2")
    # else:
    #     return [m for m in container.modules() if isinstance(m, PARAMETRIZED_MODULE_TYPES)]

    return layers[0]


def replacelayer(module, layers, classes):
    module_output = module
    # base case
    if isinstance(module, classes):
        module_output, layers[0] = layers[0][0], layers[0][1:]
    # recursive
    for name, child in module.named_children():
        module_output.add_module(name, replacelayer(child, layers, classes))
    del module
    return module_output


def loadvarstats(archname, testsize):
    mat = io.loadmat(("%s_stats_%d.mat" % (archname, testsize)))
    return np.array(mat["cov"])


# def loadrdcurves(archname,l,g,part,nchannelbatch=40000):
#     mat = io.loadmat(f'{archname}_nf_{nchannelbatch:04d}_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_{g:04d}_output_{part}')
#     return mat['%s_Y_sse'%part], mat['%s_delta'%part], mat['%s_coded'%part]


def findrdpoints(y_sse, delta, coded, lam_or_bit, is_bit=False):
    # find the optimal quant step-size
    y_sse[np.isnan(y_sse)] = float("inf")
    ind1 = np.nanargmin(y_sse, 1)
    ind0 = np.arange(ind1.shape[0]).reshape(-1, 1).repeat(ind1.shape[1], 1)
    ind2 = np.arange(ind1.shape[1]).reshape(1, -1).repeat(ind1.shape[0], 0)
    inds = np.ravel_multi_index((ind0, ind1, ind2), y_sse.shape)  # bit_depth x blocks
    y_sse = y_sse.reshape(-1)[inds]
    delta = delta.reshape(-1)[inds]
    coded = coded.reshape(-1)[inds]
    # mean = mean.reshape(-1)[inds]
    # find the minimum Lagrangian cost
    if is_bit:
        point = coded == lam_or_bit
    else:
        point = y_sse + lam_or_bit * coded == (y_sse + lam_or_bit * coded).min(0)
    return np.select(point, y_sse), np.select(point, delta), np.select(point, coded)  # , np.select(point, mean)


# import pickle as pkl
# def loadmeanstd(archname, l, part):
#     with open(f'{archname}_nr_0011_ns_0064_rdcurves_channelwise_opt_dist_act/{archname}_val_{l:03d}_0064_output_{part}_meanstd.pkl', 'rb') as f:
#         d = pkl.load(f)
#     return d


def predict(net, images, batch_size=256, num_workers=16):
    global device
    y_hat = torch.zeros(0, device=device)
    loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y_hat = torch.cat((y_hat, net(x)))
            if len(y_hat) >= calib_size:
                break
    return y_hat


def predict2(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in iter(loader):
            x = x.to(device)
            y_hat = torch.cat((y_hat, net(x)))
    return y_hat

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

def predict2_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for batch_dict in iter(loader):
            load_data_to_gpu(batch_dict)
            # y = y.to(device).float()
            pred_dicts, ret_dict=net(batch_dict)
            y_hat = torch.cat((y_hat, pred_dicts))
            y_gt = torch.cat((y_gt, ret_dict))
    return y_hat, y_gt


def predict2_activation(net, loader, layerhook):
    global device
    acts = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for x, _ in iter(loader):
            x = x.to(device)
            _ = net(x)
            acts = torch.cat((acts, layerhook.output_tensor))
    return acts


def predict_dali(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            res = net(x)
            y_hat = torch.cat((y_hat, res))
    return y_hat


def predict_dali_withgt(net, loader):
    global device
    y_hat = torch.zeros(0, device=device)
    y_gt = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            y_hat = torch.cat((y_hat, net(x)))
            y = data[0]["label"]
            y_gt = torch.cat((y_gt, y))
    return y_hat, y_gt


def predict_dali_activation(net, loader, layerhook):
    global device
    acts = torch.zeros(0, device=device)
    # loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for data in loader:
            x = data[0]["data"]
            _ = net(x)
            acts = torch.cat((acts, layerhook.output_tensor))
    return acts


import math


@torch.no_grad()
def predict_tensor(net, X, batchsize=128):
    global device
    y_hat = torch.zeros(0, device=device)
    for b in range(math.ceil(len(X) / batchsize)):
        y_hat = torch.cat((y_hat, net(X[b * batchsize : (b + 1) * batchsize])))
    return y_hat


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # pred.reshape(pred.shape[0], -1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def ternary_search(min_val, max_val, target_func, max_iters=20):
    l = min_val
    r = max_val
    cnt = 0
    while l < r:
        y_l = target_func(l)
        y_r = target_func(r)
        if y_l > y_r:
            l = l + (r - l) / 3
        else:
            r = r - (r - l) / 3

        cnt += 1
        if cnt >= max_iters:
            break
    return l


def binary_search(min_val, max_val, target_func, target_val, epsilon=0.02, max_iters=40):
    l = min_val
    r = max_val
    cnt = 0
    while l < r:
        mid = (l + r) / 2
        y_mid = target_func(mid)

        if abs(y_mid - target_val) <= epsilon:
            return mid
        elif y_mid < target_val:
            l = mid
        elif y_mid > target_val:
            r = mid

        cnt += 1
        if cnt >= max_iters:
            y_l = target_func(l)
            y_r = target_func(r)
            if abs(y_mid - target_val) > abs(y_l - target_val) and abs(y_r - target_val) > abs(y_l - target_val):
                mid = l
            elif abs(y_mid - target_val) > abs(y_r - target_val) and abs(y_l - target_val) > abs(y_r - target_val):
                mid = r
            break
    return mid


from functools import partial


def find_slope(
    model, target_sparsity, rd_dist, rd_amount, layers=None, prune_mode="unstructured", flop_budget=False, **kwargs
):
    layers = layers or findconv(model, False)
    if flop_budget:
        layer_weights = [layer.weight.clone() for layer in layers]

    def target_func(slope):
        if not flop_budget:
            total_n_weights = 0
            survived_n_weights = 0
        else:
            layer_weights = [layer.weight.clone() for layer in layers]
        pc_amount = algo.pareto_condition(layers, rd_dist, rd_amount, 2**slope, min_sp=kwargs["min_sp"])
        # print(pc_amount, slope)

        for i in range(0, len(layers)):
            prune_weights = algo.pruning(
                layers[i].weight.clone(),
                pc_amount[i][0],
                mode=prune_mode,
                rank=kwargs.get("rank", "l1"),
                grad=kwargs["grad"][i] if kwargs.get("rank", "l1") == "taylor" else None,
            )
            if not flop_budget:
                total_n_weights += prune_weights.numel()
                survived_n_weights += (prune_weights != 0).sum().float()
            else:
                layers[i].weight.data = prune_weights

        if flop_budget:
            flops = get_model_flops(model, kwargs["dataset"])
            for layer, ori_weight in zip(layers, layer_weights):
                layer.weight.data = ori_weight
            print(slope, flops)
            return -flops
        ret = 1 - survived_n_weights / total_n_weights
        return ret

    return binary_search(-1000, 1000, target_func, -target_sparsity if flop_budget else target_sparsity)


def prune_layerbylayer(model, layer_id, amount):
    layers = findconv(model, False)
    layers[layer_id].amount = amount


def hooklayers(layers):
    return [Hook(layer) for layer in layers]


class Hook:
    def __init__(self, module, backward=False):
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if isinstance(module,spconv.SubMConv3d) or isinstance(module,spconv.SparseConv3d):
            self.input_tensor = input[0]
            # self.input = torch.tensor(input[0].features.shape)
            self.output = output
            self.input = torch.tensor(input[0].features.shape)
            # self.output = torch.tensor(output.features.shape)
            # print('sparse',input[0].features.shape,output.features.shape)
            self.input_tensor = input
            self.output_tensor = output
        else:
            self.input_tensor = input[0]
            self.input = torch.tensor(input[0].shape[1:])
            self.output = torch.tensor(output[0].shape[1:])
            # print('output[0].shape[1:]',input[0].shape,output[0].shape)
            self.input_tensor = input[0]
            self.output_tensor = output[0]

            

    def close(self):
        self.hook.remove()

@torch.no_grad()
def _count_total_weights_2d_3d(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    wlist_2d=get_weights_2d(model)
    numels2d=[w2.numel() for w2 in wlist_2d]
    numels3d=[w3.numel() for w3 in wlist]
    return torch.FloatTensor(numels2d),torch.FloatTensor(numels3d)

@torch.no_grad()
def _count_total_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist_3d,mlist_2d = get_all_modules(model)
    unmaskeds_3d = []
    unmaskeds_2d=[]
    for m in mlist_3d:
        unmaskeds_3d.append(m.weight_mask.count_nonzero())
    for m in mlist_2d:
        unmaskeds_2d.append(m.weight_mask.count_nonzero())
    return torch.FloatTensor(unmaskeds_2d),torch.FloatTensor(unmaskeds_3d)


@torch.no_grad()
def get_model_flops(net, dataloader):
    net.cuda()
    net.eval()
    total_denom_flops=[]
    total_nom_flops=[]
    total_denom_flops_3d=[]
    total_nom_flops_3d=[]
    total_denom_flops_2d=[]
    total_nom_flops_2d=[]
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
    # if dataset == "cifar":
    #     dummy_input = torch.zeros((1, 3, 32, 32), device=next(net.parameters()).device)
    # else:
    #     dummy_input = torch.zeros((1, 3, 224, 224), device=next(net.parameters()).device)

        layers = findconv(net, False,True)
        layers=layers[0]+layers[1]
       
        unmaskeds_2d,unmaskeds_3d = _count_total_unmasked_weights(net)
        unmaskeds = torch.cat((unmaskeds_3d, unmaskeds_2d))

        totals2d,totals_3d=_count_total_weights_2d_3d(net)
        totals=torch.cat((totals_3d, totals2d))

        hookedlayers = hooklayers(layers)

        _,_= net(batch_dict)

        fil = [hasattr(h, "output") for h in hookedlayers]
        if False in fil:
            layers = [layers[i] for i in range(len(layers)) if fil[i]]
            hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]
            unmaskeds = [unmaskeds[i] for i in range(len(unmaskeds)) if fil[i]]
            totals = [totals[i] for i in range(len(totals)) if fil[i]]

        output_dimens = [hookedlayers[i].output for i in range(0, len(hookedlayers))]
        input_dimens = [hookedlayers[i].input for i in range(0, len(hookedlayers))]
        for l in hookedlayers:
            l.close()
        denom_flops=0.0
        nom_flops=0.0
        denom_flops_2d = 0.0
        nom_flops_2d = 0.0
        denom_flops_3d = 0.0
        nom_flops_3d = 0.0

        # new_input_dimens = input_dimens[1:11] + input_dimens[12:]
        # new_output_dimens = output_dimens[1:11] + output_dimens[12:]
        # new_unmaskeds = torch.cat((unmaskeds[1:11] , unmaskeds[12:]))
        # new_totals = torch.cat((totals[1:11] , totals[12:]))
        # new_layers = layers[1:11] + layers[12:]
        # for i_dim,o_dim, surv, tot, m in zip(new_input_dimens,new_output_dimens, new_unmaskeds, new_totals, new_layers):
        for i_dim,o_dim, surv, tot, m in zip(input_dimens,output_dimens, unmaskeds, totals, layers):
            if isinstance(m,spconv.SubMConv3d) or isinstance(m,spconv.SparseConv3d):
                # print(o_dim.indice_dict.keys())
                unique_key = next(iter(o_dim.indice_dict.keys()))
                flops=calculate_gemm_flops(o_dim,unique_key,i_dim[-1],o_dim.features.shape[-1])
                denom_flops_3d +=flops
                denom_flops+=flops
                nom_flops += flops  * surv /tot
                nom_flops_3d += flops  * surv /tot 
            elif isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                denom_flops+=(o_dim[-2:].prod() * tot + (0 if m.bias is None else o_dim.prod()))
                nom_flops += (o_dim[-2:].prod() * surv + (0 if m.bias is None else o_dim.prod()))
                denom_flops_2d += (o_dim[-2:].prod() * tot + (0 if m.bias is None else o_dim.prod()))
                nom_flops_2d += (o_dim[-2:].prod() * surv + (0 if m.bias is None else o_dim.prod()))
            elif isinstance(m, torch.nn.Linear):
                denom_flops+=(tot + (0 if m.bias is None else o_dim.prod()))
                nom_flops += (surv + (0 if m.bias is None else o_dim.prod()))
                denom_flops_2d += (tot + (0 if m.bias is None else o_dim.prod()))
                nom_flops_2d += (surv + (0 if m.bias is None else o_dim.prod()))
    

        # lin_modules = [m for m in net.modules() if isinstance(m, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.AdaptiveAvgPool2d,
        #                                                         efficientnet.ElementMul, efficientnet.ElementAdd))]
        # if len(lin_modules)>0:
        #     hookedlayers = hooklayers(lin_modules)
        #     _,_= net(batch_dict)
        #     fil = [hasattr(h, "output") for h in hookedlayers]

        #     if False in fil:
        #         lin_modules = [lin_modules[i] for i in range(len(lin_modules)) if fil[i]]
        #         hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]

        #     output_dimens = [hookedlayers[i].output for i in range(0,len(lin_modules))]
        #     for l in hookedlayers:
        #         l.close()

        #     for o_dim, m in zip(output_dimens, lin_modules):
        #         denom_flops += o_dim.prod() + int(isinstance(m, torch.nn.AdaptiveAvgPool2d))
        #         nom_flops += o_dim.prod() + int(isinstance(m, torch.nn.AdaptiveAvgPool2d))

    # att_modules = [m for m in net.modules() if isinstance(m, (vit.Attention))]
    # hookedlayers = hooklayers(att_modules)
    # _ = net(dummy_input)
    # fil = [hasattr(h, "output") for h in hookedlayers]
    # if False in fil:
    #     att_modules = [att_modules[i] for i in range(len(att_modules)) if fil[i]]
    #     hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]

    # input_dimens = [hookedlayers[i].input for i in range(0,len(att_modules))]
    # for l in hookedlayers:
    #     l.close()

    # for i_dim, m in zip(input_dimens, att_modules):
    #     P, C = i_dim
    #     H = m.num_heads
    #     C_h = C // H
    #     D = m.qkv.out_features
    #     # flop = P*C*(3*D + C_h + 1)
    #     flop = 4 * C * C_h * P
    #     denom_flops += flop
    #     nom_flops += flop

        # pool_modules = [m for m in net.modules() if isinstance(m, (torch.nn.MaxPool2d, torch.nn.AvgPool2d))]
        # if len(pool_modules)>0:
        #     hookedlayers = hooklayers(pool_modules)
        #     _ = net(batch_dict)
        #     fil = [hasattr(h, "output") for h in hookedlayers]
        #     if False in fil:
        #         pool_modules = [pool_modules[i] for i in range(len(pool_modules)) if fil[i]]
        #         hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]

        #     output_dimens = [hookedlayers[i].output for i in range(0,len(pool_modules))]
        #     for l in hookedlayers:
        #         l.close()

        #     for o_dim, m in zip(output_dimens, pool_modules):
        #         denom_flops += o_dim.prod() * (m.kernel_size[0] * m.kernel_size[1] - int(isinstance(m, torch.nn.Maxpool)))
        #         nom_flops += o_dim.prod() * (m.kernel_size[0] * m.kernel_size[1] - int(isinstance(m, torch.nn.Maxpool)))

        # bn_modules = [m for m in net.modules() if isinstance(m, (torch.nn.BatchNorm2d,torch.nn.BatchNorm1d, torch.nn.LayerNorm))]
        # if len(bn_modules)>0:
        #     hookedlayers = hooklayers(bn_modules)
        #     _ = net(batch_dict)
        #     fil = [hasattr(h, "output") for h in hookedlayers]
        #     if False in fil:
        #         bn_modules = [bn_modules[i] for i in range(len(bn_modules)) if fil[i]]
        #         hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]

        #     output_dimens = [hookedlayers[i].output for i in range(0,len(bn_modules))]
        #     for l in hookedlayers:
        #         l.close()

        #     for o_dim, m in zip(output_dimens, bn_modules):
        #         denom_flops += o_dim.prod() * (4 if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m,torch.nn.BatchNorm1d) else 10)
        #         nom_flops += o_dim.prod() * (4 if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m,torch.nn.BatchNorm1d) else 10)
            
        #     sm_modules = [m for m in net.modules() if isinstance(m, (torch.nn.Softmax))]
        #     if len(sm_modules)>0:
        #         hookedlayers = hooklayers(sm_modules)
        #         _ = net(batch_dict)
        #         fil = [hasattr(h, "output") for h in hookedlayers]
        #         if False in fil:
        #             sm_modules = [sm_modules[i] for i in range(len(sm_modules)) if fil[i]]
        #             hookedlayers = [hookedlayers[i] for i in range(len(hookedlayers)) if fil[i]]

        #         output_dimens = [hookedlayers[i].output for i in range(0,len(sm_modules))]
        #         for l in hookedlayers:
        #             l.close()

        #         for o_dim, m in zip(output_dimens, sm_modules):
        #             denom_flops += o_dim.prod() * 2 - 1
        #             nom_flops += o_dim.prod() * 2 - 1
            total_denom_flops.append(denom_flops)
            total_nom_flops.append(nom_flops)
            total_denom_flops_3d.append(denom_flops_3d)
            total_nom_flops_3d.append(nom_flops_3d)
            total_denom_flops_2d.append(denom_flops_2d)
            total_nom_flops_2d.append(nom_flops_2d)
    print(sum(total_denom_flops)/len(dataloader),sum(total_nom_flops)/len(dataloader),sum(total_denom_flops_3d)/len(dataloader))

    return (sum(total_nom_flops)/len(dataloader)) / (sum(total_denom_flops)/len(dataloader)),sum(total_nom_flops_3d)/len(dataloader),sum(total_denom_flops_3d)/len(dataloader),sum(total_nom_flops_2d)/len(dataloader),sum(total_denom_flops_2d)/len(dataloader)
