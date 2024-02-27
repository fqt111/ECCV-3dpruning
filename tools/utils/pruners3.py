import torch
from torch.nn.utils import prune
from tools.utils import get_weights, get_modules, get_model_sparsity
import numpy as np
import tools.common as common
import tools.algo as algo
import time
import os
import scipy.io as io
from itertools import product
from functools import partial
import einops
from tools.torch_pruner import struct_identity
import pickle

def weight_pruner_loader(pruner_string):
    """
    Gives you the pruning methods: LAMP, Glob, Unif, Unif+, and ERK
    """
    if pruner_string == 'lamp':
        return prune_weights_lamp
    elif pruner_string == 'glob':
        return prune_weights_global
    elif pruner_string == 'unif':
        return prune_weights_uniform
    elif pruner_string == 'unifplus':
        return prune_weights_unifplus
    elif pruner_string == 'erk':
        return prune_weights_erk
    elif pruner_string == 'rd':
        return RDPruner()
    elif pruner_string == "knapsack":
        return KnapsackPruner()
    else:
        raise ValueError('Unknown pruner')

"""
prune_weights_reparam: Allocate identity mask to every weight tensors.
prune_weights_l1predefined: Perform layerwise pruning w.r.t. given amounts.
"""

def prune_weights_soft(model, amounts, alpha):
    module_list = get_modules(model)
    for idx,m in enumerate(module_list):
        mask = algo.get_mask(m.weight if not hasattr(m, "weight_ori") else m.weight_ori, float(amounts[idx])).float()
        if not hasattr(m, "weight_ori"):
            m.weight_ori = m.weight.data.clone()
        soft_mask = mask + alpha * (1-mask)
        m.weight.data = m.weight_ori * soft_mask


def tune_weights_gradients_pg(model, amounts, beta, gm_dropout=0.):
    module_list = get_modules(model)
    for idx,m in enumerate(module_list):
        mask = algo.get_mask(m.weight if not hasattr(m, "weight_ori") else m.weight_ori, float(amounts[idx])).float()
        soft_mask = mask + beta * (1 - mask)
        if gm_dropout > 0:
            soft_mask *= torch.randint_like(soft_mask, 0, 2)
        if m.weight.grad is not None:
            m.weight.grad.data *= soft_mask

def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m,name="weight")
        
        
def prune_weights_reparam_struct(model):
    module_list = get_modules(model)
    for m in module_list:
        struct_identity(m,name="weight")


def prune_weights_l1predefined(model,amounts, only_layerids=None):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        if only_layerids is not None and idx not in only_layerids: 
            continue
        prune.l1_unstructured(m,name="weight",amount=float(amounts[idx]))
        
        
def prune_weights_l1structured(model,amounts, only_layerids=None):
    mlist = get_modules(model)
    for idx,m in enumerate(mlist):
        if only_layerids is not None and idx not in only_layerids: 
            continue
        prune.ln_structured(m,name="weight",amount=float(amounts[idx]), n=1, dim=1)
        
"""
Methods: All weights
"""    

def prune_weights_global(model,amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune,pruning_method = prune.L1Unstructured,amount=amount)

def prune_weights_lamp(model,amount):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model,amount)
    print(amounts)
    prune_weights_l1predefined(model,amounts)

def prune_weights_uniform(model,amount):
    module_list = get_modules(model)
    assert amount <= 1 # Can be updated later to handle > 1.
    for m in module_list:
        prune.l1_unstructured(m,name="weight",amount=amount)

def prune_weights_unifplus(model,amount):
    assert amount <= 1
    amounts = _compute_unifplus_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

def prune_weights_erk(model,amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model,amount)
    prune_weights_l1predefined(model,amounts)

def prune_weights_rd(model, amount, *args, **kwargs):
    assert amount <= 1
    amounts = _compute_rd_amounts(model, amount, *args, **kwargs)
    print(amounts)
    prune_weights_l1predefined(model,amounts)

def prune_weights_rd_pg(model, amount, *args, **kwargs):
    assert amount <= 1
    amounts = _compute_rd_amounts(model, amount, *args, **kwargs)
    print(amounts)
    prune_weights_soft(model,amounts)

"""
These are not intended to be exported.
"""

def _extract_weight_tuples(model):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    return tuple([(m,'weight') for m in mlist])


def gen_rd_curves(net, loader, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = ('./%s_ndz_%04d_rdcurves_channelwise_opt_dist' % (args.model, args.maxdeadzones))
    else:
        path_output = ('%s/%s_ndz_%04d_rdcurves_channelwise_opt_dist/%s/' % (prefix, args.model, args.maxdeadzones, suffix))
    
    layers = common.findconv(net, False)
    hookedlayers = common.hooklayers(layers)
    try:
        dummy_input = next(iter(loader))[0]['data'].cuda()
    except:
        dummy_input = next(iter(loader))[0].cuda()
    
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
    for l in hookedlayers:
        l.close()
        
    print('total number of layers: %d' % (len(layers)))
    print(f'saving to {path_output}')
    isExists=os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        return algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
#     Y, labels = common.predict2_withgt(net, loader, calib_size=256)
    try:
        Y, labels = common.predict2_withgt(net, loader)
    except:
        Y, labels = common.predict_dali_withgt(net, loader)
        print(Y.shape)

    top_1, top_5 = common.accuracy(Y, labels, topk=(1,5))
    print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.model, top_1, top_5))

    rd_dists = []
    rd_amounts = []
    rd_dist_mses = []
    
    if not hasattr(args, "num_parts"):
        args.num_parts = 1
        args.part_id = 0
    len_part = len(layers) // args.num_parts

    with torch.no_grad():
        for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxdeadzones).cuda()
            rst_dist = torch.ones(args.maxdeadzones).cuda()
            rst_dist_mse = torch.ones(args.maxdeadzones).cuda()

            end = time.time()

            min_dist = 1e8
            min_mse = 1e8
            pre_mse = 1e8
            pre_dist = 1e8

            min_amount = 0 if not args.change_curve_scale else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())

            for d in range(args.maxdeadzones):
                amount = (1. - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[args.maxdeadzones-d-1] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount)
                
                cur_mse = ((prune_weights - layers[layerid].weight)**2).mean()
                layers[layerid].weight.data = prune_weights
                
#                 Y_hat = common.predict2(net, loader, calib_size=256)
                try:
                    Y_hat = common.predict2(net, loader)
                except:
                    Y_hat = common.predict_dali(net, loader)
                    
                if args.worst_case_curve:
                    cur_dist = ((Y - Y_hat) ** 2).mean(dim=1).max()
                else:
                    cur_dist = ((Y - Y_hat) ** 2).mean()
                
                top_1, _ = common.accuracy(Y_hat, labels, topk=(1, 5))
                # print('%s | layer %d: amount %6.6f mse %6.6f distortion %6.6f top1 %.2f | time %f' \
                #     % (args.model, layerid, amount, cur_mse, cur_dist, top_1, time.time() - end))
                end = time.time()
                # if (cur_dist < min_dist):
                # if amount > 0.9:
                #     import pdb; pdb.set_trace()
                rst_dist[args.maxdeadzones-d-1] = cur_dist
                min_dist = cur_dist
                if (cur_mse < min_mse):
                    rst_dist_mse[args.maxdeadzones-d-1] = cur_mse
                    min_mse = cur_mse

                pre_mse = cur_mse
                pre_dist = cur_dist
                layers[layerid].weight.data = layer_weights

            if args.smooth_curve:
                # import pdb; pdb.set_trace()
                rst_dist, rst_amount = algo.refine_curve(rst_dist, rst_amount)

            io.savemat(('%s/%s_%03d.mat' % (path_output, args.model, layerid)),
                {'rd_amount': rst_amount.cpu().numpy(), 'rd_dist': rst_dist.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()})

            rd_dists.append(rst_dist)
            rd_amounts.append(rst_amount)
            rd_dist_mses.append(rst_dist_mse)  
    
    rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
    return rd_dists, rd_amounts

def gen_rd_curves_singlelayer(net, loader, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = ('./%s_ndz_%04d_rdcurves_channelwise_opt_singlelayerdist' % (args.model, args.maxdeadzones))
    else:
        path_output = ('%s/%s_ndz_%04d_rdcurves_channelwise_opt_singlelayerdist/%s/' % (prefix, args.model, args.maxdeadzones, suffix))
    
    layers = common.findconv(net, False)
    hookedlayers = common.hooklayers(layers)
    dummy_input = next(iter(loader))[0]['data'].cuda()
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
    for l in hookedlayers:
        l.close()
        
    print('total number of layers: %d' % (len(layers)))
    print(f'saving to {path_output}')
    isExists=os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        return algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
#     Y, labels = common.predict2_withgt(net, loader, calib_size=256)
    try:
        Y, labels = common.predict2_withgt(net, loader, calib_size=args.calib_size)
    except:
        Y, labels = common.predict_dali_withgt(net, loader)

    top_1, top_5 = common.accuracy(Y, labels, topk=(1,5))
    print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.model, top_1, top_5))

    rd_dists = []
    rd_amounts = []
    rd_dist_mses = []
    
    if not hasattr(args, "num_parts"):
        args.num_parts = 1
        args.part_id = 0
    len_part = len(layers) // args.num_parts

    with torch.no_grad():
        for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
            hook = common.Hook(layers[layerid])
            try:
                A = common.predict2_activation(net, loader, hook, calib_size=args.calib_size)
            except:
                A = common.predict_dali_activation(net, loader, hook)
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxdeadzones).cuda()
            rst_dist = torch.ones(args.maxdeadzones).cuda()
            rst_dist_mse = torch.ones(args.maxdeadzones).cuda()

            end = time.time()

            min_dist = 1e8
            min_mse = 1e8
            pre_mse = 1e8
            pre_dist = 1e8

            min_amount = 0 if not args.change_curve_scale else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())

            for d in range(args.maxdeadzones):
                end = time.time()
                
                amount = (1. - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[args.maxdeadzones-d-1] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount)
                
                cur_mse = ((prune_weights - layers[layerid].weight)**2).mean()
                layers[layerid].weight.data = prune_weights
                
#                 Y_hat = common.predict2(net, loader, calib_size=256)
                try:
                    A_hat = common.predict2_activation(net, loader, hook, calib_size=args.calib_size)
                except:
                    A_hat = common.predict_dali_activation(net, loader, hook)

                if args.worst_case_curve:
                    cur_dist = ((A - A_hat) ** 2).view(A.shape[0], -1).mean(dim=1).max()
                else:
                    cur_dist = ((A - A_hat) ** 2).mean()
                
                # top_1, _ = common.accuracy(Y_hat, labels, topk=(1, 5))
                print('%s | layer %d: amount %6.6f mse %6.6f distortion %6.6f | time %f' \
                    % (args.model, layerid, amount, cur_mse, cur_dist, time.time() - end))
                # if (cur_dist < min_dist):
                # if amount > 0.9:
                #     import pdb; pdb.set_trace()
                rst_dist[args.maxdeadzones-d-1] = cur_dist
                min_dist = cur_dist
                if (cur_mse < min_mse):
                    rst_dist_mse[args.maxdeadzones-d-1] = cur_mse
                    min_mse = cur_mse

                pre_mse = cur_mse
                pre_dist = cur_dist
                layers[layerid].weight.data = layer_weights

            if args.smooth_curve:
                # import pdb; pdb.set_trace()
                rst_dist, rst_amount = algo.refine_curve(rst_dist, rst_amount)

            io.savemat(('%s/%s_%03d.mat' % (path_output, args.model, layerid)),
                {'rd_amount': rst_amount.cpu().numpy(), 'rd_dist': rst_dist.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()})

            rd_dists.append(rst_dist)
            rd_amounts.append(rst_amount)
            rd_dist_mses.append(rst_dist_mse)  
            hook.close()
    
    rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
    return rd_dists, rd_amounts

def gen_rd_curves_synth_data(net, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = ('./%s_ndz_%04d_rdcurves_channelwise_opt_dist' % (args.model, args.maxdeadzones))
    else:
        path_output = ('%s/%s_ndz_%04d_rdcurves_channelwise_opt_dist/%s/' % (prefix, args.model, args.maxdeadzones, suffix))
    
    layers = common.findconv(net, False)
    print('total number of layers: %d' % (len(layers)))
    print(f'saving to {path_output}')
    isExists=os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        return algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
    X = torch.normal(torch.zeros(1024, 3, 224, 224).cuda(), torch.ones(1024, 3, 224, 224).cuda())
    Y = common.predict_tensor(net, X, 256)

    rd_dists = []
    rd_amounts = []
    rd_dist_mses = []
    
    if not hasattr(args, "num_parts"):
        args.num_parts = 1
        args.part_id = 0
    len_part = len(layers) // args.num_parts

    with torch.no_grad():
        for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxdeadzones).cuda()
            rst_dist = torch.ones(args.maxdeadzones).cuda()
            rst_dist_mse = torch.ones(args.maxdeadzones).cuda()

            min_dist = 1e8
            min_mse = 1e8
            pre_mse = 1e8
            pre_dist = 1e8

            min_amount = 0 if not args.change_curve_scale else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())

            for d in range(args.maxdeadzones):
                amount = (1. - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[args.maxdeadzones-d-1] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount)
                
                cur_mse = ((prune_weights - layers[layerid].weight)**2).mean()
                layers[layerid].weight.data = prune_weights
                
                Y_hat = common.predict_tensor(net, X, 256)
                if args.worst_case_curve:
                    cur_dist = ((Y - Y_hat) ** 2).mean(dim=1).max()
                else:
                    cur_dist = ((Y - Y_hat) ** 2).mean()
                
                rst_dist[args.maxdeadzones-d-1] = cur_dist
                min_dist = cur_dist
                if (cur_mse < min_mse):
                    rst_dist_mse[args.maxdeadzones-d-1] = cur_mse
                    min_mse = cur_mse

                pre_mse = cur_mse
                pre_dist = cur_dist
                layers[layerid].weight.data = layer_weights

            if args.smooth_curve:
                # import pdb; pdb.set_trace()
                rst_dist, rst_amount = algo.refine_curve(rst_dist, rst_amount)

            io.savemat(('%s/%s_%03d.mat' % (path_output, args.model, layerid)),
                {'rd_amount': rst_amount.cpu().numpy(), 'rd_dist': rst_dist.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()})

            rd_dists.append(rst_dist)
            rd_amounts.append(rst_amount)
            rd_dist_mses.append(rst_dist_mse)  
    
    rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
    return rd_dists, rd_amounts


# def hessian_fischer_approx(gw, damp_lamda=1e-7, blocksize=128):
#     flat_gw = gw.view(-1)
#     if blocksize == -1:
#         return damp_lamda * torch.eye(len(flat_gw),device="cuda") + torch.outer(flat_gw, flat_gw)
#     ret = damp_lamda * torch.eye(len(flat_gw), device="cuda")
#     for i in range(len(flat_gw)//blocksize):
#         gw_i = flat_gw[blocksize*i:blocksize*(i+1)]
#         for j in range(len(flat_gw)//blocksize):
#             if i < j: continue
#             gw_j = flat_gw[blocksize*j:blocksize*(j+1)]
#             ret[blocksize*i:blocksize*(i+1), blocksize*j:blocksize*(j+1)] = torch.outer(gw_i, gw_j)
#             ret[blocksize*j:blocksize*(j+1), blocksize*i:blocksize*(i+1)] = ret[blocksize*i:blocksize*(i+1), blocksize*j:blocksize*(j+1)].T
#     return ret

@torch.no_grad()
def taylor_2nd_order_fischer_approx(delta_weight, gw, damp_lamda=1e-7, blocksize=128):
    flat_delta_w = delta_weight.view(-1)
    flat_gw = gw.view(-1)
    c = damp_lamda * torch.eye(blocksize,device="cuda")
    ret = torch.zeros([1],device="cuda")
    for i in range(len(flat_delta_w)//blocksize):
        dw_i = flat_delta_w[blocksize*i:blocksize*(i+1)]
        gw_i = flat_gw[blocksize*i:blocksize*(i+1)]
        for j in range(len(flat_delta_w)//blocksize):
            if i < j: continue
            dw_j = flat_delta_w[blocksize*j:blocksize*(j+1)]
            gw_j = flat_gw[blocksize*j:blocksize*(j+1)]
            ret += ((0.5 if i == j else 1) * dw_i.view(1, -1) @ ((c if i == j else 0) + torch.outer(gw_i, gw_j)) @ dw_j).squeeze()
    return ret.item()

@torch.no_grad()
def hessian_deltaw(delta_delta_weight, gw, damp_lamda=1e-7, blocksize=128):
    flat_delta_w = delta_delta_weight.view(-1)
    nonzero_idx = flat_delta_w.nonzero().squeeze()
    # breakpoint()
    ret = gw.view(-1, 1) @ (gw.view(-1)[nonzero_idx].view(1, -1) @ flat_delta_w[nonzero_idx])
    ret[nonzero_idx] += damp_lamda
    return ret


def gen_rd_curves_approx(net, loader, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = ('./%s_ndz_%04d_rdcurves_opt_dist' % (args.model, args.maxdeadzones))
    else:
        path_output = ('%s/%s_ndz_%04d_rdcurves_opt_dist/%s/' % (prefix, args.model, args.maxdeadzones, suffix))
    
    layers = common.findconv(net, False)
    hookedlayers = common.hooklayers(layers)
    dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
    for l in hookedlayers:
        l.close()
        
    
    grad_list = []
    net.train()
    mlist = get_modules(net)
    for c, data in enumerate(loader):
        try:
            x = data[0]["data"]
            y = data[0]["label"]
        except:
            x, y = data
            x = x.cuda()
            y = y.cuda()
        # res = torch.mean((net(x).max(1)[0] - y) ** 2) 
        res = torch.mean(net(x) ** 2) 
        res.backward()
        for idx,m in enumerate(mlist):
            if len(grad_list) < len(mlist):
                grad_list.append(m.weight.grad.data / len(loader))
            else:
                grad_list[idx] += m.weight.grad.data / len(loader)
        for p in net.parameters():
            if p.grad is not None:
                torch.nn.init.zeros_(p.grad.data)

    print('total number of layers: %d' % (len(layers)))
    print(f'saving to {path_output}')
    isExists=os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
        return rd_dists, rd_amounts, grad_list

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
    rd_dists = []
    rd_amounts = []
    rd_dist_mses = []
    
    if not hasattr(args, "num_parts"):
        args.num_parts = 1
        args.part_id = 0
    len_part = len(layers) // args.num_parts

    with torch.no_grad():
        for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxdeadzones).cuda()
            rst_dist = torch.ones(args.maxdeadzones).cuda()
            rst_dist_mse = torch.ones(args.maxdeadzones).cuda()

            end = time.time()

            min_dist = 1e8
            min_mse = 1e8
            pre_mse = 1e8
            pre_dist = 1e8

            min_amount = 0 if not args.change_curve_scale else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())

            # if args.second_order:
            #     hessian = hessian_fischer_approx(grad_list[layerid].clone(), blocksize=2048)
            prev_prune_weights = None

            for d in range(args.maxdeadzones):
                amount = (1. - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[args.maxdeadzones-d-1] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount, mode=args.prune_mode, rank=args.ranking, grad=grad_list[layerid].clone() if args.ranking == "taylor" else None)
                if d > 0:
                    delta_delta_weight = prune_weights - prev_prune_weights
                prev_prune_weights = prune_weights.clone()
                
                delta_weight = prune_weights - layer_weights
                gw = grad_list[layerid].clone()
                cur_mse = (delta_weight ** 2).mean()
                cur_dist = ((delta_weight * gw)**2).mean()
                print("first term", cur_dist)
                if args.second_order:
                    print("calculating taylor 2nd for layer", layerid)
                    if d == 0:
                        prev_second_term = taylor_2nd_order_fischer_approx(delta_weight.clone(), gw, blocksize=2**15) #2**13
                    else:
                        tmp = 0.5 * (delta_delta_weight + 2 * delta_weight).view(-1) @ hessian_deltaw(delta_delta_weight, gw, blocksize=-1)
                        prev_second_term += tmp
                    cur_dist += prev_second_term
                    print("second term", cur_dist)
                rst_dist[args.maxdeadzones-d-1] = cur_dist
                min_dist = cur_dist
                if (cur_mse < min_mse):
                    rst_dist_mse[args.maxdeadzones-d-1] = cur_mse
                    min_mse = cur_mse

                pre_mse = cur_mse
                pre_dist = cur_dist
                
            if args.smooth_curve:
                rst_dist = algo.smooth(rst_dist, 0.1)
                rst_dist, rst_amount, _ = algo.refine_curve(rst_dist, rst_amount)

            rst_dist, rst_amount = rst_dist[None, ...], rst_amount[None, ...]
            
            save_dict = {'rd_amount': rst_amount.cpu().numpy(), 'rd_dist': rst_dist.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()}
                
            io.savemat(('%s/%s_%03d.mat' % (path_output, args.model, layerid)), save_dict)

            rd_dists.append(rst_dist)
            rd_amounts.append(rst_amount)
            rd_dist_mses.append(rst_dist_mse)
    
    rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
    return rd_dists, rd_amounts, grad_list


def gen_rd_curves_structured(net, loader, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = ('./%s_ndz_%04d_rdcurves_structured_opt_dist' % (args.model, args.maxdeadzones))
    else:
        path_output = ('%s/%s_ndz_%04d_rdcurves_structured_opt_dist/%s/' % (prefix, args.model, args.maxdeadzones, suffix))
    
    layers = common.findconv(net, False)
    hookedlayers = common.hooklayers(layers)
    dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
    for l in hookedlayers:
        l.close()
        
    print('total number of layers: %d' % (len(layers)))
    print(f'saving to {path_output}')
    isExists=os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        return algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
#     Y, labels = common.predict2_withgt(net, loader, calib_size=256)
    try:
        Y, labels = common.predict2_withgt(net, loader, calib_size=args.calib_size)
    except:
        Y, labels = common.predict_dali_withgt(net, loader)
        loader.reset()
        

    top_1, top_5 = common.accuracy(Y, labels, topk=(1,5))
    print('original network %s accuracy: top1 %5.2f top5 %5.2f' % (args.model, top_1, top_5))

    rd_dists = []
    rd_amounts = []
    rd_dist_mses = []
    
    if not hasattr(args, "num_parts"):
        args.num_parts = 1
        args.part_id = 0
    len_part = len(layers) // args.num_parts

    with torch.no_grad():
        for layerid in range(args.part_id * len_part, (args.part_id + 1) * len_part):
            if os.path.exists('%s/%s_%03d.mat' % (path_output, args.model, layerid)):
                # m = io.loadmat('%s/%s_%03d.mat' % (path_output, args.model, layerid))
                # rst_amount = m["rd_amount"]
                # rst_dist = m["rd_dist"]
                # rst_dist_mse = m["rst_dist_mse"]
                # rd_dists.append(rst_dist)
                # rd_amounts.append(rst_amount)
                # rd_dist_mses.append(rst_dist_mse)  
                continue
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxdeadzones).cuda()
            rst_dist = torch.ones(args.maxdeadzones).cuda()
            rst_dist_mse = torch.ones(args.maxdeadzones).cuda()

            end = time.time()

            min_dist = 1e8
            min_mse = 1e8
            pre_mse = 1e8
            pre_dist = 1e8

            min_amount = 0 if not args.change_curve_scale else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())

            for d in range(args.maxdeadzones):
                amount = (1. - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[args.maxdeadzones-d-1] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount, mode="structured")
                
                cur_mse = ((prune_weights - layers[layerid].weight)**2).mean()
                layers[layerid].weight.data = prune_weights
                
#                 Y_hat = common.predict2(net, loader, calib_size=256)
                try:
                    Y_hat = common.predict2(net, loader, calib_size=args.calib_size)
                except:
                    Y_hat = common.predict_dali(net, loader)
                    
                    
                if args.worst_case_curve:
                    cur_dist = ((Y - Y_hat) ** 2).mean(dim=1).max()
                else:
                    cur_dist = ((Y - Y_hat) ** 2).mean()
                
                top_1, _ = common.accuracy(Y_hat, labels, topk=(1, 5))
                print('%s | layer %d: amount %6.6f mse %6.6f distortion %6.6f top1 %.2f | time %f' \
                    % (args.model, layerid, amount, cur_mse, cur_dist, top_1, time.time() - end))
                end = time.time()
                # if (cur_dist < min_dist):
                # if amount > 0.9:
                #     import pdb; pdb.set_trace()
                rst_dist[args.maxdeadzones-d-1] = cur_dist
                min_dist = cur_dist
                if (cur_mse < min_mse):
                    rst_dist_mse[args.maxdeadzones-d-1] = cur_mse
                    min_mse = cur_mse

                pre_mse = cur_mse
                pre_dist = cur_dist
                layers[layerid].weight.data = layer_weights

            if args.smooth_curve:
                # import pdb; pdb.set_trace()
                rst_dist, rst_amount = algo.refine_curve(rst_dist, rst_amount)

            io.savemat(('%s/%s_%03d.mat' % (path_output, args.model, layerid)),
                {'rd_amount': rst_amount.cpu().numpy(), 'rd_dist': rst_dist.cpu().numpy(), 'rst_dist_mse': rst_dist_mse.cpu().numpy()})

            rd_dists.append(rst_dist)
            rd_amounts.append(rst_amount)
            rd_dist_mses.append(rst_dist_mse)  
    
    rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
    return rd_dists, rd_amounts


class RDPruner:
    # def __init__(self, amount, args):
    #     self.config = args
    #     self.iter_cnt = 1
    #     self.amount = amount

    def __call__(self, model, amount, args, val_loader, container, to_prune_layerids=None, epoch_cnt=0):
        if not hasattr(self, "amount"):     # initialize at first iter
            assert amount <= 1
            self.amount = amount
            self.iter_cnt = args.iter_start 
            if hasattr(args, "add_layer_per_iter"):
                self.prunedlayers = defaultdict(set)
            unmaskeds = _count_unmasked_weights(model)
            totals = _count_total_weights(model)
            self.prev_pc = [[float(1. - surv/tot)] for surv, tot in zip(unmaskeds, totals)]
        
        if hasattr(self, "prunedlayers") and len(self.prunedlayers[self.iter_cnt]):
            print(f"pruning {to_prune_layerids}-th layer")
            amounts = self.amounts
        else:
            sd = model.state_dict()
            new = sd.copy()
            for k, v in sd.items():
                if "weight_orig" in k:
                    new[k.replace("weight_orig", "weight")] = v * sd[k.replace("weight_orig", "weight_mask")]

            # new = {k: v for k, v in new.items() if "weight_orig" not in k and "mask" not in k}
            container.load_state_dict(new, strict=False) #
            if not hasattr(self, "layers"):
                self.layers = common.findconv(container, False)
            target_sparsity = 1. - (1. - self.amount) ** self.iter_cnt
            
            if args.singlelayer:
                rd_dist, rd_phi = gen_rd_curves_singlelayer(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/singlelayer/", suffix=f'sp{target_sparsity}')  
            elif args.synth_data:
                rd_dist, rd_phi = gen_rd_curves_synth_data(container, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/", suffix=f'sp{target_sparsity}')    
            elif args.prune_mode == "approx":
                outputs = gen_rd_curves_approx(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}{'/second_order_approx/' if args.second_order else ''}", suffix=f'sp{target_sparsity}')
                rd_dist, rd_phi, grad_list = outputs
            elif args.prune_mode == "structured":
                rd_dist, rd_phi = gen_rd_curves_structured(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/", suffix=f'sp{target_sparsity}')
            else:
                rd_dist, rd_phi = gen_rd_curves(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/", suffix=f'sp{target_sparsity}')
            begin = time.time()
            args.slope = common.find_slope(container, target_sparsity, rd_dist, rd_phi, prune_mode=args.prune_mode, blocksize=args.blocksize, rank=args.ranking, grad=grad_list, flop_budget=args.flop_budget, dataset=args.dataset, min_sp=args.min_sp)
            print(args.slope)
            print("time:", time.time() - begin)    
            pc_phi = algo.pareto_condition(self.layers, rd_dist, rd_phi, 2 ** args.slope, min_sp=args.min_sp)

            amounts = [torch.Tensor([max(0, 1 - (1 - p[0]) / (1 - pp[0]))])[0].cuda() for p, pp in zip(pc_phi, self.prev_pc)]

            self.prev_pc = pc_phi
            print(amounts)

        self.amounts = amounts
        if args.prune_mode == "structured":
            prune_weights_l1structured(model, amounts, to_prune_layerids)
        else:
            prune_weights_l1predefined(model, amounts, to_prune_layerids)
        
        mask_save_path = f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/sp{target_sparsity}_{args.model}_ndz_{args.maxdeadzones:04d}_rdcurves_ranking_{args.ranking}_{'secondorder' if args.second_order else ''}approx_opt_dist_mask.pt"
        to_save = {k: v for k, v in model.state_dict().items() if "weight_mask" in k}
        torch.save(to_save, mask_save_path)
        
        if hasattr(self, "prunedlayers"):
            self.prunedlayers[self.iter_cnt] = self.prunedlayers[self.iter_cnt].union(set(to_prune_layerids))
            if not hasattr(self, "amounts"):
                self.amounts = amounts
        if not hasattr(self, "prunedlayers") or len(self.prunedlayers[self.iter_cnt]) == len(amounts):  
            self.iter_cnt += 1


def _compute_unifplus_amounts(model,amount):
    """
    Compute # of weights to prune in each layer.
    """
    amounts = []
    wlist = get_weights(model)
    unmaskeds = _count_unmasked_weights(model)
    totals = _count_total_weights(model)

    last_layer_minimum = np.round(totals[-1]*0.2) # Minimum number of last-layer weights to keep
    total_to_prune = np.round(unmaskeds.sum()*amount)
    
    if wlist[0].dim() == 4:
        amounts.append(0) # Leave the first layer unpruned.
        frac_to_prune = (total_to_prune*1.0)/(unmaskeds[1:].sum())
        if frac_to_prune > 1.0:
            raise ValueError("Cannot be pruned further by the Unif+ scheme! (first layer exception)")
        last_layer_to_surv_planned = np.round((1.0-frac_to_prune)*unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune-last_layer_to_prune)*1.0)/(unmaskeds[1:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (first+last layer exception)")
            amounts.extend([frac_to_prune_middle]*(unmaskeds.size(0)-2))
            amounts.append((last_layer_to_prune*1.0)/unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune]*(unmaskeds.size(0)-1))
    else:
        frac_to_prune = (total_to_prune*1.0)/(unmaskeds.sum())
        last_layer_to_surv_planned = np.round((1.0-frac_to_prune)*unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune-last_layer_to_prune)*1.0)/(unmaskeds[:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (last layer exception)")
            amounts.extend([frac_to_prune_middle]*(unmaskeds.size(0)-1))
            amounts.append((last_layer_to_prune*1.0)/unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune]*(unmaskeds.size(0)))
    return amounts

def _compute_erk_amounts(model,amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds,erks,amount)

def _amounts_from_eps(unmaskeds,ers,amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0-amount)*unmaskeds.sum() # Total to keep.
    
    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds*(1-layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense*unmaskeds).sum()
        
        ers_of_prunables = ers*(1.0-layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables*ers_of_prunables/ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx]/unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx]/unmaskeds[idx]
        
        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)
    
    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx]/unmaskeds[idx])
    return amounts

def _compute_lamp_amounts(model,amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum()*(1.0-amount)))
    
    flattened_scores = [_normalize_scores(w**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores,dim=0)
    topks,_ = torch.topk(concat_scores,num_surv)
    threshold = topks[-1]
    
    # We don't care much about tiebreakers, for now.
    final_survs = [torch.ge(score,threshold*torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores]
    amounts = []
    for idx,final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv/unmaskeds[idx]))
    
    return amounts

def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx,w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0)+w.size(1)+w.size(2)+w.size(3)
        else:
            erks[idx] = w.size(0)+w.size(1)
    return erks

@torch.no_grad()
def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight.count_nonzero())
    return torch.FloatTensor(unmaskeds)

@torch.no_grad()
def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)

def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
    new_scores[sorted_idx] = sorted_scores
    
    return new_scores.view(scores.shape)
