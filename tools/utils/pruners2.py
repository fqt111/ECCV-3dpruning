import torch
from torch.nn.utils import prune
from utils.utils import get_weights,get_weights_2d, get_modules, get_all_modules,get_model_sparsity
import numpy as np
import utils.common as common
import utils.algo as algo
import time
import os
import scipy.io as io
from itertools import product
from functools import partial
import einops
# from tools.torch_pruner import struct_identity
import pickle
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import tqdm
def weight_pruner_loader(pruner_string):
    """
    Gives you the pruning methods: LAMP, Glob, Unif, Unif+, and ERK
    """
    if pruner_string == "lamp":
        return prune_weights_lamp
    elif pruner_string == "glob":
        return prune_weights_global
    elif pruner_string == "unif":
        return prune_weights_uniform
    elif pruner_string == "unifplus":
        return prune_weights_unifplus
    elif pruner_string == "erk":
        return prune_weights_erk
    elif pruner_string == "rd":
        return RDPruner()
    elif pruner_string == "knapsack":
        return KnapsackPruner()
    else:
        raise ValueError("Unknown pruner")


"""
prune_weights_reparam: Allocate identity mask to every weight tensors.
prune_weights_l1predefined: Perform layerwise pruning w.r.t. given amounts.
"""


def prune_weights_soft(model, amounts, alpha):
    module_list = get_modules(model)
    for idx, m in enumerate(module_list):
        mask = algo.get_mask(m.weight if not hasattr(m, "weight_ori") else m.weight_ori, float(amounts[idx])).float()
        if not hasattr(m, "weight_ori"):
            m.weight_ori = m.weight.data.clone()
        soft_mask = mask + alpha * (1 - mask)
        m.weight.data = m.weight_ori * soft_mask


def tune_weights_gradients_pg(model, amounts, beta, gm_dropout=0.0):
    module_list = get_modules(model)
    for idx, m in enumerate(module_list):
        mask = algo.get_mask(m.weight if not hasattr(m, "weight_ori") else m.weight_ori, float(amounts[idx])).float()
        soft_mask = mask + beta * (1 - mask)
        if gm_dropout > 0:
            soft_mask *= torch.randint_like(soft_mask, 0, 2)
        if m.weight.grad is not None:
            m.weight.grad.data *= soft_mask


def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m, name="weight")


# def prune_weights_reparam_struct(model):
#     module_list = get_modules(model)
#     for m in module_list:
#         struct_identity(m,name="weight")


def prune_weights_l1predefined(model, amounts, only_layerids=None):
    mlist = get_modules(model)
    for idx, m in enumerate(mlist):
        if only_layerids is not None and idx not in only_layerids:
            continue
        prune.l1_unstructured(m, name="weight", amount=float(amounts[idx]))


def prune_weights_l1structured(model, amounts, only_layerids=None):
    mlist = get_modules(model)
    for idx, m in enumerate(mlist):
        if only_layerids is not None and idx not in only_layerids:
            continue
        prune.ln_structured(m, name="weight", amount=float(amounts[idx]), n=1, dim=1)


"""
Methods: All weights
"""


def prune_weights_global(model, amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


def prune_weights_lamp(model, amount):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model, amount)
    print(amounts)
    prune_weights_l1predefined(model, amounts)


def prune_weights_uniform(model, amount):
    module_list = get_modules(model)
    assert amount <= 1  # Can be updated later to handle > 1.
    for m in module_list:
        prune.l1_unstructured(m, name="weight", amount=amount)


def prune_weights_unifplus(model, amount):
    assert amount <= 1
    amounts = _compute_unifplus_amounts(model, amount)
    prune_weights_l1predefined(model, amounts)


def prune_weights_erk(model, amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model, amount)
    prune_weights_l1predefined(model, amounts)


def prune_weights_rd(model, amount, *args, **kwargs):
    assert amount <= 1
    amounts = _compute_rd_amounts(model, amount, *args, **kwargs)
    print(amounts)
    prune_weights_l1predefined(model, amounts)


def prune_weights_rd_pg(model, amount, *args, **kwargs):
    assert amount <= 1
    amounts = _compute_rd_amounts(model, amount, *args, **kwargs)
    print(amounts)
    prune_weights_soft(model, amounts)


"""
These are not intended to be exported.
"""

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')
def _extract_weight_tuples(model):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    return tuple([(m, "weight") for m in mlist])


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

<<<<<<< HEAD
import torch.nn as nn
def hooklayers(layers):
    return [Hook(layer) for layer in layers]


class Hook:
    def __init__(self, module, backward=False):
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        if isinstance(module,nn.Conv2d):
            self.input_tensor = input[0]
            self.input = torch.tensor(input[0].shape[1:])
            self.output = torch.tensor(output[0].shape[1:])
            self.input_tensor = input[0]
            self.output_tensor = output[0]
        else:
            self.input_tensor = input[0]
            self.input = torch.tensor(input[0].features)
            self.output = torch.tensor(output.features)
            self.input_tensor = input
            self.output_tensor = output

    def close(self):
        self.hook.remove()
=======
>>>>>>> 2e51e307413772c068a6fdd79796bf06ee4b5552

def eval_one_epoch(cfg, model, dataloader, epoch_id, dist_test=False, save_to_file=False, result_dir=None):
    # result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = os.path.join(result_dir ,'final_result' ,'data')
    # if save_to_file:
    #     final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.train()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    grad_list = []
    mlist = get_modules(model)
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
<<<<<<< HEAD
        # hookedlayers = hooklayers(mlist)
        # with torch.no_grad():
        output = model(batch_dict,pruning=True)
        # output = [hookedlayers[i].output for i in range(0, len(hookedlayers))]
        # final_layer_output=output[-1]
        res = torch.mean((output.features ** 2))
        # res.requires_grad_(True)
        res.backward()
        # for l in hookedlayers:
        #     l.close()

        for idx, m in enumerate(mlist):
            if len(grad_list) < len(mlist):
                grad_list.append(m.weight.grad.data / len(dataloader))
            else:
                grad_list[idx] += m.weight.grad.data / len(dataloader)
        for p in model.parameters():
            if p.grad is not None:
                torch.nn.init.zeros_(p.grad.data)
=======
        # with torch.no_grad():
        loss,pred_dicts, ret_dict = model(batch_dict)

        loss['loss'].backward()
        if method=='grad':
            for idx, m in enumerate(mlist):
                if len(grad_list) < len(mlist):
                    grad_list.append(m.weight.grad.data / len(dataloader))
                else:
                    grad_list[idx] += m.weight.grad.data / len(dataloader)
            for p in model.parameters():
                if p.grad is not None:
                    torch.nn.init.zeros_(p.grad.data)
        elif method =='final_layer':
            for idx, m in enumerate(mlist):
                if len(grad_list) < len(mlist):
                    grad_list.append(m.weight  / len(dataloader))
                else:
                    grad_list[idx] += m.weight.grad.data / len(dataloader)
            for p in model.parameters():
                if p.grad is not None:
                    torch.nn.init.zeros_(p.grad.data)
>>>>>>> 2e51e307413772c068a6fdd79796bf06ee4b5552
    
    #     statistics_info(cfg, ret_dict, metric, disp_dict)
    #     annos = dataset.generate_prediction_dicts(
    #         batch_dict, pred_dicts, class_names,
    #         output_path=final_output_dir if save_to_file else None
    #     )
    #     det_annos += annos
        if cfg.LOCAL_RANK == 0:
            # progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # if dist_test:
    #     rank, world_size = common_utils.get_dist_info()
    #     det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=os.path.join(result_dir ,'tmpdir'))
    #     metric = common_utils.merge_results_dist([metric], world_size, tmpdir=os.path.join(result_dir ,'tmpdir'))

    # sec_per_example = (time.time() - start_time) / len(dataloader.dataset)

    # if cfg.LOCAL_RANK != 0:
    #     return {}

    # ret_dict = {}
    # if dist_test:
    #     for key, val in metric[0].items():
    #         for k in range(1, world_size):
    #             metric[0][key] += metric[k][key]
    #     metric = metric[0]

    # gt_num_cnt = metric['gt_num']
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
    #     ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
    #     ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    # total_pred_objects = 0
    # for anno in det_annos:
    #     total_pred_objects += anno['name'].__len__()

    # with open(os.path.join(result_dir ,'result.pkl') , 'wb') as f:
    #     pickle.dump(det_annos, f)

    # result_str, result_dict = dataset.evaluation(
    #     det_annos, class_names,
    #     eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #     output_path=final_output_dir
    # )

    # ret_dict.update(result_dict)

    return grad_list

def eval_single_ckpt(model, test_loader, args, eval_output_dir, epoch_id, dist_test=False):
    # load checkpoint
    # model.load_params_from_file(filename=args.ckpt, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    grad_list=eval_one_epoch(
        cfg,
        model,
        test_loader,
        epoch_id,
        dist_test=dist_test,
        result_dir=eval_output_dir,
    )
    return grad_list

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
    c = damp_lamda * torch.eye(blocksize, device="cuda")
    ret = torch.zeros([1], device="cuda")
    for i in range(len(flat_delta_w) // blocksize):
        dw_i = flat_delta_w[blocksize * i : blocksize * (i + 1)]
        gw_i = flat_gw[blocksize * i : blocksize * (i + 1)]
        for j in range(len(flat_delta_w) // blocksize):
            if i < j:
                continue
            dw_j = flat_delta_w[blocksize * j : blocksize * (j + 1)]
            gw_j = flat_gw[blocksize * j : blocksize * (j + 1)]
            ret += (
                (0.5 if i == j else 1) * dw_i.view(1, -1) @ ((c if i == j else 0) + torch.outer(gw_i, gw_j)) @ dw_j
            ).squeeze()
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
        path_output = "./%s_ndz_%04d_rdcurves_opt_dist" % (args.model, args.maxdeadzones)
    else:
        path_output = "%s/%s_ndz_%04d_rdcurves_opt_dist/%s/" % (prefix, args.model, args.maxdeadzones, suffix)

    layers = common.findconv(net, False)
    # hookedlayers = common.hooklayers(layers)
    # dummy_input = torch.zeros((1, 3, 224, 224)).cuda()
    # _ = net(dummy_input)
    # fil = [hasattr(h, "output") for h in hookedlayers]
    # if False in fil:
    #     layers = [layers[i] for i in range(len(layers)) if fil[i]]
    # for l in hookedlayers:
    #     l.close()

    

    print("total number of layers: %d" % (len(layers)))
    print(f"saving to {path_output}")
    isExists = os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        rd_dists, rd_amounts = algo.load_rd_curve(args.model, layers, args.maxdeadzones, path_output)
        return rd_dists, rd_amounts

    grad_list = []
    net.train()
 
    grad_list=eval_single_ckpt(net, loader, args, path_output, args.iter_start, dist_test=True)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)
    
    
    # net.eval()
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

            min_amount = (
                0
                if not args.change_curve_scale
                else (1 - layers[layerid].weight.count_nonzero() / layers[layerid].weight.numel())
            )

            # if args.second_order:
            #     hessian = hessian_fischer_approx(grad_list[layerid].clone(), blocksize=2048)
            prev_prune_weights = None

            for d in range(args.maxdeadzones):
                amount = (1.0 - min_amount) * d / args.maxdeadzones + min_amount
                rst_amount[args.maxdeadzones - d - 1] = amount
                prune_weights = algo.pruning(
                    layers[layerid].weight,
                    amount,
                    mode=args.prune_mode,
                    rank=args.ranking,
                    grad=grad_list[layerid].clone() if args.ranking == "taylor" else None,
                )
                if d > 0:
                    delta_delta_weight = prune_weights - prev_prune_weights
                prev_prune_weights = prune_weights.clone()

                delta_weight = prune_weights - layer_weights
                gw = grad_list[layerid].clone()
                cur_mse = (delta_weight**2).mean()
                cur_dist = ((delta_weight * gw) ** 2).mean()
                print("first term", cur_dist)
                if args.second_order:
                    print("calculating taylor 2nd for layer", layerid)
                    if d == 0:
                        prev_second_term = taylor_2nd_order_fischer_approx(
                            delta_weight.clone(), gw, blocksize=2**15
                        )  # 2**13
                    else:
                        tmp = (
                            0.5
                            * (delta_delta_weight + 2 * delta_weight).view(-1)
                            @ hessian_deltaw(delta_delta_weight, gw, blocksize=-1)
                        )
                        prev_second_term += tmp
                    cur_dist += prev_second_term
                    print("second term", cur_dist)
                rst_dist[args.maxdeadzones - d - 1] = cur_dist
                min_dist = cur_dist
                if cur_mse < min_mse:
                    rst_dist_mse[args.maxdeadzones - d - 1] = cur_mse
                    min_mse = cur_mse

                pre_mse = cur_mse
                pre_dist = cur_dist

            if args.smooth_curve:
                rst_dist = algo.smooth(rst_dist, 0.1)
                rst_dist, rst_amount, _ = algo.refine_curve(rst_dist, rst_amount)

            rst_dist, rst_amount = rst_dist[None, ...], rst_amount[None, ...]

            save_dict = {
                "rd_amount": rst_amount.cpu().numpy(),
                "rd_dist": rst_dist.cpu().numpy(),
                "rst_dist_mse": rst_dist_mse.cpu().numpy(),
            }

            io.savemat(("%s/%s_%03d.mat" % (path_output, args.model, layerid)), save_dict)

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

    def __call__(self, model, amount, args, val_loader, container, iter_start, to_prune_layerids=None, epoch_cnt=0):
        if not hasattr(self, "amount"):     # initialize at first iter
            assert amount <= 1
            self.amount = amount
            self.iter_cnt = iter_start
            if hasattr(args, "add_layer_per_iter"):
                self.prunedlayers = defaultdict(set)
            unmaskeds = _count_unmasked_weights(model)
            totals = _count_total_weights(model)
            totals2d,totals3d=_count_total_weights_2d_3d(model)
            print('comparision between 2d and 3d model',sum(totals2d),sum(totals3d))
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
            print(self.iter_cnt)
            
            if args.singlelayer:
                rd_dist, rd_phi = gen_rd_curves_singlelayer(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/singlelayer/", suffix=f'sp{target_sparsity}')  
            elif args.synth_data:
                rd_dist, rd_phi = gen_rd_curves_synth_data(container, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/", suffix=f'sp{target_sparsity}')    
            elif args.prune_mode == "approx":
                outputs = gen_rd_curves_approx(container, val_loader, args, prefix=f"./rd_retrain_approx{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/ranking_{args.ranking}{'/second_order_approx/' if args.second_order else ''}", suffix=f'sp{target_sparsity}')
                rd_dist, rd_phi = outputs
            elif args.prune_mode == "structured":
                rd_dist, rd_phi = gen_rd_curves_structured(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/ranking_{args.ranking}/", suffix=f'sp{target_sparsity}')
            else:
                rd_dist, rd_phi = gen_rd_curves(container, val_loader, args, prefix=f"./rd_retrain{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/ranking_{args.ranking}/", suffix=f'sp{target_sparsity}')
            begin = time.time()
            args.slope = common.find_slope(container, target_sparsity, rd_dist, rd_phi, prune_mode=args.prune_mode, rank=args.ranking, grad=None, flop_budget=args.flop_budget, dataset=args.dataset, min_sp=args.min_sp)
            print(args.slope)
            print("time:", time.time() - begin)    
            pc_phi = algo.pareto_condition(self.layers, rd_dist, rd_phi, 2 ** args.slope, min_sp=args.min_sp)
            print(self.prev_pc,pc_phi)
            amounts = [torch.Tensor([max(0, 1 - (1 - p[0]) / (1 - pp[0]))])[0].cuda() for p, pp in zip(pc_phi, self.prev_pc)]

            self.prev_pc = pc_phi
            print(amounts)

        self.amounts = amounts
        if args.prune_mode == "structured":
            prune_weights_l1structured(model, amounts, to_prune_layerids)
        else:
            prune_weights_l1predefined(model, amounts, to_prune_layerids)
        
        mask_save_path = f"./rd_retrain_approx{'_layerbylayer' if hasattr(args, 'add_layer_per_iter') else ''}/weight_rewind_{args.weight_rewind}/sp{target_sparsity}_{args.model}_ndz_{args.maxdeadzones:04d}_rdcurves_ranking_{args.ranking}_{'secondorder' if args.second_order else ''}approx_opt_dist_mask.pt"
        to_save = {k: v for k, v in model.state_dict().items() if "weight_mask" in k}
        torch.save(to_save, mask_save_path)
        
        if hasattr(self, "prunedlayers"):
            self.prunedlayers[self.iter_cnt] = self.prunedlayers[self.iter_cnt].union(set(to_prune_layerids))
            if not hasattr(self, "amounts"):
                self.amounts = amounts
        if not hasattr(self, "prunedlayers") or len(self.prunedlayers[self.iter_cnt]) == len(amounts):  
            self.iter_cnt += 1


def _compute_unifplus_amounts(model, amount):
    """
    Compute # of weights to prune in each layer.
    """
    amounts = []
    wlist = get_weights(model)
    unmaskeds = _count_unmasked_weights(model)
    totals = _count_total_weights(model)

    last_layer_minimum = np.round(totals[-1] * 0.2)  # Minimum number of last-layer weights to keep
    total_to_prune = np.round(unmaskeds.sum() * amount)

    if wlist[0].dim() == 4:
        amounts.append(0)  # Leave the first layer unpruned.
        frac_to_prune = (total_to_prune * 1.0) / (unmaskeds[1:].sum())
        if frac_to_prune > 1.0:
            raise ValueError("Cannot be pruned further by the Unif+ scheme! (first layer exception)")
        last_layer_to_surv_planned = np.round((1.0 - frac_to_prune) * unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune - last_layer_to_prune) * 1.0) / (unmaskeds[1:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (first+last layer exception)")
            amounts.extend([frac_to_prune_middle] * (unmaskeds.size(0) - 2))
            amounts.append((last_layer_to_prune * 1.0) / unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune] * (unmaskeds.size(0) - 1))
    else:
        frac_to_prune = (total_to_prune * 1.0) / (unmaskeds.sum())
        last_layer_to_surv_planned = np.round((1.0 - frac_to_prune) * unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune - last_layer_to_prune) * 1.0) / (unmaskeds[:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (last layer exception)")
            amounts.extend([frac_to_prune_middle] * (unmaskeds.size(0) - 1))
            amounts.append((last_layer_to_prune * 1.0) / unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune] * (unmaskeds.size(0)))
    return amounts


def _compute_erk_amounts(model, amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds, erks, amount)


def _amounts_from_eps(unmaskeds, ers, amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0 - amount) * unmaskeds.sum()  # Total to keep.

    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds * (1 - layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense * unmaskeds).sum()

        ers_of_prunables = ers * (1.0 - layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables * ers_of_prunables / ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx] / unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx] / unmaskeds[idx]

        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)

    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx] / unmaskeds[idx])
    return amounts


def _compute_lamp_amounts(model, amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum() * (1.0 - amount)))

    flattened_scores = [_normalize_scores(w**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores, dim=0)
    topks, _ = torch.topk(concat_scores, num_surv)
    threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [
        torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores
    ]
    amounts = []
    for idx, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / unmaskeds[idx]))

    return amounts


def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0) + w.size(1) + w.size(2) + w.size(3)
        else:
            erks[idx] = w.size(0) + w.size(1)
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
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[: len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum
    sorted_scores /= scores.sum() - scores_cumsum
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)
