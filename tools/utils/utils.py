import torch
import torch.nn as nn
from copy import deepcopy
import spconv.pytorch as spconv
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
# Preliminaries. Not to be exported.

def _is_2d_prunable_module(m):
    # if (isinstance(m,BaseBEVBackbone)):
    return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d)) or isinstance(m,nn.ConvTranspose2d)

def _is_prunable_module(m):
    return  isinstance(m,spconv.SubMConv3d) or isinstance(m,spconv.SparseConv3d)

def _get_sparsity(tsr):
    total = tsr.numel()
    nnz = tsr.nonzero().size(0)
    return nnz/total
    
def _get_nnz(tsr):
    return tsr.nonzero().size(0)

# Modules

def get_weights_2d(model):
    weights = []
    for m in model.modules():
        if _is_2d_prunable_module(m):
            weights.append(m.weight)
    return weights

def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m) or _is_2d_prunable_module(m):
            weights.append(m.weight)
    return weights

def get_convweights(model):
    weights = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            weights.append(m.weight)
    return weights

def get_all_modules(model):
    modules_2d = []
    modules_3d = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules_3d.append(m)
        elif _is_2d_prunable_module(m):
            modules_2d.append(m)
    return [modules_3d,modules_2d]


def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m) or _is_2d_prunable_module(m):
            modules.append(m)
    return modules

def get_convmodules(model):
    modules = []
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            modules.append(m)
    return modules

def get_copied_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(deepcopy(m).cpu())
    return modules

def get_model_sparsity(model):
    prunables = 0
    nnzs = 0
    for m in model.modules():
        if _is_prunable_module(m) or _is_2d_prunable_module(m):
            prunables += m.weight.data.numel()
            nnzs += m.weight.data.nonzero().size(0)
    return nnzs/prunables

def get_sparsities(model):
    return [_get_sparsity(m.weight.data) for m in model.modules() if _is_prunable_module(m)]

def get_nnzs(model):
    return [_get_nnz(m.weight.data) for m in model.modules() if _is_prunable_module(m)]
