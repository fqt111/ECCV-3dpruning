import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as tmodels
from functools import partial
from utils.models import *
import utils.models.resnet_cifar as resnet_cifar
# import tools.models.models_mae as mae
import timm
from utils.pruners import prune_weights_reparam

def model_and_opt_loader(model_string,DEVICE, reparam=True, weight_rewind=False):
    if DEVICE == None:
        raise ValueError('No cuda device!')
    if model_string == 'vgg16':
        model = VGG16().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'resnet18':
        model = ResNet18().to(DEVICE)
        amount = 0.2
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string in resnet_cifar.__dict__:
        model = resnet_cifar.__dict__[model_string]().to(DEVICE)
        amount = 0.2 #0.1087
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
    elif model_string == 'densenet':
        model = DenseNet121().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 80000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 60000,
            "scheduler": None
        }
    elif model_string == 'effnet':
        model = EfficientNetB0().to(DEVICE)
        amount = 0.20
        batch_size = 100
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 50000,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": None
        }
    elif model_string == 'vgg16_bn':
        model = vgg16_bn(True).to(DEVICE)
        amount = 0.6
        batch_size = 32
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 0,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD if not weight_rewind else optim.Adam,lr=0.01, weight_decay=0.0001),
            # "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 80000,
            "scheduler": partial(sched.CosineAnnealingLR, T_max=80000)
        }
    elif model_string == 'resnet50':
        model = resnet50(True).to(DEVICE)
        amount = 0.2
        batch_size = 64
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 0,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD if not weight_rewind else optim.Adam,lr=0.01, weight_decay=0.0001),
            # "optimizer": partial(optim.Adam,lr=0.0003),
            "steps": 40000,
            "scheduler": partial(sched.CosineAnnealingLR, T_max=40000)
        }
    elif "vit" in model_string or "deit" in model_string:
        model = timm.create_model(model_string, pretrained=True).to(DEVICE)
        amount = 0.5
        batch_size = 16
        opt_pre = {
            "optimizer": partial(optim.AdamW if not weight_rewind else optim.Adam,lr=0.0003),
            "steps": 0,
            "scheduler": None
        }
        opt_post = {
            "optimizer": partial(optim.SGD,lr=0.01, weight_decay=0.0001),
            # "optimizer": partial(optim.AdamW,lr=0.0003, weight_decay=0.001),
            "steps": 80000,
            "scheduler": partial(sched.CosineAnnealingLR, T_max=80000),
            'lambda_kl': 0.,
        }
    else:
        raise ValueError(f'Unknown model: {model_string}')
    if reparam:
        prune_weights_reparam(model)
    return model,amount,batch_size,opt_pre,opt_post
