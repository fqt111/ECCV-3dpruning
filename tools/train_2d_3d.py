import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt,eval_single_ckpt
import re
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from torch.nn.utils import prune
from utils import *
import utils.common as common
from eval_utils import eval_utils
import copy
def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--cfg_file", type=str, default='cfgs/kitti_models/voxel_rcnn_car_spss_ratio0.5_sprs_ratio0.5.yaml', help="specify the config for training")

    parser.add_argument("--batch_size", type=int, default=64, required=False, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=20
                        , required=False, help="number of epochs to train for")
    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--extra_tag", type=str, default="default", help="extra tag for this experiment")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to start from")
    parser.add_argument("--pretrained_model", type=str, default=None, help="pretrained_model")
    parser.add_argument("--finetune_model", type=str, default=None, help="finetune_model")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18889, help="tcp port for distrbuted training")
    parser.add_argument("--sync_bn", action="store_true", default=False, help="whether to use sync bn")
    parser.add_argument("--fix_random_seed", action="store_true", default=False, help="")
    parser.add_argument("--ckpt_save_interval", type=int, default=1, help="number of training epochs")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--max_ckpt_save_num", type=int, default=20, help="max number of saved checkpoint")
    parser.add_argument("--merge_all_iters_to_one_epoch", action="store_true", default=False, help="")
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="set extra config keys if needed"
    )
    parser.add_argument("--max_waiting_mins", type=int, default=0, help="max waiting minutes")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument("--num_epochs_to_eval", type=int, default=20, help="number of checkpoints to be evaluated")
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')


    parser.add_argument("--sparsity", type=float, default=0, help="target sparsity")
    parser.add_argument("--cuda", type=int, help="cuda number")
    parser.add_argument("--model", default='voxel-rcnn',type=str, help="network")
    parser.add_argument("--pruner", default='rd',type=str, help="pruning method")
    parser.add_argument("--dataset", type=str, choices=["cifar", "imagenet"], default="cifar")
    parser.add_argument("--iter_start", type=int, default=1, help="start iteration for pruning")
    parser.add_argument("--iter_end", type=int, default=20, help="start iteration for pruning")
    parser.add_argument("--maxdeadzones", type=int, default=1000)
    parser.add_argument("--ranking", type=str, default="l1", choices=["l1", "taylor"])
    parser.add_argument(
        "--prune_mode", "-pm", type=str, default="rd_retrain_approx", choices=["unstructured", "structured", "approx","rd_retrain","rd_retrain_approx"]
    )
    parser.add_argument("--calib_size", type=int, default=20)
    parser.add_argument("--weight_rewind", action="store_true")
    parser.add_argument("--smooth_curve", action="store_true")
    parser.add_argument("--change_curve_scale", action="store_true")
    parser.add_argument("--worst_case_curve", "-wcc", action="store_true")
    parser.add_argument("--synth_data", action="store_true")
    parser.add_argument("--singlelayer", action="store_true")
    parser.add_argument("--algorithm", type=str, default="greedy")
    parser.add_argument("--approx", action="store_true")
    parser.add_argument("--second_order", action="store_true")
    parser.add_argument(
        "--flop_budget",
        action="store_true",
        help="use flop as the targeting budget in ternary search instead of sparsity. if true, `amounts` and `target_sparsity` variables in the codes will represent flops instead",
    )
    parser.add_argument("--min_sp", type=float, default=0)


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

   
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if args.launcher == "none":
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, "init_dist_%s" % args.launcher)(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        print('total_gpus',total_gpus)
        assert args.batch_size % total_gpus == 0, "Batch size should match the number of gpus"
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / "results_finetune_head" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag

    # output_dir = os.path.join("output" ,cfg.EXP_GROUP_PATH , cfg.TAG ,args.extra_tag) 
    ckpt_dir = output_dir /args.prune_mode/ "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ("log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info("**********************Start logging**********************")
    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ.keys() else "ALL"
    logger.info("CUDA_VISIBLE_DEVICES=%s" % gpu_list)

    if dist_train:
        logger.info("total_batch_size: %d" % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system("cp %s %s" % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
    )
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    prune_set, prune_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False,
        val='prune'
    )


    """ PRETRAIN (IF NEEDED) """
    # if os.path.exists(args.ckpt):
    #     model.load_params_from_file(filename=args.ckpt)
    #     model.cuda()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if not args.pretrained_model: 
        module_list = get_modules(model)
        for m in module_list:
            prune.identity(m, name="weight")
    # cfg.MODEL.NAME='VoxelRCNN_pruning'
    # model_pruning=build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)

    container=copy.deepcopy(model)    
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    def find_max_numbered_folder(path):
        numbered_folders = [folder for folder in os.listdir(path) if folder.isdigit()]
        numbered_folders.sort(key=lambda x: int(x), reverse=True)
        # since should decide prev_pc to get value of amount, so use -2
        return numbered_folders if numbered_folders else None,int(numbered_folders[0]) if numbered_folders else None
    
    start_epoch = it = 0
    last_epoch = start_epoch+1
    iter_start=args.iter_start
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
    elif args.finetune_model:
        path_components = args.finetune_model.split('/')
        iter_start = int(path_components[-2])
        model.load_params_from_file(filename=args.finetune_model, to_cpu=dist_train, logger=logger)
        match = re.search(r"checkpoint_epoch_(\d+)", args.finetune_model)
        if match:
            start_epoch = int(match.group(1))
            print(f'matching current iteration %s and start_epoch %s',iter_start,start_epoch)
    else:
        numbered_folders,iter = find_max_numbered_folder(ckpt_dir)
        iter_start=iter
        print('iter_start',iter_start)
        if numbered_folders and len(numbered_folders)>1:
            cur_folder=os.path.join(ckpt_dir, numbered_folders[0])
            cur_ckpt_list = glob.glob(str(os.path.join(cur_folder ,"*checkpoint_epoch_*.pth")))
            prev_folder=os.path.join(ckpt_dir, numbered_folders[1])
            prev_ckpt_list = glob.glob(str(os.path.join(prev_folder ,"*checkpoint_epoch_*.pth")))
            if cur_ckpt_list:
                cur_ckpt_list.sort(key=os.path.getmtime)
                pretrained_ckpt=cur_ckpt_list[-1]
                model.load_params_from_file(filename=pretrained_ckpt, to_cpu=dist_train, logger=logger)
                # checkpoint = torch.load(pretrained_ckpt)
                # sd=checkpoint['model_state']
                # new = sd.copy()
                # for k, v in sd.items():
                #     if "weight_orig" in k:
                #         new[k.replace("weight_orig", "weight")] = v * sd[k.replace("weight_orig", "weight_mask")]
                # model.load_state_dict(new, strict=False) 
                print('load the ckpt',pretrained_ckpt)

                match = re.search(r"checkpoint_epoch_(\d+)", pretrained_ckpt)
                if match:
                    start_epoch = int(match.group(1))
                    print(f'matching current iteration %s and start_epoch %s',iter_start,start_epoch)
            elif prev_ckpt_list:
                # epoch_number = re.findall(r'\d+', file_name)
                prev_ckpt_list.sort(key=os.path.getmtime)
                pretrained_ckpt=prev_ckpt_list[-1]
                model.load_params_from_file(filename=pretrained_ckpt, to_cpu=dist_train, logger=logger)
                # checkpoint = torch.load(pretrained_ckpt)
                # sd=checkpoint['model_state']
                # new = sd.copy()
                # for k, v in sd.items():
                #     if "weight_orig" in k:
                #         new[k.replace("weight_orig", "weight")] = v * sd[k.replace("weight_orig", "weight_mask")]
                # model.load_state_dict(new, strict=False) 
                print('load the ckpt',pretrained_ckpt)
    
                match = re.search(r"checkpoint_epoch_(\d+)", pretrained_ckpt)
                if match:
                    epoch = int(match.group(1))
                    assert epoch == args.iter_end, (f"previous folder must already trained %s epoch",args.iter_end)
            
        elif numbered_folders and len(numbered_folders)==1:
            cur_folder=os.path.join(ckpt_dir, numbered_folders[0])
            cur_ckpt_list = glob.glob(str(os.path.join(cur_folder ,"*checkpoint_epoch_*.pth")))            
            if cur_ckpt_list:
                cur_ckpt_list.sort(key=os.path.getmtime)
                pretrained_ckpt=cur_ckpt_list[-1]
                model.load_params_from_file(filename=pretrained_ckpt, to_cpu=dist_train, logger=logger)
                # checkpoint = torch.load(pretrained_ckpt)
                # sd=checkpoint['model_state']
                # new = sd.copy()
                # for k, v in sd.items():
                #     if "weight_orig" in k:
                #         new[k.replace("weight_orig", "weight")] = v * sd[k.replace("weight_orig", "weight_mask")]
                # model.load_state_dict(new, strict=False) 
                print('load the ckpt',pretrained_ckpt)
                match = re.search(r"checkpoint_epoch_(\d+)", pretrained_ckpt)
                if match:
                    start_epoch = int(match.group(1))
                    print(f'matching current iteration %s and start_epoch %s',iter_start,start_epoch)
    if args.pretrained_model:
        module_list = get_modules(model)
        for m in module_list:
            prune.identity(m, name="weight")
    # if args.ckpt is not None:
    #     it, start_epoch = model.load_params_with_optimizer(
    #         args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger
    #     )
    #     last_epoch = start_epoch + 1
    #     print(it, start_epoch)
    # else:
    # ckpt_list = glob.glob(str(ckpt_dir / "*checkpoint_epoch_*.pth"))
    # if len(ckpt_list) > 0:
    #     ckpt_list.sort(key=os.path.getmtime)
    #     it, start_epoch = model.load_params_with_optimizer(
    #         ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
    #     )
    #     last_epoch = start_epoch + 1

    
    # flops = common.get_model_flops(model, args.dataset)
    # print(f"Before prune: FLOPs: {flops}")
    # args.remask_per_iter = opt_post["steps"]  # just for naming
    # print(model.state_dict().keys())
    
    pruner = weight_pruner_loader(args.pruner)
    # print(model.state_dict().keys())
    # # """ SET SAVE PATHS """
    # DICT_PATH = f"./dicts/{cfg.MODEL.NAME}/{cfg.MODEL.BACKBONE_3D.NAME}/"
    # if not os.path.exists(DICT_PATH):
    #     os.makedirs(DICT_PATH)
    # if args.weight_rewind:
    #     DICT_PATH += "/weight_rewind/"
    # # if args.lambda_std > 0:
    # #     args.ranking = args.ranking + f"_std{args.lambda_std}"
    # if args.ranking != "l1":
    #     DICT_PATH += f"/ranking_{args.ranking}/"
    # if args.second_order:
    #     DICT_PATH += f"/second_order_approx/"

    # if not os.path.exists(DICT_PATH):
    #     os.makedirs(DICT_PATH)



    if isinstance(model,nn.parallel.DistributedDataParallel): 
        model=model.module
    if start_epoch==0:
        # flops_ratio,nom_flops_3d,denom_flops_3d,nom_flops_2d,denom_flops_2d = common.get_model_flops(model,prune_loader)
        # logger.info(
        # "**********************before pruning/ flops_ratio:%s/ nom_flops3d:%s denon_flops3d:%s nom_flops2d:%s denon_flops2d:%s*********************"
        # % (flops_ratio, nom_flops_3d, denom_flops_3d,nom_flops_2d,denom_flops_2d)
        # )
        amounts,mask,totals=pruner(model,args, prune_loader, container,it,output_dir,sparsity=args.sparsity)
        flops_ratio,nom_flops_3d,denom_flops_3d,nom_flops_2d,denom_flops_2d = common.get_model_flops(model,prune_loader)
        logger.info(
        "**********************after pruning/ flops_ratio:%s/ nom_flops3d:%s denon_flops3d:%s nom_flops2d:%s denon_flops2d:%s*********************"
        % (flops_ratio, nom_flops_3d, denom_flops_3d,nom_flops_2d,denom_flops_2d)
        )
        sparse = utils.get_model_sparsity(model)
        logger.info(f"sparsity: {sparse}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
        "*********************each layer amount:%s/ whole network parameters:%s total pruning weight parameters:%s mask:%s *********************"
        % (amounts, total_params, totals,mask)
        )

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader),
        total_epochs=args.epochs,
        last_epoch=last_epoch,
        optim_cfg=cfg.OPTIMIZATION,
    )
    
    # -----------------------start training---------------------------
    logger.info(
        "**********************Start training /%s/%s(%s))*********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )
    new_ckpt_dir=output_dir / args.prune_mode/"ckpt"/str(args.sparsity)
    new_ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=new_ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if hasattr(train_set, "use_shared_memory") and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info(
        "**********************End training %s/%s(%s)**********************\n\n\n"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )

    logger.info(
        "**********************Start evaluation / %s/%s(%s)**********************"
        % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )
    
    eval_output_dir = output_dir / "eval" / "eval_with_train"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(
        args.epochs - args.num_epochs_to_eval, 0
    )  # Only evaluate the last args.num_epochs_to_eval epochs

    # repeat_eval_ckpt(
    #     model.module if dist_train else model,
    #     test_loader,
    #     args,
    #     eval_output_dir,
    #     logger,
    #     new_ckpt_dir,
    #     dist_test=dist_train,
    #     eval_pruning=True,
    # )
    # eval_single_ckpt(model.module if dist_train else model, test_loader, args, eval_output_dir, logger, 1,dist_test=dist_train,eval_pruning=True)
    logger.info(
        "**********************End evaluation epoch%s/%s/%s(%s)**********************"
        % (str(it),cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    )
        


if __name__ == "__main__":
    main()
