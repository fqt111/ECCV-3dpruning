python3 -m torch.distributed.launch --nproc_per_node=8 --rdzv_endpoint=localhost:49999 sparsity_flops.py --launcher pytorch --cfg_file /home/OpenPCDet/tools/cfgs/kitti_models/second.yaml --pretrained_model /home/OpenPCDet/output/second_7862.pth --sparsity 0.5
python3 -m torch.distributed.launch --nproc_per_node=8 --rdzv_endpoint=localhost:49999 sparsity_flops.py --launcher pytorch --cfg_file /home/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml --pretrained_model /home/OpenPCDet/output/pv_rcnn_8369.pth --sparsity 0.6
python3 -m torch.distributed.launch --nproc_per_node=8 --rdzv_endpoint=localhost:49999 sparsity_flops.py --launcher pytorch --cfg_file /home/OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_car.yaml --pretrained_model /home/OpenPCDet/output/voxel_rcnn_car_84.54.pth --sparsity 0.7

