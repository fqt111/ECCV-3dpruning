#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python3 -m torch.distributed.launch \
        --nproc_per_node=8 \
        --rdzv_endpoint=localhost:49999 \
        train.py \
        --launcher pytorch \
        --cfg_file /home/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml \
        #--pretrained_model /home/OpenPCDet/output/voxelnext_nuscenes_kernel1.pth \
        #--sparsity 0.5

