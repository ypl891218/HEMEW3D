#!/bin/bash
# run_torchrun.sh

. /scratch/10049/weichusheng/python-envs/test/bin/activate
torchrun \
    --nnodes=$1 \
    --nproc_per_node=$3 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$2:29500 \
    /scratch/10049/weichusheng/scratchBackup/HEMEW3D/models/train_ffno3d_DDP_torchrun_multinode.py