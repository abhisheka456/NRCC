#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='cifar10'
ROOT=../../..

mkdir -p log

export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# use torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$2 \
    $ROOT/main.py cifar10_r18_byol_s.yml 2>&1 | tee log/seg_$now.txt


# python -m torch.distributed.launch --nproc_per_node=4 --master_port=17672 main.py config/cifar10_r18_psclr2.yml