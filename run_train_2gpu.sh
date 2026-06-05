#!/bin/bash
set -e

export https_proxy=http://agent.baidu.com:8891
export http_proxy=http://agent.baidu.com:8891

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 \
    train_edit_lora.py \
    --train \
    --batch_size 1 \
    --grad_accum 4 \
    --val_every 200
