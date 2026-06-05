#!/bin/bash
# Z-Image-Edit 全参数微调 8卡启动脚本
# 使用 FSDP 在 8×A100-40GB 上训练

set -e

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=1800

torchrun --nproc_per_node=8 --master_port=29600 \
    train_edit_full.py --train \
    --epochs 30 \
    --batch_size 1 \
    --lr 5e-6 \
    --grad_accum 4 \
    --val_every 100 \
    --save_every 200
