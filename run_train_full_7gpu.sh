#!/bin/bash
# Z-Image-Edit 全参数微调 7卡启动脚本
# 使用 FSDP 在 7×A100-40GB 上训练 (GPU 1-7)

set -e

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qwen_lora

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=1800

torchrun --nproc_per_node=7 --master_port=29600 \
    train_edit_full.py --train \
    --epochs 128 \
    --batch_size 1 \
    --lr 5e-6 \
    --semantic_lr 1e-4 \
    --grad_accum 4 \
    --val_every 1000 \
    --save_every 5000
