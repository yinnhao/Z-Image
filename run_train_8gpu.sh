#!/bin/bash
# 8-GPU DDP 训练启动脚本
# 使用 torchrun 启动分布式训练

set -e

export https_proxy=http://agent.baidu.com:8891
export http_proxy=http://agent.baidu.com:8891

# 训练参数（可按需修改）
EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-1}
LR=${LR:-1e-4}
RANK=${RANK:-64}
GRAD_ACCUM=${GRAD_ACCUM:-2}
VAL_EVERY=${VAL_EVERY:-200}

echo "=== 8-GPU DDP Edit LoRA Training ==="
echo "Epochs: $EPOCHS"
echo "Per-GPU batch size: $BATCH_SIZE"
echo "Effective batch size: $((BATCH_SIZE * 8 * GRAD_ACCUM))"
echo "Learning rate: $LR"
echo "LoRA rank: $RANK"
echo "Validate every: $VAL_EVERY steps"
echo ""

torchrun --nproc_per_node=8 --master_port=29500 \
    train_edit_lora.py \
    --train \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --rank $RANK \
    --grad_accum $GRAD_ACCUM \
    --val_every $VAL_EVERY
