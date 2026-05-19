#!/bin/bash
# Z-Image LoRA Training Pipeline (基于 Z-Image 基础模型)

# 清理旧的训练输出和缓存（换模型后需重新预计算）
rm -rf output/lora_3d_icon/
rm -rf output/precomputed/

# Step 1: 预计算 latents 和 text embeddings
CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute

# Step 2: 训练（500 epochs，每 100 步生成验证图片）
CUDA_VISIBLE_DEVICES=0 python train_lora.py --train

# Step 3: 用 LoRA 权重对全部训练集 prompt 生成图片 (不传 --prompt 则自动用全部)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference
