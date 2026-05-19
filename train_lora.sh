#!/bin/bash
# Z-Image LoRA Training Pipeline

# 清理旧的训练输出
rm -rf output/lora_3d_icon/

# Step 1: 预计算 latents 和 text embeddings（只需运行一次，已有缓存可跳过）
# CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute

# Step 2: 训练（默认 100 epochs，每 100 步生成验证图片 + loss 曲线）
CUDA_VISIBLE_DEVICES=0 python train_lora.py --train

# Step 3: 用 LoRA 权重生成图片
# CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference --prompt "a 3dicon, a cute cat on purple background"
CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference --prompt "a 3dicon, netflix logo with popcorn and a cup"
