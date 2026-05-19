#!/bin/bash
# Z-Image LoRA 消融实验
#
# 用法:
#   bash run_ablation.sh              # 运行全部消融实验 (不含 baseline)
#   bash run_ablation.sh A_no_timestep_fix  # 只运行实验 A
#   bash run_ablation.sh E_8steps_no_cfg    # E 不需重训，最快
#
# 实验列表:
#   A_no_timestep_fix  - 不做时间步修复 (直接传 sigma)
#   B_turbo_model      - 使用 Z-Image-Turbo 模型
#   C_attention_only   - LoRA 只作用于 attention 层
#   D_rank16           - LoRA rank=16
#   E_8steps_no_cfg    - 推理 8步+无CFG (复用 baseline 权重)
#
# Baseline 已完成, 结果在: output/lora_3d_icon/

set -e

# 链接 baseline 结果到 ablation 目录 (方便 TensorBoard 一起对比)
mkdir -p output/ablation
if [ ! -e output/ablation/baseline ]; then
    ln -sf /root/zyh/Z-Image/output/lora_3d_icon output/ablation/baseline
    echo "Linked baseline -> output/lora_3d_icon"
fi

if [ $# -eq 0 ]; then
    echo "Running all ablation experiments (A~E)..."
    CUDA_VISIBLE_DEVICES=0 python ablation_study.py --run \
        A_no_timestep_fix B_turbo_model C_attention_only D_rank16 E_8steps_no_cfg
else
    echo "Running selected experiments: $@"
    CUDA_VISIBLE_DEVICES=0 python ablation_study.py --run "$@"
fi

echo ""
echo "=========================================="
echo "  查看结果:"
echo "  tensorboard --logdir output/ablation --port 6006 --bind_all"
echo "=========================================="
