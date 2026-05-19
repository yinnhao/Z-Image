#!/bin/bash
# Z-Image LoRA 消融实验 - 4卡并行执行
#
# GPU 分配:
#   GPU 0: A_no_timestep_fix (不做时间步修复)
#   GPU 1: B_turbo_model     (使用 Turbo 模型)
#   GPU 2: C_attention_only  (仅 attention 层)
#   GPU 3: D_rank16          (rank=16)
#   GPU 0: E_8steps_no_cfg   (推理only, 等 A 跑完后执行, 极快)

set -e

OUTPUT_DIR="output/ablation"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 链接 baseline 结果
mkdir -p "$OUTPUT_DIR"
if [ ! -e "${OUTPUT_DIR}/baseline" ]; then
    ln -sf /root/zyh/Z-Image/output/lora_3d_icon "${OUTPUT_DIR}/baseline"
    echo "[INFO] Linked baseline -> output/lora_3d_icon"
fi

echo "============================================"
echo "  Z-Image LoRA 消融实验 (4-GPU 并行)"
echo "============================================"
echo ""
echo "  GPU 0: A_no_timestep_fix"
echo "  GPU 1: B_turbo_model"
echo "  GPU 2: C_attention_only"
echo "  GPU 3: D_rank16"
echo ""
echo "  日志目录: ${LOG_DIR}/"
echo "============================================"
echo ""

# 启动 4 个训练实验并行
echo "[$(date '+%H:%M:%S')] Starting experiments on 4 GPUs..."

CUDA_VISIBLE_DEVICES=0 python ablation_study.py --run A_no_timestep_fix --output_dir "$OUTPUT_DIR" \
    > "${LOG_DIR}/A_no_timestep_fix.log" 2>&1 &
PID_A=$!
echo "  [GPU 0] A_no_timestep_fix  (PID: $PID_A)"

CUDA_VISIBLE_DEVICES=1 python ablation_study.py --run B_turbo_model --output_dir "$OUTPUT_DIR" \
    > "${LOG_DIR}/B_turbo_model.log" 2>&1 &
PID_B=$!
echo "  [GPU 1] B_turbo_model      (PID: $PID_B)"

CUDA_VISIBLE_DEVICES=2 python ablation_study.py --run C_attention_only --output_dir "$OUTPUT_DIR" \
    > "${LOG_DIR}/C_attention_only.log" 2>&1 &
PID_C=$!
echo "  [GPU 2] C_attention_only   (PID: $PID_C)"

CUDA_VISIBLE_DEVICES=3 python ablation_study.py --run D_rank16 --output_dir "$OUTPUT_DIR" \
    > "${LOG_DIR}/D_rank16.log" 2>&1 &
PID_D=$!
echo "  [GPU 3] D_rank16           (PID: $PID_D)"

echo ""
echo "[$(date '+%H:%M:%S')] All training experiments launched. Waiting..."
echo ""

# 实时显示各实验状态
monitor_progress() {
    while kill -0 $PID_A 2>/dev/null || kill -0 $PID_B 2>/dev/null || \
          kill -0 $PID_C 2>/dev/null || kill -0 $PID_D 2>/dev/null; do
        echo -n "  [$(date '+%H:%M:%S')] Running: "
        kill -0 $PID_A 2>/dev/null && echo -n "A " || echo -n "A✓ "
        kill -0 $PID_B 2>/dev/null && echo -n "B " || echo -n "B✓ "
        kill -0 $PID_C 2>/dev/null && echo -n "C " || echo -n "C✓ "
        kill -0 $PID_D 2>/dev/null && echo -n "D " || echo -n "D✓ "
        echo ""
        sleep 60
    done
}
monitor_progress &
MONITOR_PID=$!

# 等待所有训练完成
wait $PID_A
EXIT_A=$?
echo "[$(date '+%H:%M:%S')] A_no_timestep_fix finished (exit: $EXIT_A)"

wait $PID_B
EXIT_B=$?
echo "[$(date '+%H:%M:%S')] B_turbo_model finished (exit: $EXIT_B)"

wait $PID_C
EXIT_C=$?
echo "[$(date '+%H:%M:%S')] C_attention_only finished (exit: $EXIT_C)"

wait $PID_D
EXIT_D=$?
echo "[$(date '+%H:%M:%S')] D_rank16 finished (exit: $EXIT_D)"

# 停止监控
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "[$(date '+%H:%M:%S')] All training done. Running inference-only experiment E..."

# 实验 E: 只需推理, 使用 baseline 权重
CUDA_VISIBLE_DEVICES=0 python ablation_study.py --run E_8steps_no_cfg --output_dir "$OUTPUT_DIR" \
    > "${LOG_DIR}/E_8steps_no_cfg.log" 2>&1
EXIT_E=$?
echo "[$(date '+%H:%M:%S')] E_8steps_no_cfg finished (exit: $EXIT_E)"

# 汇总结果
echo ""
echo "============================================"
echo "  消融实验完成"
echo "============================================"
echo ""
echo "  实验结果:"
printf "    A_no_timestep_fix:  %s\n" "$([ $EXIT_A -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
printf "    B_turbo_model:      %s\n" "$([ $EXIT_B -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
printf "    C_attention_only:   %s\n" "$([ $EXIT_C -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
printf "    D_rank16:           %s\n" "$([ $EXIT_D -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
printf "    E_8steps_no_cfg:    %s\n" "$([ $EXIT_E -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""
echo "  查看 loss 曲线 + 生成图片:"
echo "    tensorboard --logdir ${OUTPUT_DIR} --port 6006 --bind_all"
echo ""
echo "  查看日志:"
echo "    tail -f ${LOG_DIR}/<experiment>.log"
echo ""
echo "  对比生成图片:"
echo "    ls ${OUTPUT_DIR}/*/samples/"
echo "============================================"
