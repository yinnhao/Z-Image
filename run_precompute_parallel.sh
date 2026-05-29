#!/bin/bash
# 8-GPU 并行预计算脚本
# 每张卡处理 1/8 的数据，最后合并

set -e

export https_proxy=http://agent.baidu.com:8891
export http_proxy=http://agent.baidu.com:8891

DATA_DIR="/root/paddlejob/workspace/env/vfs_benchmark_cnn/xuziyuan01/zhushou_image_edit_train"
PROMPT_LEVEL="medium"
NUM_SHARDS=8
LOG_DIR="output/edit_precomputed"

mkdir -p "$LOG_DIR"

echo "=== 8-GPU 并行预计算 ==="
echo "数据: $DATA_DIR"
echo "Prompt 级别: $PROMPT_LEVEL"
echo "分片数: $NUM_SHARDS"
echo ""

# 启动 8 个并行进程
PIDS=()
for i in $(seq 0 $((NUM_SHARDS-1))); do
    echo "[$(date '+%H:%M:%S')] 启动 shard $i/$NUM_SHARDS (GPU $i)"
    CUDA_VISIBLE_DEVICES=$i python train_edit_lora.py \
        --precompute \
        --data_dir "$DATA_DIR" \
        --prompt_level "$PROMPT_LEVEL" \
        --shard "$i/$NUM_SHARDS" \
        2>&1 | tee "${LOG_DIR}/log_shard${i}.txt" | grep -E "(Encoding|Saved|Processing|INFO)" | sed "s/^/[GPU$i] /" &
    PIDS+=($!)
done

echo ""
echo "[$(date '+%H:%M:%S')] 所有 $NUM_SHARDS 个 shard 已启动，等待完成..."
echo "  日志文件: ${LOG_DIR}/log_shard{0..7}.txt"
echo ""

# 等待所有进程完成，并检查退出码
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait ${PIDS[$i]}; then
        echo "[ERROR] Shard $i 失败！查看: ${LOG_DIR}/log_shard${i}.txt"
        FAILED=$((FAILED+1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "[ERROR] $FAILED 个 shard 失败，请检查日志"
    exit 1
fi

echo ""
echo "[$(date '+%H:%M:%S')] === 所有 shard 完成，合并中... ==="

# 合并所有 shard
python train_edit_lora.py --merge_shards $NUM_SHARDS

echo ""
echo "[$(date '+%H:%M:%S')] === 完成！==="
echo "缓存目录: $LOG_DIR/"
ls -lh ${LOG_DIR}/source_latents.pt ${LOG_DIR}/target_latents.pt ${LOG_DIR}/semantic_features.pt ${LOG_DIR}/text_embeddings.pt 2>/dev/null
