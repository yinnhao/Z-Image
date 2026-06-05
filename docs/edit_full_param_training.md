# Z-Image-Edit 全参数微调方案

## 背景

`train_edit_lora.py` 只训练 LoRA (~103M) + SemanticProcessor (~2.6M)，冻结 6B transformer。由于 SemanticProcessor 是全新初始化的模块，LoRA 容量不足以让 transformer 充分学习新的语义输入路径，导致生成图像畸形。需要全参数微调来解决这个问题。

8×A100-40GB 用 FSDP 可以轻松容纳 6B 模型全参数训练（估算 ~25-30GB/卡）。

## 文件结构

| 文件 | 用途 |
|------|------|
| `train_edit_full.py` | FSDP 全参数微调训练脚本 |
| `run_train_full_8gpu.sh` | 8 卡 torchrun 启动脚本 |

## LoRA 版 vs 全参数版对比

| 方面 | LoRA 版 | 全参数版 |
|------|---------|----------|
| 并行策略 | 手动 all-reduce LoRA 梯度 | FSDP (ZeRO-3) 自动分片 |
| 可训练参数 | ~105M | ~6B + 2.6M |
| 优化器 | AdamW8bit | AdamW (fp32 states, FSDP 分片) |
| 学习率 | 1e-4 | 5e-6 |
| Epochs | 500 | 30 |
| 梯度同步 | 手动 flatten→all_reduce | FSDP 内置 + `no_sync()` |

## FSDP 配置

- **分片粒度**: 按 `ZImageTransformerBlock` 自动 wrap
- **Mixed Precision**: param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=bf16
- **Sharding Strategy**: FULL_SHARD (ZeRO-3)
- **Activation Checkpointing**: 每个 transformer block 使用 NO_REENTRANT checkpoint

## 模型加载策略

- Rank 0: meta device → `load_sharded_safetensors` 到 CPU → assign
- 其他 rank: meta device → `to_empty(device="cpu")`
- FSDP `sync_module_states=True` 自动从 rank 0 广播

## SemanticProcessor

仅 2.6M 参数，不做 FSDP 分片，直接 DDP 包裹。

## Gradient Accumulation

使用 FSDP 的 `no_sync()` 避免非最后一个 micro-step 的冗余 all-reduce：

```python
if is_accumulating:
    with transformer.no_sync():
        loss.backward()
else:
    loss.backward()  # 此处触发 all-reduce
```

## Checkpoint 保存

使用 `FULL_STATE_DICT` + `offload_to_cpu` + `rank0_only`，保存完整 transformer state_dict 到单文件。支持通过 `latest_checkpoint` 符号链接恢复训练。

## 默认超参数

```python
learning_rate = 5e-6
weight_decay = 0.01
epochs = 30
warmup_steps = 500
batch_size = 1          # per GPU
gradient_accumulation_steps = 4  # effective batch = 32
max_grad_norm = 1.0
validate_every_steps = 100
save_every_steps = 200
cfg_dropout_prob = 0.1
```

## 使用方法

```bash
# 预计算（复用 train_edit_lora.py 的预计算结果，共享 output/edit_precomputed/）

# 训练
bash run_train_full_8gpu.sh

# 推理
python train_edit_full.py --inference --source input.png --prompt "把 logo 改成绿色"
```

## 验证方法

1. 启动训练：`bash run_train_full_8gpu.sh`
2. 确认 8 卡显存 < 35GB/卡（`nvidia-smi`）
3. 验证 loss 下降趋势（TensorBoard: `output/edit_full_param/runs/`）
4. 检查验证图片质量（`output/edit_full_param/val_samples/`）
5. 对比 LoRA 版本的验证图确认质量提升
