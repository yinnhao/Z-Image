# Z-Image 全参数微调学习率问题分析

本文档分析 Z-Image 全参数微调失败的原因，以及学习率调度的正确配置。

---

## 问题现象

在使用 `run_train_full_7gpu.sh` 训练 29000 步后：
- 输出图像严重失真
- Loss 先下降后上升
- 相同任务使用 LoRA 训练则正常

run_train_full_7gpu.sh的内容：
```shell
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
    --epochs 30000 \
    --batch_size 1 \
    --lr 5e-6 \
    --grad_accum 4 \
    --val_every 1000 \
    --save_every 5000

```

---

## 根本原因：`--epochs` 参数误解

### 1. 代码中的 `effective_steps` 计算

`train_edit_full.py` 第 499-508 行：

```python
steps_per_epoch = math.ceil(len(train_dataset) / (config.batch_size * world_size))
total_steps = config.epochs * steps_per_epoch
effective_steps = total_steps // config.gradient_accumulation_steps

def lr_lambda(step):
    if step < config.warmup_steps:
        return step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, effective_steps - config.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
```

### 2. 实际数值计算

当前配置：
- `--epochs 30000`（被解读为 **30,000 个 epoch**，而非 30,000 步）
- 数据集大小：6564 样本
- 7 卡并行 (GPU 1-7)
- `batch_size = 1`
- `gradient_accumulation_steps = 4`
- `warmup_steps = 500`

计算过程：
```
steps_per_epoch = ceil(6564 / (1 × 7)) = 938   # forward iterations per epoch
optimizer_steps_per_epoch = 938 / 4 = 235      # optimizer steps per epoch
total_steps = 30000 × 938 = 28,140,000         # forward iterations
effective_steps = 28140000 / 4 = 7,035,000     # optimizer steps
```

### 3. Step 29000 时的学习率

```python
progress = (29000 - 500) / (7035000 - 500) ≈ 0.0041
lr = 5e-6 × 0.5 × (1 + cos(π × 0.0041)) ≈ 5e-6
```

**结论**：在训练 29000 步时，余弦衰减几乎还没开始（进度仅 0.41%），学习率仍保持初始值 5e-6。

---

## 为什么学习率应该衰减？

### 全参数微调 vs LoRA 的对比

| | LoRA | 全参数微调 |
|---|---|---|
| 可训练参数量 | ~160M (2.6%) | ~6B (100%) |
| 典型学习率 | 1e-4 ~ 5e-4 | 1e-6 ~ 5e-6 |
| 对基座模型影响 | 小（只修改 adapter） | 大（直接修改所有权重） |
| 风险 | 低（不易遗忘预训练知识） | 高（易破坏预训练能力） |

### 正确的学习率调度

一个标准的余弦衰减学习率调度：

```
          Warmup (500 steps)             Cosine Decay
        ───────────────────────────────────────────────────────→ Step
      ╱                                                            ╲
     │                                                              │
    │                                                                │
   │                                                                  │
  │                                                                    │
 │                                                                      │
0│                                                                       │1 (relative scale)
│                                                                         │
│                                                                          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────
    0               500                29,000           6,427,500

由于 effective_steps 被错误地设置为 6.4M，step 29000 处进度仅 0.45%，
学习率几乎完全没有衰减，一直在 5e-6 的高位运行。

结果：模型一直在用较大的步长修改所有 6B 参数，快速破坏预训练知识 → 输出失真。
```

---

## 解决方案

### 方案 1：修正 `--epochs` 参数

如果要训练 30,000 个 optimizer steps（因为 `global_step` 在 `optimizer.step()` 后递增），应该计算正确的 epoch 数：

```bash
steps_per_epoch = ceil(6564 / (1 × 7)) = 938  # forward iterations per epoch
optimizer_steps_per_epoch = 938 / 4 = 235     # 由于 gradient_accumulation_steps=4
epochs = ceil(30000 / 235) = 128
```

修改 `run_train_full_7gpu.sh`：

```bash
torchrun --nproc_per_node=7 --master_port=29600 \
    train_edit_full.py --train \
    --epochs 128 \          # 修改这里：30000 → 128
    --batch_size 1 \
    --lr 5e-6 \
    --grad_accum 4 \
    --val_every 1000 \
    --save_every 5000
```

### 方案 2：添加 `--total_steps` 参数（推荐）

更清晰的 API：直接指定总步数而非 epoch 数。

修改 `train_edit_full.py` 添加参数：

```python
parser.add_argument("--total_steps", type=int, default=None,
                    help="Total training steps (overrides epochs calculation)")
```

修改 `--epochs 30000` 为 `--total_steps 30000`。

---

## 为什么学习率设置为 5e-6？

全参数微调使用较小的学习率是标准实践：

1. **参数量大**：6B 参数全部可训练，大学习率会导致剧烈变化
2. **保留预训练能力**：小学习率 + 衰减确保微调基于预训练知识而非破坏它
3. **与 LoRA 对比**：
   - LoRA 只训练 2.6% 参数，用大学习率（1e-4~5e-4）快速适应新任务
   - 全参数需要小学习率（1e-6~5e-6），通过衰减逐步调整

---

## 验证修复

修复后（`--epochs 128`），step 29000 的学习率应该是：

```python
steps_per_epoch = 938
optimizer_steps_per_epoch = 938 / 4 = 235
effective_steps = 128 × 235 = 30,080
progress = (29000 - 500) / (30080 - 500) ≈ 0.986
lr = 5e-6 × 0.5 × (1 + cos(π × 0.986)) ≈ 5e-6 × 0.5 × 0.008 ≈ 2e-8
```

这才符合预期：训练接近尾声，学习率已衰减到接近 0。

---

## 总结

| | 错误配置 | 正确配置 |
|---|---|---|
| `--epochs` | 30000 | 128 |
| `effective_steps` | 7,035,000 | 30,080 |
| Step 29000 时的进度 | 0.41% | 98.6% |
| Step 29000 时的 LR | 5e-6 (几乎不变) | ~2e-8 (已衰减完) |
| 结果 | 模型破坏，输出失真 | 收敛，正常编辑 |

**核心问题**：`--epochs 30000` 被误解为 30,000 个 epoch，导致余弦衰减从未真正触发，学习率一直保持高位，破坏了预训练模型的能力。

**修复**：将 `--epochs` 改为 **128**（计算依据：30000 optimizer steps / 235 optimizer steps per epoch）。