# Z-Image-Edit LoRA 训练文档

## 概述

Z-Image-Edit 通过 LoRA 微调使 Z-Image 基础模型获得 **"源图 + 编辑指令 → 编辑后图片"** 的能力。

核心思路：将源图信息通过两条路径注入 transformer 的 Single-Stream Attention：
1. **语义路径**：源图 → SigLip-2（frozen）→ Semantic Processor → 语义 tokens（与 text tokens 拼接）
2. **像素路径**：源图 → VAE encode → source latent（与 noised target 在 T 维拼接）

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `train_edit_lora.py` | 完整训练脚本（预计算 + 训练 + 推理） |

---

## 架构设计

### Transformer 统一序列

```
统一序列 = [noised_target_tokens, source_vae_tokens, text_tokens, semantic_tokens]
             ─────────── image 部分 ──────────    ────── context 部分 ──────
             经过 noise_refiner (有 timestep 调制)   经过 context_refiner (无 timestep)
```

位置编码（3D RoPE）：
- **text tokens**: axis0 = 1..N_text, axis1=0, axis2=0
- **semantic tokens**: axis0 = N_text+1..N_text+N_sem, axis1=0, axis2=0
- **image tokens**: frame 0 = noised target, frame 1 = source（clean）

### Semantic Processor

```python
class SemanticProcessor(nn.Module):
    """SigLip-2 特征 [B, 576, 1024] → [B, 576, 2560]"""
    def __init__(self, siglip_dim=1024, output_dim=2560):
        self.norm = nn.LayerNorm(siglip_dim)
        self.proj = nn.Linear(siglip_dim, output_dim)
```

输出 2560 维（与 text embedding 相同），拼接后一起经过 transformer 内部的 `cap_embedder`（RMSNorm + Linear 2560→3840）和 `context_refiner`，**无需修改 transformer 源码**。

### 训练信息流

```
源图 ──→ SigLip-2 (frozen) ──→ SemanticProcessor (trained) ──→ semantic [576, 2560] ─┐
                                                                                       ├─→ cap_feats [seq+576, 2560]
编辑指令 ──→ Qwen3 (frozen) ──→ text_embedding [seq, 2560] ───────────────────────────┘

源图 ──→ VAE (frozen) ──→ source_latent [16, 1, H, W] ──→ ┐
                                                            ├─→ x_combined [16, 2, H, W]
目标图 ──→ VAE (frozen) ──→ target_latent + noise ─────────┘

transformer(x_combined, timestep, cap_feats) → pred [16, 2, H, W]
                                                      取 frame 0 ↓
                                               loss = MSE(pred_frame0, target - noise)
```

---

## 数据集格式

```
edit_data/
├── metadata.jsonl       # 每行一个 JSON
├── source/              # 源图（编辑前）
│   ├── src_001.png
│   └── ...
└── target/              # 目标图（编辑后）
    ├── tgt_001.png
    └── ...
```

`metadata.jsonl` 格式：
```json
{"source": "src_001.png", "target": "tgt_001.png", "prompt": "turn the logo green"}
{"source": "src_002.png", "target": "tgt_002.png", "prompt": "add a hat to the cat"}
```

---

## 使用方法

### 1. 预计算

```bash
CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --precompute --data_dir edit_data/
```

输出到 `output/edit_precomputed/`：
- `source_latents.pt` — 源图 VAE latents `[N, 16, H/8, W/8]`
- `target_latents.pt` — 目标图 VAE latents `[N, 16, H/8, W/8]`
- `semantic_features.pt` — SigLip-2 特征 `[N, 576, 1024]`
- `text_embeddings.pt` — Qwen3 embeddings（变长 list）
- `prompts.json` — 原始 prompt 文本

### 2. 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --train
```

输出到 `output/edit_lora/`：
- `tensorboard/` — TensorBoard 日志
- `checkpoint-{step}/` — 中间 checkpoint
- `final/lora/` — 最终 LoRA 权重
- `final/semantic_processor.pt` — 最终 Semantic Processor 权重

监控训练：
```bash
tensorboard --logdir output/edit_lora/tensorboard
```

### 3. 推理

```bash
python train_edit_lora.py --inference --source input.png --prompt "turn the logo green"
```

可选参数：
```bash
python train_edit_lora.py --inference \
    --source input.png \
    --prompt "make it look like a watercolor painting" \
    --weights_path output/edit_lora/final \
    --steps 50 \
    --cfg 5.0 \
    --resolution 512
```

---

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SigLip-2 | frozen | `google/siglip2-large-patch16-384`（1024 dim, 576 tokens） |
| Semantic Processor | 全量训练 | LayerNorm + Linear（1024→2560），~2.6M 参数 |
| Transformer LoRA | rank=64 | 覆盖 attention（to_q/k/v/out）+ FFN（w1/w2/w3） |
| Learning rate | 1e-4 | AdamW8bit |
| Resolution | 512 | 源图和目标图统一 resize |
| Batch size | 1 × 4 accumulation | 有效 batch=4 |
| Epochs | 500 | 可通过 `--epochs` 覆盖 |
| CFG dropout | 0.1 | 训练时随机丢弃条件（支持推理时 CFG） |
| 推理步数 | 50 | 基础模型（非 Turbo） |
| Guidance scale | 5.0 | classifier-free guidance 强度 |

---

## 关键设计决策

### 1. 为什么 Semantic Processor 输出 2560 而不是 3840？

Transformer 内部有 `cap_embedder`（RMSNorm + Linear 2560→3840），所有 cap_feats 都会经过这一层。如果直接输出 3840，就需要修改 transformer 代码跳过 cap_embedder。输出 2560 可以和 text embedding 走完全相同的处理路径，**零侵入**。

### 2. 为什么用 2-frame 拼接而不是 channel 拼接？

Z-Image transformer 的 `patchify_and_embed` 天然支持多帧输入（视频设计），通过 T 维拼接：
- Frame 0：noised target（参与去噪）
- Frame 1：source（clean，时间步调制为 t=1）

Transformer 内部的 noise_refiner 对两帧使用相同的 timestep 调制，但 source 帧实际上是 clean 的，这允许模型学习在去噪过程中参考 source 信息。

### 3. 推理时为什么取反模型输出？

Z-Image 使用非标准的 sign convention：模型预测 `x_0 - ε`（去噪方向），而 Euler scheduler 期望标准 velocity。取反后变为 `ε - x_0`（噪声方向），与 scheduler 约定一致。

### 4. CFG Dropout 策略

训练时以 10% 概率将 cap_feats 替换为零向量，使模型学习无条件生成。推理时通过对比有条件/无条件输出，用 guidance_scale 放大编辑效果。

---

## 命令行参数总览

```
python train_edit_lora.py [MODE] [OPTIONS]

MODE（三选一）:
  --precompute          预计算所有编码
  --train               训练 LoRA + Semantic Processor
  --inference           编辑推理

OPTIONS:
  --data_dir DIR        编辑数据集目录（默认: edit_data）
  --output_dir DIR      输出目录（默认: output/edit_lora）
  --weights_path PATH   推理时权重路径（默认: output/edit_lora/final）
  --source PATH         推理时源图路径
  --prompt TEXT         推理时编辑指令
  --epochs N            覆盖训练轮数
  --lr FLOAT            覆盖学习率
  --rank N              覆盖 LoRA rank（alpha 同步设置）
  --batch_size N        覆盖 batch size
  --resolution N        覆盖分辨率
  --steps N             覆盖推理步数
  --cfg FLOAT           覆盖 guidance scale
```

---

## 依赖

在现有 Z-Image 依赖基础上，额外需要：
```bash
pip install transformers  # SigLip-2（已有 Qwen3 依赖）
pip install peft          # LoRA（已有）
pip install bitsandbytes  # 8-bit AdamW（已有）
```

SigLip-2 模型首次使用时会自动从 HuggingFace 下载。如需代理：
```bash
export https_proxy=http://agent.baidu.com:8891
```
