# Z-Image LoRA 微调训练指南

## 概述

本文档介绍如何基于 `linoyts/3d_icon` 数据集，对 Z-Image (DiT + Flow Matching) 模型进行 LoRA 微调训练，使其生成 3D icon 风格的图片。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA A100-40GB（单卡） |
| Conda 环境 | `qwen_lora` |
| 关键依赖 | PyTorch 2.6+, peft 0.17+, bitsandbytes, accelerate |
| 模型权重 | `ckpts/Z-Image-Turbo/`（~30.5GB） |
| 训练分辨率 | 512×512 |
| 数据集 | linoyts/3d_icon（23 张图片 + prompt） |

---

## 快速开始

```bash
# Step 1: 预计算 VAE latents 和 text embeddings（只需运行一次，约 18 秒）
CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute

# Step 2: 开始训练（默认 100 epochs）
CUDA_VISIBLE_DEVICES=0 python train_lora.py --train

# Step 3: 用 LoRA 权重进行推理
CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference --prompt "a 3dicon, a cute cat on purple background"
```

---

## 训练流程详解

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    预计算阶段（一次性）                         │
├─────────────────────────────────────────────────────────────┤
│  Image (512x512)                                            │
│       │                                                     │
│       ▼ VAE Encoder (frozen, FP32)                          │
│  Latent [16, 64, 64]  ──────────────▶  保存到磁盘            │
│                                                             │
│  Prompt                                                     │
│       │                                                     │
│       ▼ Tokenizer + Qwen3 Text Encoder (frozen, BF16)       │
│  Text Embedding [N, 2560]  ──────────▶  保存到磁盘           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      训练阶段                                 │
├─────────────────────────────────────────────────────────────┤
│  加载预计算的 latent + embedding（不需要 VAE/TextEncoder）     │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────┐               │
│  │ Transformer (DiT) + LoRA Adapters        │               │
│  │  - 6.17B params frozen                   │               │
│  │  - 16.7M LoRA params trainable (0.27%)   │               │
│  │  - target: to_q, to_k, to_v, to_out.0   │               │
│  └──────────────────────────────────────────┘               │
│       │                                                     │
│       ▼                                                     │
│  Flow Matching MSE Loss                                     │
│  target = x_0 - noise（模型预测负速度）                       │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: 预计算

将 VAE 编码和文本编码的结果缓存到磁盘，训练时无需加载这两个大模型，大幅节省显存。

```
output/precomputed/
├── latents.pt       # 23 个 [16, 64, 64] tensor
├── embeddings.pt    # 23 个 [N, 2560] 变长 tensor
└── prompts.json     # 23 条 prompt 文本
```

**VAE 编码逻辑：**

```python
def vae_encode(vae, images, scaling_factor, shift_factor):
    h = vae.encoder(images)               # [B, 32, 64, 64]
    mean, _ = h.chunk(2, dim=1)           # [B, 16, 64, 64]
    latents = (mean - shift_factor) * scaling_factor
    return latents
```

- `scaling_factor = 0.3611`
- `shift_factor = 0.1159`
- 使用均值（确定性编码），不采样

**文本编码逻辑：**

```python
# 1. Chat Template 格式化
formatted = tokenizer.apply_chat_template(messages, enable_thinking=True, ...)

# 2. Tokenize (padding to 512)
text_inputs = tokenizer(formatted, padding="max_length", max_length=512, ...)

# 3. Encode (取倒数第二层)
hidden_states = text_encoder(..., output_hidden_states=True).hidden_states[-2]

# 4. 去掉 padding，只保留有效 token
embedding = hidden_states[attention_mask]
```

### Step 2: 训练

#### LoRA 配置

```python
LoraConfig(
    r=16,                                          # LoRA rank
    lora_alpha=16,                                 # scaling factor (alpha/r = 1.0)
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # 注入到 attention 层
    lora_dropout=0.0,
    bias="none",
)
```

注入位置覆盖模型中所有 attention 层：
- `noise_refiner` (2 层) — 有时间步调制
- `context_refiner` (2 层) — 无调制
- `layers` (30 层) — 主体，有时间步调制

#### Flow Matching 训练目标

```python
# 1. 采样 sigma ∈ [0, 1]（logit-normal 分布）
sigma = torch.sigmoid(torch.randn(B))

# 2. 线性插值构造 noisy latent
x_t = (1 - sigma) * x_0 + sigma * noise

# 3. 模型预测负速度
target = x_0 - noise  # 即 -(noise - x_0)

# 4. MSE Loss
loss = F.mse_loss(model_output, target)
```

#### 默认超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 1e-4 | AdamW 8-bit |
| `epochs` | 100 | 小数据集需要多轮 |
| `batch_size` | 1 | 显存限制 |
| `gradient_accumulation_steps` | 4 | 等效 batch=4 |
| `warmup_steps` | 50 | 线性 warmup |
| `lora_rank` | 16 | LoRA 秩 |
| `cfg_dropout_prob` | 0.1 | 10% 概率 drop text（支持 CFG 推理） |
| `save_every_steps` | 500 | 每 500 步保存 checkpoint |

#### 显存优化策略

| 策略 | 效果 |
|------|------|
| 预计算 latents + embeddings | 训练时不加载 VAE (~160MB) 和 TextEncoder (~7.5GB) |
| Transformer BF16 | 权重显存减半 (~12GB vs ~24GB FP32) |
| LoRA (仅 0.27% 参数可训练) | 优化器状态极小 |
| AdamW 8-bit (bitsandbytes) | 优化器状态再减 50% |
| batch_size=1 | 激活显存最小化 |
| Gradient clipping (max_norm=1.0) | 训练稳定性 |

### Step 3: 推理验证

训练完成后，LoRA 权重保存在 `output/lora_3d_icon/lora_weights/`：

```
output/lora_3d_icon/lora_weights/
├── adapter_config.json        # LoRA 配置
├── adapter_model.safetensors  # 权重（~64MB）
└── README.md
```

推理时自动 merge LoRA 权重到基础模型：

```python
from peft import PeftModel

transformer = PeftModel.from_pretrained(transformer, lora_path)
transformer = transformer.merge_and_unload()  # 合并权重，无额外推理开销
```

---

## 命令行参数

```bash
python train_lora.py [--precompute] [--train] [--inference] [options]

必选（三选一）:
  --precompute         预计算 latents 和 text embeddings
  --train              运行 LoRA 训练
  --inference          用 LoRA 权重生成图片

可选:
  --lora_path PATH     LoRA 权重路径 (默认: output/lora_3d_icon/lora_weights)
  --prompt TEXT        推理时的 prompt
  --epochs N           训练轮数 (默认: 100)
  --lr FLOAT           学习率 (默认: 1e-4)
  --rank N             LoRA rank (默认: 16)
  --batch_size N       批大小 (默认: 1)
  --resolution N       训练分辨率 (默认: 512)
```

---

## 训练结果参考

单 epoch (23 张图片) 测试结果：

```
Step 1 | Loss: 1.823472 | LR: 2.00e-06
Step 2 | Loss: 1.769384 | LR: 4.00e-06
Step 3 | Loss: 0.611877 | LR: 6.00e-06
Step 4 | Loss: 1.769776 | LR: 8.00e-06
Step 5 | Loss: 1.100127 | LR: 1.00e-05
```

- 训练速度: ~1 秒/step
- 100 epochs 预计总时间: ~30 分钟
- Loss 预期从 ~1.8 下降到 ~0.1 以下

---

## 技术细节

### 为什么选择 LoRA？

| 方案 | 可训练参数 | 显存需求 | 训练时间 | 效果 |
|------|-----------|---------|---------|------|
| 全量微调 | 6.17B | >80GB | 小时级 | 最好但易过拟合 |
| **LoRA** | **16.7M (0.27%)** | **~25GB** | **~30 分钟** | **性价比最高** |
| Adapter | ~50M | ~30GB | ~1 小时 | 中等 |

23 张图片的小数据集，LoRA 是最合适的选择——参数少不易过拟合，显存友好单卡可跑。

### Flow Matching vs DDPM 训练对比

| 项目 | Flow Matching (Z-Image) | DDPM (SD 1.x) |
|------|------------------------|---------------|
| 训练目标 | 速度场 v = x_0 - noise | 噪声 ε |
| 噪声调度 | sigma ∈ [0,1] 均匀/logit-normal | t ∈ [0,1000] 均匀 |
| Loss | MSE(pred, x_0 - noise) | MSE(pred, noise) |
| 推理步数 | 8 步 (Turbo) | 20-50 步 |

### 模型架构关键数据

| 组件 | 参数量 | 训练时状态 |
|------|--------|-----------|
| Transformer (DiT) | 6.17B | BF16, LoRA 可训练 |
| Text Encoder (Qwen3) | 3.1B | 预计算后不加载 |
| VAE | 83M | 预计算后不加载 |
| **LoRA 适配器** | **16.7M** | **FP32 训练** |

---

## 目录结构

```
Z-Image/
├── train_lora.py                  # 训练脚本
├── ckpts/Z-Image-Turbo/           # 预训练模型权重
│   ├── transformer/
│   ├── vae/
│   ├── text_encoder/
│   ├── tokenizer/
│   └── scheduler/
├── output/
│   ├── precomputed/               # 预计算缓存
│   │   ├── latents.pt
│   │   ├── embeddings.pt
│   │   └── prompts.json
│   └── lora_3d_icon/              # 训练输出
│       ├── lora_weights/          # 最终 LoRA 权重
│       ├── checkpoint-500/        # 中间 checkpoint
│       └── samples/               # 推理生成的样例图
└── src/zimage/                    # 模型源码
    ├── transformer.py
    ├── autoencoder.py
    ├── pipeline.py
    └── scheduler.py
```
