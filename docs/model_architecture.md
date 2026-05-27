# 现代图像生成模型原理：以 Z-Image 为例

本文以 Z-Image 代码库为例，详细讲解基于 **DiT（Diffusion Transformer）+ Flow Matching** 的现代文生图模型的工作原理。每个模块按照 **直觉 → 数学 → 工程 Trick → 代码** 的范式展开。

---

## 0. 整体架构概览

### 推理数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Z-Image Generation Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   "一只猫在月球上"                                                            │
│         │                                                                    │
│         ▼                                                                    │
│   ┌───────────┐     ┌──────────────┐                                        │
│   │ Tokenizer │────▶│ Text Encoder │──── text_embeds [N, 2560]              │
│   └───────────┘     └──────────────┘           │                            │
│                                                 ▼                            │
│   Random Noise ──▶ ┌─────────────────────────────────────┐                  │
│   [1,16,128,128]   │         Transformer (DiT)           │                  │
│                    │  noise_refiner → context_refiner     │                  │
│   Timestep t ───▶  │  → 30 layers joint attention        │                  │
│                    └──────────────────┬──────────────────┘                  │
│                                       │ (×8 steps)                          │
│                                       ▼                                      │
│                              Denoised Latent                                 │
│                              [1,16,128,128]                                  │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────┐                                 │
│                              │ VAE Decoder │                                 │
│                              └──────┬──────┘                                 │
│                                     ▼                                        │
│                              Output Image                                    │
│                              [1024 × 1024]                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 训练数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Z-Image Training Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─── 离线预计算（只做一次）───────────────────────────────────────────────────┐  │
│  │                                                                             │  │
│  │  训练图片            "a 3d icon of ..."                                     │  │
│  │  [1024×1024]              │                                                 │  │
│  │       │                   ▼                                                 │  │
│  │       ▼            ┌───────────┐    ┌──────────────┐                        │  │
│  │  ┌─────────┐      │ Tokenizer │───▶│ Text Encoder │──▶ text_embeds [N,2560]│  │
│  │  │   VAE   │      └───────────┘    └──────────────┘        │               │  │
│  │  │ Encoder │                                                │               │  │
│  │  └────┬────┘                                                │               │  │
│  │       ▼                                                     ▼               │  │
│  │  x_0 (latent)                                         保存到磁盘             │  │
│  │  [B, 16, 64, 64]                                     (cache_dir/)           │  │
│  │       │                                                     │               │  │
│  └───────┼─────────────────────────────────────────────────────┼───────────────┘  │
│          │                                                     │                  │
│  ┌─── 在线训练循环（每个 step）─────────────────────────────────┼───────────────┐  │
│  │       │                                                     │               │  │
│  │       ▼                                                     │               │  │
│  │  ┌──────────────────────────────────────────┐               │               │  │
│  │  │  Flow Matching 加噪                       │               │               │  │
│  │  │                                          │               │               │  │
│  │  │  σ ~ sigmoid(N(0,1))  ← logit-normal    │               │               │  │
│  │  │  noise ~ N(0, I)                         │               │               │  │
│  │  │                                          │               │               │  │
│  │  │  x_t = (1-σ) * x_0 + σ * noise          │               │               │  │
│  │  └───────────┬──────────────────────────────┘               │               │  │
│  │              │                                               │               │  │
│  │              ▼                                               ▼               │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                    Transformer (DiT + LoRA)                           │   │  │
│  │  │                                                                      │   │  │
│  │  │   输入:  x_t [B,16,1,64,64]                                          │   │  │
│  │  │          model_timestep = 1 - σ  (去噪进度: 0=噪声, 1=干净)           │   │  │
│  │  │          text_embeds [N, 2560]  (概率 p 替换为零向量, CFG dropout)     │   │  │
│  │  │                                                                      │   │  │
│  │  │   输出:  pred ≈ x_0 - noise  (预测去噪方向)                           │   │  │
│  │  └──────────────────────────────────┬───────────────────────────────────┘   │  │
│  │                                     │                                       │  │
│  │                                     ▼                                       │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │  │
│  │  │  Loss & 反向传播                                                      │   │  │
│  │  │                                                                      │   │  │
│  │  │  target = x_0 - noise                                                │   │  │
│  │  │  loss = MSE(pred, target)    ← float32 计算                           │   │  │
│  │  │                                                                      │   │  │
│  │  │  loss.backward() → 只更新 LoRA A/B 矩阵 (其余参数冻结)               │   │  │
│  │  │  clip_grad_norm_(params, 1.0)                                        │   │  │
│  │  │  optimizer.step() (AdamW, cosine LR schedule)                        │   │  │
│  │  └──────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 训练 vs 推理的关键差异

| | 训练 | 推理 |
|---|---|---|
| **Transformer 运行次数** | 每个 sample 只跑 1 次 | 循环跑 N 步（Turbo=8, 普通=28） |
| **时间步** | 随机采样 σ，传入 `1-σ` | 按 schedule 从 1000→0 递减，传入 `(1000-t)/1000` |
| **噪声** | 随机构造 x_t 作为输入 | 从纯噪声出发，逐步去噪 |
| **文本条件** | 概率 p 替换为零向量（CFG dropout） | 同时跑正/负条件，组合输出 |
| **模型输出** | 直接与 target 算 MSE | 取负后传给 scheduler 做 Euler step |
| **VAE** | Encoder 离线编码图片为 latent | Decoder 将去噪后 latent 解码为图片 |
| **梯度** | 开启，只更新 LoRA 参数 | 关闭（`torch.no_grad()`） |

### 各模块职责

| 模块 | 职责 | Z-Image 具体实现 |
|------|------|-----------------|
| **Tokenizer** | 将文本切分为 token 序列 | Qwen2Tokenizer, vocab=151936 |
| **Text Encoder** | 将 token 映射为语义向量 | Qwen3Model, 36层, hidden_size=2560 |
| **VAE** | 图像 ↔ 潜在空间的压缩/解压 | AutoencoderKL, 16x 空间下采样 |
| **Transformer** | 在潜在空间预测去噪方向 | 30层 DiT, dim=3840, 30 heads |
| **Scheduler** | 控制去噪步骤的时间调度 | Flow Match Euler, 8步(Turbo) |

---

## 1. 文本编码：Tokenizer & Text Encoder

### 1.1 直觉

图像生成模型需要"理解"用户的文字描述。这个理解过程分两步：

1. **Tokenizer（分词器）**：把一句话拆成模型认识的最小单元。类比于把一本书拆成一个个字/词。例如 `"一只可爱的猫"` → `[一, 只, 可爱, 的, 猫]`。实际上现代 tokenizer 使用子词（subword）粒度，能处理任何语言的任何词汇。

2. **Text Encoder（文本编码器）**：把 token 序列转换成高维向量序列。每个 token 不再是一个孤立的符号，而是承载了上下文语义的 2560 维向量。"猫"这个 token 在"一只可爱的猫"和"薛定谔的猫"中，会得到不同的向量表示。

Z-Image 的独特之处：使用 **Qwen3**（一个大语言模型）作为文本编码器，而非传统的 CLIP。这意味着模型对语言的理解能力远超 CLIP——它能处理复杂的指令、长描述、甚至推理性质的 prompt。

### 1.2 数学

#### BPE (Byte-Pair Encoding) 分词

Tokenizer 使用 BPE 算法构建词表：

1. 初始化：将所有字符作为基础 token
2. 统计：找到语料中最频繁共现的相邻 token 对
3. 合并：将该 pair 合并为新 token
4. 重复步骤 2-3，直到词表达到目标大小（Z-Image: 151,936）

最终每个输入文本被编码为 token ID 序列：$\text{text} \rightarrow [t_1, t_2, ..., t_n], \quad t_i \in \{0, 1, ..., 151935\}$

#### Transformer Encoder

Text Encoder 本质是一个 Transformer，核心是 Multi-Head Self-Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = XW_Q$，$K = XW_K$，$V = XW_V$，$d_k$ 是 key 的维度。

#### GQA (Grouped Query Attention)

Z-Image 的 Qwen3 使用 GQA 而非标准 MHA：

- 标准 MHA：32 个 Q heads，32 个 K heads，32 个 V heads
- GQA：32 个 Q heads，**8 个 KV heads**（每 4 个 Q head 共享 1 组 KV）

$$\text{GQA}: \quad Q \in \mathbb{R}^{n \times 32 \times 128}, \quad K,V \in \mathbb{R}^{n \times 8 \times 128}$$

优势：减少 KV cache 的显存占用，推理速度更快，精度损失极小。

#### RoPE (Rotary Position Embedding)

位置信息通过旋转矩阵注入：

$$\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}$$

其中 $m$ 是位置索引，$\theta_j = 10000^{-2j/d}$ 是频率。直觉上，这让 attention score 天然具备相对位置感知能力：$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle$ 只取决于相对距离 $m-n$。

#### 为什么用倒数第二层？

Text Encoder 输出 `hidden_states[-2]`（第 35 层，共 36 层）而非最后一层。原因：

- 最后一层过度适配了语言模型的下一个 token 预测任务
- 倒数第二层保留了更通用的语义表示，更适合作为图像生成的条件信号

### 1.3 工程 Trick

**Trick 1: Chat Template 格式化**

不直接把 prompt 喂给模型，而是用 Qwen3 的对话格式包装：

```
<|im_start|>user
一只可爱的猫<|im_end|>
<|im_start|>assistant
```

为什么？因为 Qwen3 在这种格式上训练过，这样能激活模型最好的语义理解能力。`enable_thinking=True` 进一步激活模型的推理模式。

**Trick 2: Padding + Attention Mask 过滤**

所有 prompt 统一 pad 到 `max_length=512`，但通过 attention mask 标记哪些是真实 token、哪些是填充。编码后只保留真实 token 的 embedding：

```
输入: [真, 真, 真, PAD, PAD, ...]  (512个)
输出: [emb1, emb2, emb3]           (只保留3个)
```

**Trick 3: 维度对齐 (cap_embedder)**

Qwen3 输出维度是 2560，但 Transformer 的工作维度是 3840。通过一个投影层对齐：

$$\text{cap\_embedder}(x) = \text{Linear}(\text{RMSNorm}(x)), \quad \mathbb{R}^{2560} \rightarrow \mathbb{R}^{3840}$$

**Trick 4: Context Refiner**

文本 embedding 在和图像 token 拼接之前，先经过 2 层独立的 Transformer block 精炼。这些层没有时间步调制（modulation），专注于优化文本特征的内部表示。

### 1.4 代码

文本编码主流程（`src/zimage/pipeline.py:108-138`）：

```python
# 1. Chat Template 格式化
formatted_prompts = []
for p in prompt:
    messages = [{"role": "user", "content": p}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    formatted_prompts.append(formatted_prompt)

# 2. Tokenize (padding to 512)
text_inputs = tokenizer(
    formatted_prompts,
    padding="max_length", max_length=max_sequence_length,
    truncation=True, return_tensors="pt",
)
text_input_ids = text_inputs.input_ids.to(device)
prompt_masks = text_inputs.attention_mask.to(device).bool()

# 3. Encode (取倒数第二层)
prompt_embeds = text_encoder(
    input_ids=text_input_ids,
    attention_mask=prompt_masks,
    output_hidden_states=True,
).hidden_states[-2]

# 4. 去掉 padding，只保留有效 token
prompt_embeds_list = []
for i in range(len(prompt_embeds)):
    prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])
```

维度对齐与精炼（`src/zimage/transformer.py:315-326`）：

```python
# cap_embedder: RMSNorm + Linear(2560 → 3840)
self.cap_embedder = nn.Sequential(
    RMSNorm(cap_feat_dim, eps=norm_eps),
    nn.Linear(cap_feat_dim, dim, bias=True),
)

# context_refiner: 2层 Transformer Block (无时间步调制)
self.context_refiner = nn.ModuleList([
    ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
    for layer_id in range(n_refiner_layers)
])
```

---

## 2. VAE（变分自编码器）

### 2.1 直觉

一张 1024×1024 的 RGB 图像有 $1024 \times 1024 \times 3 = 3,145,728$ 个像素值。如果直接在像素空间做扩散生成，计算量极其恐怖。

VAE 的作用就像一个**智能压缩器**：

- **Encoder**：把 1024×1024×3 的图像压缩为 128×128×4 的潜在表示（latent），空间缩小 16 倍，总数据量缩小 $16^2 \times 3/4 = 192$ 倍
- **Decoder**：从潜在表示还原出高质量图像

类比：就像 JPEG 压缩，但 VAE 是可学习的——它学会了保留对人眼最重要的信息，丢弃冗余细节。与 JPEG 不同的是，VAE 的潜在空间是**连续的、结构化的**，相近的潜在向量对应相似的图像。

### 2.2 数学

#### 编码器 $q_\phi(z|x)$

编码器将图像 $x$ 映射为潜在分布的参数：

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) \cdot I)$$

网络输出均值 $\mu$ 和对数方差 $\log\sigma^2$（8 个通道：4个给 $\mu$，4 个给 $\log\sigma^2$）。

#### 重参数化技巧 (Reparameterization Trick)

为了能通过采样操作反向传播梯度：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

将随机性"外置"到 $\epsilon$，使得 $z$ 对 $\mu$ 和 $\sigma$ 可微。

#### 解码器 $p_\theta(x|z)$

解码器将潜在向量 $z$ 映射回像素空间：

$$\hat{x} = \text{Decoder}_\theta(z)$$

#### 训练目标

$$\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{重建损失}} + \underbrace{\beta \cdot D_{KL}(q_\phi(z|x) \| \mathcal{N}(0, I))}_{\text{KL 正则}}$$

KL 散度的解析形式：

$$D_{KL} = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

#### 空间下采样

对于 Z-Image 的 VAE（4 层 DownBlock，每层 stride-2 卷积）：

$$\text{下采样比例} = 2^{(\text{num\_blocks} - 1)} = 2^3 = 8$$

再加上 DiT 的 patch_size=2，总下采样比例为 $8 \times 2 = 16$。

因此：$1024 \times 1024$ 图像 → $64 \times 64$ patches（$128 \times 128$ latent with 4 channels）。

### 2.3 工程 Trick

**Trick 1: VAE 使用 FP32 精度**

```python
vae.to(device=device, dtype=torch.float32)  # 非 bfloat16!
```

为什么？VAE 解码是最终输出前的最后一步，精度损失会直接表现为图像伪影（色块、条纹）。FP32 的数值精度确保解码质量。

**Trick 2: scaling_factor = 0.18215**

训练好的 VAE 输出的 latent 方差较大。为了让 latent 更接近标准正态分布（方便扩散模型处理）：

$$z_{\text{scaled}} = z_{\text{raw}} \times 0.18215$$

这个常数是在大规模数据上统计得出的，使得 $z_{\text{scaled}}$ 的标准差接近 1。

**Trick 3: shift_factor 偏移补偿**

解码时需要逆操作：

$$z_{\text{decode}} = z_{\text{scaled}} / \text{scaling\_factor} + \text{shift\_factor}$$

shift_factor 补偿 latent 分布的均值偏移。

**Trick 4: Mid-block Self-Attention**

在 Encoder 和 Decoder 的中间（最小分辨率处）插入一个 Self-Attention 层。此时 feature map 最小（如 32×32），attention 的 $O(N^2)$ 复杂度可接受，但能显著提升全局一致性。

**Trick 5: GroupNorm + SiLU**

所有 ResNet block 使用 GroupNorm（32 groups）而非 BatchNorm：
- 不依赖 batch size，推理时行为一致
- SiLU（$x \cdot \text{sigmoid}(x)$）比 ReLU 更平滑，梯度流更好

### 2.4 代码

VAE 定义（`src/zimage/autoencoder.py:304`）：

```python
class AutoencoderKL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, latent_channels=4,
                 block_out_channels=(128,), layers_per_block=1,
                 norm_num_groups=32, scaling_factor=0.18215, ...):
        # Encoder: 图像 → latent (输出 2*latent_channels 用于 μ 和 logσ²)
        self.encoder = Encoder(in_channels, latent_channels * 2, ...)
        # Decoder: latent → 图像
        self.decoder = Decoder(latent_channels, out_channels, ...)
        # 1x1 卷积用于量化前后处理
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
```

Encoder 结构（`src/zimage/autoencoder.py:204`）：

```python
class Encoder(nn.Module):
    # 结构: Conv3x3 → [DownBlock × N] → MidBlock(ResNet + Attention + ResNet) → Norm → Conv3x3
    # 每个 DownBlock: [ResNet × layers_per_block] + Downsample(stride-2 Conv)
```

解码流程（`src/zimage/pipeline.py:278-293`）：

```python
# 逆缩放
latents = latents / vae.config.scaling_factor + shift_factor

# VAE 解码 (FP32)
image = vae.decode(latents).sample

# 后处理: [-1,1] → [0,1] → uint8 → PIL
image = (image / 2 + 0.5).clamp(0, 1)
image = image.permute(0, 2, 3, 1).float().numpy()
images = (image * 255).round().astype("uint8")
```

---

## 3. Transformer（DiT - Diffusion Transformer）

### 3.1 直觉

传统扩散模型（如 Stable Diffusion 1.x/2.x）使用 UNet 作为去噪网络。Z-Image 使用 **DiT（Diffusion Transformer）**，核心思想：

1. **Patchify**：把潜在图像切成 2×2 的小块（patch），每个 patch 展平为一个 "token"。128×128 的 latent → 64×64 = 4096 个 token。

2. **统一 Self-Attention**：文本 token 和图像 token 拼接成一个长序列，共同做 Self-Attention。这样文本和图像之间的关联通过 attention 自然建立——不需要单独的 Cross-Attention 层。

3. **AdaLN 调制**：时间步信息（"现在是去噪的第几步"）通过调制归一化层的 scale 和 gate 来注入，而非拼接或加法。

为什么 DiT 优于 UNet？
- **Scaling 更好**：Transformer 的性能随参数量/数据量增长更可预测
- **架构更统一**：文本和图像用同一种结构处理
- **训练更稳定**：没有 skip connection 的复杂交互

### 3.2 数学

#### Patch Embedding

将 latent $z \in \mathbb{R}^{C \times F \times H \times W}$ 转换为 token 序列：

$$z_{\text{patches}} = \text{Reshape}(z, [N, p_f \cdot p_h \cdot p_w \cdot C])$$

其中 $N = \frac{F}{p_f} \cdot \frac{H}{p_h} \cdot \frac{W}{p_w}$，patch_size $(p_h, p_w) = (2, 2)$，$p_f = 1$。

然后通过线性层投影：$x = z_{\text{patches}} W_{\text{embed}} + b, \quad W_{\text{embed}} \in \mathbb{R}^{(p^2 \cdot C) \times D}$

Z-Image: $C=16, p=2, D=3840$，所以 $W_{\text{embed}} \in \mathbb{R}^{64 \times 3840}$。

#### 3D RoPE

Z-Image 使用三维旋转位置编码，分别编码 (frame, height, width) 三个轴：

$$\text{RoPE}_{3D}(q, f, h, w) = \text{RoPE}_f(q[:32], f) \oplus \text{RoPE}_h(q[32:80], h) \oplus \text{RoPE}_w(q[80:128], w)$$

- axes_dims = [32, 48, 48]，总和 = 128 = head_dim
- axes_lens = [1536, 512, 512]，各轴最大支持长度
- $\theta = 256$（比标准 LLM 的 10000 小，因为空间位置范围较小）

#### AdaLN Modulation

标准 LayerNorm：$\text{LN}(x) = \frac{x - \mu}{\sigma}$

AdaLN 在此基础上用时间步 embedding 生成 scale 和 gate：

$$[\text{scale}_{msa}, \text{gate}_{msa}, \text{scale}_{mlp}, \text{gate}_{mlp}] = \text{Linear}(\text{adaln\_input})$$

$$\text{gate} = \tanh(\text{gate}), \quad \text{scale} = 1 + \text{scale}$$

$$\text{output} = x + \text{gate}_{msa} \cdot \text{Norm}(\text{Attn}(\text{Norm}(x) \cdot \text{scale}_{msa}))$$

#### SwiGLU FFN

比标准 FFN（Linear + ReLU + Linear）更强的前馈网络：

$$\text{FFN}(x) = W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x)$$

其中 $W_1, W_3 \in \mathbb{R}^{D \times D_{ff}}$，$W_2 \in \mathbb{R}^{D_{ff} \times D}$，$D_{ff} = \lfloor D \cdot 8/3 \rfloor = 10240$。

$\odot$ 是逐元素乘法。$W_3 x$ 作为"门控"信号，让网络学会选择性地传递信息。

#### Timestep Embedding

时间步 $t \in [0, 1]$ 首先通过正弦编码展开为高维向量：

$$\text{PE}(t, 2i) = \sin\left(\frac{t \cdot 1000}{\theta^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t \cdot 1000}{\theta^{2i/d}}\right)$$

其中 $d = 256$，$\theta = 10000$。然后通过 MLP：$\text{emb}(t) = \text{Linear}(\text{SiLU}(\text{Linear}(\text{PE}(t))))$

### 3.3 工程 Trick

**Trick 1: 三阶段架构**

```
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
│ Noise Refiner│     │ Context Refiner│     │   Main Layers    │
│  (2 layers)  │     │   (2 layers)   │     │   (30 layers)    │
│              │     │                │     │                  │
│ 只处理图像token │     │  只处理文本token  │     │ 联合处理拼接序列   │
│ 有AdaLN调制   │     │  无AdaLN调制    │     │  有AdaLN调制      │
└──────────────┘     └────────────────┘     └──────────────────┘
```

为什么这样设计？
- Noise Refiner：让图像 token 在拼接前先"理解"当前噪声水平
- Context Refiner：让文本特征独立精炼，不受噪声时间步干扰
- Main Layers：文本和图像联合 attention，实现跨模态交互

**Trick 2: QK-Norm**

对每个 head 的 Q 和 K 做 RMSNorm：

$$Q_{\text{norm}} = \text{RMSNorm}(Q), \quad K_{\text{norm}} = \text{RMSNorm}(K)$$

为什么？在大模型中，Q 和 K 的内积容易数值爆炸，导致 attention softmax 饱和。QK-Norm 确保内积值在合理范围内，大幅稳定训练。

**Trick 3: tanh 门控**

$$\text{gate} = \tanh(\text{raw\_gate})$$

将门控值限制在 $[-1, 1]$，防止某一层的残差更新过大导致训练发散。相比直接用 sigmoid（$[0,1]$），tanh 允许负值，给模型更多表达自由。

**Trick 4: 可插拔 Attention 后端**

Z-Image 支持多种 attention 实现，运行时切换：

| 后端 | 特点 |
|------|------|
| Flash Attention 2 | IO-aware，速度快 2-4x，省显存 |
| Flash Attention 3 | Hopper GPU 优化，更快 |
| PyTorch SDPA | 内置，兼容性最好 |
| MPS Flash | Apple Silicon 加速 |

**Trick 5: torch.compile**

对 Transformer 和 VAE 使用 `torch.compile`，让 PyTorch 编译器融合算子、优化内存访问，Hopper GPU 上可达亚秒级生成。

### 3.4 代码

模型定义（`src/zimage/transformer.py:266-336`）：

```python
class ZImageTransformer2DModel(nn.Module):
    def __init__(self, all_patch_size=(2,), in_channels=16, dim=3840,
                 n_layers=30, n_refiner_layers=2, n_heads=30, n_kv_heads=30,
                 cap_feat_dim=2560, ...):
        # Patch Embedding: 64 → 3840
        self.all_x_embedder = nn.ModuleDict({
            f"{ps}-{fps}": nn.Linear(fps * ps * ps * in_channels, dim)
            for ps, fps in zip(all_patch_size, all_f_patch_size)
        })

        # 时间步编码: scalar → 256-dim
        self.t_embedder = TimestepEmbedder(min(dim, 256), mid_size=1024)

        # 文本投影: 2560 → 3840
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        # 三阶段
        self.noise_refiner = nn.ModuleList([...])     # 2 layers, with modulation
        self.context_refiner = nn.ModuleList([...])   # 2 layers, without modulation
        self.layers = nn.ModuleList([...])            # 30 layers, with modulation
```

单个 Transformer Block（`src/zimage/transformer.py:143-200`）：

```python
class ZImageTransformerBlock(nn.Module):
    def forward(self, x, freqs_cis, adaln_input=None):
        # AdaLN: 时间步 → 4个调制向量
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp

        # Attention 分支
        attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)

        # FFN 分支
        ff_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        x = x + gate_mlp * self.ffn_norm2(ff_out)
        return x
```

Attention 实现（`src/zimage/transformer.py:86-130`）：

```python
class ZImageAttention(nn.Module):
    def forward(self, x, freqs_cis):
        qkv = self.qkv(x)  # Linear projection
        q, k, v = qkv.split([head_dim * n_heads, head_dim * n_kv_heads, head_dim * n_kv_heads], dim=-1)

        # QK-Norm
        q = self.q_norm(q.view(..., n_heads, head_dim))
        k = self.k_norm(k.view(..., n_kv_heads, head_dim))

        # RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Attention (dispatch to best available backend)
        out = dispatch_attention(q, k, v)
        return self.proj(out)
```

---

## 4. Scheduler（调度器 / 采样器）

### 4.1 直觉

Scheduler 控制着"如何从纯噪声一步步走向清晰图像"。

**Flow Matching vs DDPM 的直觉对比：**

- **DDPM（老方法）**：模型学习"预测噪声"，每步去掉一点噪声。数学复杂，需要 50-1000 步。
- **Flow Matching（新方法）**：模型学习"速度场"——在数据点和噪声之间画一条直线，模型预测当前位置的"速度方向"。数学更简洁，8 步即可。

想象一下：你在一个雾气弥漫的空间（纯噪声），想走到一个特定目标（生成图像）。
- DDPM：每步先估计"这里有多少雾"，减去一些
- Flow Matching：每步直接问"我该往哪个方向走"，然后迈一步

**Euler 方法**就是最简单的"迈步"策略：沿着当前速度方向，走 $\Delta t$ 的距离。

**为什么 8 步就够？** Z-Image-Turbo 是经过蒸馏的模型，每一步的速度预测更准确，加上优化过的时间调度（shift=3.0 让关键步骤获得更多"预算"）。

### 4.2 数学

#### Flow Matching ODE

定义从噪声 $x_0 \sim \mathcal{N}(0, I)$ 到数据 $x_1$ 的线性路径：

$$x_t = (1-t) \cdot x_0 + t \cdot x_1, \quad t \in [0, 1]$$

对应的速度场：$v(x_t, t) = x_1 - x_0$

模型学习拟合这个速度场：$v_\theta(x_t, t) \approx v(x_t, t)$

生成时求解 ODE：

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) = \text{noise}$$

#### Euler 离散化

将 $[0, 1]$ 区间分成 $N$ 步（Z-Image-Turbo: N=8）：

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

代码中：`prev_sample = sample + dt * model_output`

#### Sigma Schedule（时间步映射）

原始线性调度 $\sigma = t$ 不是最优的。通过一个非线性变换重新分配步长：

$$\sigma(t) = \frac{s \cdot t}{1 + (s-1) \cdot t}$$

其中 $s = \text{shift} = 3.0$。

这个变换的效果：
- 当 $t$ 接近 0（噪声最大）时，$\sigma$ 变化快 → 早期步骤跨度大
- 当 $t$ 接近 1（接近数据）时，$\sigma$ 变化慢 → 后期步骤更精细

直觉：早期"大刀阔斧"确定整体结构，后期"精雕细琢"完善细节。

#### Dynamic Shifting（分辨率自适应）

不同分辨率需要不同的 shift 强度：

$$\mu = m \cdot L + b$$

其中 $L$ 是图像序列长度（patch 数），$m$ 和 $b$ 由两个锚点决定：
- $L = 256$（256×256 图像）→ $\mu = 0.5$
- $L = 4096$（1024×1024 图像）→ $\mu = 1.15$

然后：$\sigma(t) = \frac{e^\mu}{e^\mu + (1/t - 1)^\sigma}$

### 4.3 工程 Trick

**Trick 1: 分辨率自适应 shift**

```python
def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

为什么？高分辨率图像有更多 patch，需要更强的 shift 来确保早期步骤足够"大胆"。

**Trick 2: CFG Truncation（引导截断）**

```python
if t_norm > cfg_truncation:  # 归一化时间 > 阈值
    current_guidance_scale = 0.0  # 关闭 CFG
```

在去噪早期（高噪声），CFG 的引导方向不可靠（因为信噪比太低），强行引导反而引入伪影。所以只在后期（低噪声、细节阶段）使用 CFG。

**Trick 3: CFG Normalization（引导归一化）**

$$\text{if } \|v_{\text{guided}}\| > c \cdot \|v_{\text{pos}}\|: \quad v_{\text{guided}} = v_{\text{guided}} \cdot \frac{c \cdot \|v_{\text{pos}}\|}{\|v_{\text{guided}}\|}$$

防止 CFG 把速度向量的模长吹得太大，导致颜色过饱和。

**Trick 4: 模型输出取反**

```python
noise_pred = -noise_pred.squeeze(2)
```

Z-Image 的 Transformer 预测的是"从数据指向噪声"的方向（$x_0 - x_1$），而 scheduler 期望"从噪声指向数据"的方向（$x_1 - x_0$），所以需要取反。

**Trick 5: Scheduler Step 用 FP32**

```python
sample = sample.to(torch.float32)  # 确保精度
prev_sample = sample + dt * model_output
```

虽然 Transformer 用 bfloat16 加速，但 ODE 积分的累积误差对精度敏感。用 FP32 做 scheduler step 防止数值漂移。

### 4.4 代码

调度器完整实现（`src/zimage/scheduler.py:28-151`）：

```python
class FlowMatchEulerDiscreteScheduler:
    def __init__(self, num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False):
        # 初始化 sigma schedule
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps)[::-1]
        sigmas = timesteps / num_train_timesteps

        if not use_dynamic_shifting:
            # 静态 shift: σ' = s*σ / (1 + (s-1)*σ)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

    def step(self, model_output, timestep, sample):
        """Euler step: x_{t+dt} = x_t + dt * v"""
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        dt = sigma_next - sigma  # 注意: sigma 递减, 所以 dt < 0
        prev_sample = sample + dt * model_output

        self._step_index += 1
        return prev_sample

    def time_shift(self, mu, sigma, t):
        """Dynamic shifting for resolution adaptation"""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
```

去噪循环（`src/zimage/pipeline.py:216-276`）：

```python
for i, t in enumerate(timesteps):
    if t == 0 and i == len(timesteps) - 1:
        continue

    # 时间步归一化到 [0, 1]
    timestep = (1000 - t.expand(batch_size)) / 1000

    # CFG: 拼接正负条件
    if do_classifier_free_guidance:
        latent_model_input = torch.cat([latents, latents])
        prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
    else:
        latent_model_input = latents

    # Transformer 前向推理
    latent_model_input = latent_model_input.unsqueeze(2)  # 添加 frame 维度
    model_out = transformer(latent_model_input_list, timestep, prompt_embeds_model_input)

    # CFG 组合
    if do_classifier_free_guidance:
        pos, neg = model_out.chunk(2)
        pred = pos + guidance_scale * (pos - neg)

    # Euler step (FP32)
    noise_pred = -noise_pred.squeeze(2)
    latents = scheduler.step(noise_pred.to(torch.float32), t, latents)[0]
```

---

## 5. 总结：从 Prompt 到 Image 的完整流程

以 "A cat on the moon, cinematic lighting" 在 1024×1024 分辨率、8 步 Turbo 模式为例：

### 步骤分解

| 步骤 | 操作 | 数据形状 | 耗时占比 |
|------|------|----------|---------|
| 1 | Tokenize + Chat Template | str → [1, 512] int | <1% |
| 2 | Text Encode (Qwen3, 36层) | [1, 512] → [1, ~30, 2560] | ~15% |
| 3 | 生成初始噪声 | → [1, 16, 128, 128] | <1% |
| 4 | Noise Refiner (2层) | [1, 4096, 3840] | ~3% |
| 5 | Context Refiner (2层) | [1, ~30, 3840] | ~1% |
| 6 | Main Transformer ×8 步 | [1, ~4126, 3840] ×8 | ~75% |
| 7 | VAE Decode | [1, 16, 128, 128] → [1, 3, 1024, 1024] | ~5% |
| 8 | 后处理 → PIL Image | tensor → PNG | <1% |

### 参数量概览

| 模块 | 参数量 | 精度 |
|------|--------|------|
| Text Encoder (Qwen3) | ~3.1B | bfloat16 |
| Transformer (DiT) | ~4.5B (估算: 30层 × 3840 dim) | bfloat16 |
| VAE | ~83M | float32 |
| **总计** | **~7.7B** | - |

### 关键设计哲学

1. **统一 Self-Attention** 取代 Cross-Attention：更简洁，scaling 更好
2. **Flow Matching** 取代 DDPM：数学更简洁，推理步数更少
3. **大语言模型** 取代 CLIP：语言理解能力质的飞跃
4. **三阶段精炼**：各司其职，互不干扰
5. **工程细节决定产品质量**：FP32 精度、QK-Norm、tanh 门控、CFG 截断...每个 trick 都在消除一类伪影

---

*本文基于 Z-Image 代码库 (commit 8559077) 编写，代码引用行号可能随版本更新变化。*
