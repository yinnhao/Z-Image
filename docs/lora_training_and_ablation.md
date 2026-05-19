# Z-Image LoRA 微调方案与消融实验设计

## 1. 任务概述

基于 Z-Image 基础模型，使用 LoRA（Low-Rank Adaptation）对 DiT（Diffusion Transformer）架构进行风格微调，训练数据为 `linoyts/3d_icon` 数据集（23 张 3D 图标风格图片），使模型生成具有 3D 图标风格的图片。

## 2. Baseline 方案

### 2.1 基础模型选择

| 模型 | 训练方式 | 推理步数 | CFG | 适合微调 |
|------|----------|----------|-----|----------|
| Z-Image | Pre-training + SFT | 50 | 是 | **是** |
| Z-Image-Turbo | + RL 蒸馏 | 8 | 否 | 否 |

**选择 Z-Image 基础模型**，原因：
- Turbo 经过蒸馏优化，velocity field 仅在少数固定步长上准确，不适合随机时间步训练
- Z-Image 在全时间步范围内都有准确的 velocity field，与 LoRA 训练的随机时间步采样兼容

### 2.2 关键技术细节

#### 2.2.1 时间步约定（核心修复）

Z-Image 推理 pipeline 中，传给 transformer 的时间步为：

```
model_timestep = (1000 - scheduler_t) / 1000
```

语义为：**0 = 纯噪声，1 = 干净图片**。

训练时采样 `sigma`（noise level，0=干净，1=纯噪声），需要转换后再传入模型：

```python
model_timestep = 1 - sigma  # 对齐推理约定
```

#### 2.2.2 训练目标

Flow matching 插值：
```
x_σ = (1 - σ) * x_0 + σ * ε
```

模型预测目标：
```
target = x_0 - ε = latents - noise
```

推理 pipeline 会对模型输出取反后传给 scheduler，因此模型直接输出 `latents - noise` 是正确的。

#### 2.2.3 LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| rank | 64 | 对 dim=3840 的模型提供充足容量 |
| alpha | 64 | alpha/rank=1.0，全强度 |
| 目标模块 | `to_q, to_k, to_v, to_out.0, w1, w2, w3` | 覆盖 attention + FFN |

目标模块选择依据：
- `to_q/k/v/out`：注意力层，控制 token 间的信息交互
- `w1/w2/w3`：FFN 层（SwiGLU 结构），承载主要的特征变换和风格信息

#### 2.2.4 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| resolution | 512 | 训练和推理分辨率一致 |
| batch_size | 1 | 显存限制 |
| gradient_accumulation | 4 | 有效 batch=4 |
| learning_rate | 1e-4 | LoRA 标准学习率 |
| epochs | 500 | 23 samples × 500 epochs = 2500 有效步 |
| warmup_steps | 100 | 占总步数 ~4% |
| lr_schedule | cosine decay | warmup 后余弦退火 |
| cfg_dropout_prob | 0.1 | 10% 概率丢弃文本条件，支持推理时 CFG |

#### 2.2.5 推理配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_inference_steps | 30 | Z-Image 基础模型设计用于多步推理 |
| guidance_scale | 3.5 | CFG 引导，增强 prompt 对齐 |
| resolution | 512 | 与训练一致 |

### 2.3 训练流程

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Precompute│───▶│ 2. Train     │───▶│ 3. Inference │
│   VAE encode │    │   LoRA       │    │   全集测试    │
│   Text encode│    │   500 epochs │    │   23 prompts │
└─────────────┘    └──────────────┘    └──────────────┘
```

### 2.4 监控

- **TensorBoard** 记录 `train/loss`、`train/lr`、`validation/sample`
- 每 100 步生成一张验证图片（固定 prompt + seed）
- 训练完成后对全部 23 个训练集 prompt 生成图片作为测试集

## 3. 消融实验设计

### 3.1 实验目标

验证 Baseline 方案中各个关键设计决策的贡献，量化每个因素对生成质量的影响。

### 3.2 实验矩阵

以 Baseline 为对照组，每次只改变一个变量：

| 实验 | 变量 | Baseline 值 | 消融值 | 验证目标 |
|------|------|-------------|--------|----------|
| **A** | 时间步约定 | `1 - sigma` | `sigma`（不修复） | 时间步修复的必要性 |
| **B** | 基础模型 | Z-Image | Z-Image-Turbo | 基础模型选择的影响 |
| **C** | LoRA 目标模块 | attention + FFN | 仅 attention | FFN 层对风格学习的贡献 |
| **D** | LoRA rank | 64 | 16 | 模型容量的影响 |
| **E** | 推理参数 | 30步 + CFG 3.5 | 8步 + 无CFG | 推理策略的影响（不重训） |

### 3.3 控制变量

所有消融实验共享以下不变条件：
- 训练数据：`linoyts/3d_icon`（23 samples）
- 训练 epochs：500
- 学习率：1e-4
- 随机种子：42
- 评估 prompt 集：训练集全部 23 个 prompt

### 3.4 评估方法

#### 定性评估
- 对全部 23 个训练集 prompt 生成图片
- 与训练集原图进行视觉对比
- 关注：风格一致性、细节清晰度、是否有失真/artifacts

#### 定量指标（通过 TensorBoard 对比）
- **训练 loss 曲线**：收敛速度和最终 loss 值
- **loss 下降幅度**：`(initial_loss - final_loss) / initial_loss`

### 3.5 预期结论

| 实验 | 预期效果 |
|------|----------|
| **A** (无时间步修复) | 生成结果与原模型几乎无区别，LoRA 不生效 |
| **B** (Turbo 模型) | 有部分效果但出现失真/artifacts |
| **C** (仅 attention) | 风格学习不完整，缺少 3D 质感细节 |
| **D** (rank=16) | 效果弱于 baseline，部分 prompt 风格不够强 |
| **E** (8步无CFG) | 图像质量下降，细节模糊，prompt 对齐度降低 |

### 3.6 执行命令

```bash
# 运行全部消融实验
bash run_ablation.sh

# 或逐个运行（按预估耗时排序）
bash run_ablation.sh E_8steps_no_cfg    # ~2min  (不需训练)
bash run_ablation.sh D_rank16           # ~30min (参数少，训练快)
bash run_ablation.sh C_attention_only   # ~30min
bash run_ablation.sh A_no_timestep_fix  # ~40min
bash run_ablation.sh B_turbo_model      # ~40min

# 查看结果对比
tensorboard --logdir output/ablation --port 6006 --bind_all
```

### 3.7 结果目录结构

```
output/ablation/
├── baseline/              -> output/lora_3d_icon/ (symlink)
│   ├── tensorboard/
│   ├── lora_weights/
│   └── samples/           (23 张生成图)
├── A_no_timestep_fix/
│   ├── config.json        (实验配置记录)
│   ├── tensorboard/       (loss 曲线)
│   ├── lora_weights/
│   └── samples/
│       ├── 00_the_tik_tok_logo_...png
│       ├── 01_a_group_of_colorful_...png
│       ├── ...
│       └── results.json   (prompt→文件映射)
├── B_turbo_model/
├── C_attention_only/
├── D_rank16/
└── E_8steps_no_cfg/
```

## 4. 关键发现与经验总结

### 4.1 开发过程中遇到的问题

| 问题 | 现象 | 根因 | 解决方案 |
|------|------|------|----------|
| LoRA 完全不生效 | 输出与原模型一致 | 时间步语义反转（sigma vs 1-sigma） | 训练时传 `1-sigma` 给模型 |
| 有效果但失真 | 3D 风格出现但有 artifacts | Turbo 模型 velocity field 不准确 | 换用 Z-Image 基础模型 |
| 风格不够强 | 与训练集差距大 | 只覆盖 attention 层 + rank 太小 | 增加 FFN 层 + rank=64 |

### 4.2 LoRA 微调 DiT 模型的最佳实践

1. **选择非蒸馏基础模型**：蒸馏模型的 velocity field 在随机时间步上不准确
2. **对齐时间步约定**：训练传给模型的时间步必须与推理 pipeline 语义一致
3. **覆盖 FFN 层**：DiT 中 FFN（w1/w2/w3）承载大量风格信息，仅训练 attention 不够
4. **推理步数与 CFG**：微调后的模型需要多步推理 + CFG 才能出好效果
5. **训练/推理分辨率一致**：避免不同分辨率下 latent shape 不匹配的问题
