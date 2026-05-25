"""
Z-Image LoRA Fine-tuning Script
基于 Flow Matching 的 DiT (Diffusion Transformer) 模型 LoRA 微调脚本。

本脚本实现了完整的 LoRA 微调流水线，包括：
1. 预计算阶段：将图片编码为 VAE latents，将文本编码为 embeddings（避免训练时重复计算）
2. 训练阶段：使用 Flow Matching 目标函数训练 LoRA 适配器
3. 推理阶段：加载训练好的 LoRA 权重生成图片

核心设计要点：
- 基于 Z-Image 基础模型（非 Turbo 蒸馏版），确保 velocity field 在全时间步范围内准确
- 时间步约定：模型接收的时间步为 (1-sigma)，其中 sigma 为噪声水平
  - 模型输入 0 = 纯噪声状态，模型输入 1 = 干净状态
  - 这与推理 pipeline 中 (1000-t)/1000 的约定一致
- LoRA 同时覆盖 attention 层 (to_q/k/v/out) 和 FFN 层 (w1/w2/w3)
- 使用 TensorBoard 监控训练 loss 和定期生成的验证图片

Usage:
    # Step 1: 预计算 latents 和 text embeddings（只需运行一次）
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute

    # Step 2: 训练 LoRA
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --train

    # Step 3: 推理验证（不传 --prompt 则对训练集全部 prompt 生成）
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference --prompt "a 3dicon, a cute cat"
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 训练配置
# ============================================================================

class TrainConfig:
    """训练超参数配置。所有参数均可通过命令行覆盖。"""

    # --- 模型与数据路径 ---
    model_path = "ckpts/Z-Image"            # 基础模型路径（使用非 Turbo 版本，适合微调）
    dataset_name = "linoyts/3d_icon"        # HuggingFace 数据集名称
    output_dir = "output/lora_3d_icon"      # 训练输出目录（权重、日志、验证图片）
    cache_dir = "output/precomputed"        # 预计算缓存目录（latents + embeddings）

    # --- 训练超参数 ---
    resolution = 512                        # 训练图片分辨率（推理时也使用相同分辨率）
    batch_size = 1                          # 单卡 batch size（受显存限制）
    gradient_accumulation_steps = 4         # 梯度累积步数，有效 batch = 1 * 4 = 4
    learning_rate = 1e-4                    # LoRA 学习率
    epochs = 500                            # 训练轮数（23 samples × 500 epochs ÷ 4 ≈ 2875 有效步）
    warmup_steps = 100                      # 学习率线性预热步数
    save_every_steps = 500                  # 每 N 步保存 checkpoint
    log_every_steps = 10                    # 每 N 步记录 loss 到 TensorBoard
    validate_every_steps = 100              # 每 N 步生成一张验证图片
    validate_prompt = "a 3dicon, a cute cat on purple background"  # 验证用的固定 prompt
    seed = 42                               # 随机种子（保证可复现）

    # --- LoRA 配置 ---
    lora_rank = 64                          # LoRA 秩（对 dim=3840 的模型，64 提供充足容量）
    lora_alpha = 64                         # LoRA 缩放因子（alpha/rank=1.0 表示全强度）
    lora_target_modules = [                 # LoRA 注入的目标模块
        "to_q", "to_k", "to_v", "to_out.0",  # Attention 层：控制 token 间信息交互
        "w1", "w2", "w3",                     # FFN 层（SwiGLU）：承载特征变换和风格信息
    ]

    # --- Flow Matching 配置 ---
    cfg_dropout_prob = 0.1                  # CFG dropout 概率（训练时随机丢弃文本条件，
                                            # 使模型支持推理时的 classifier-free guidance）

    # --- VAE 常量（从模型 config 中获取） ---
    vae_scaling_factor = 0.3611             # VAE latent 缩放因子
    vae_shift_factor = 0.1159              # VAE latent 偏移因子

    # --- 文本编码 ---
    max_sequence_length = 512               # 文本 token 最大长度


# ============================================================================
# 工具函数
# ============================================================================

def vae_encode(vae, images, scaling_factor, shift_factor):
    """
    将图片编码到 VAE latent 空间。

    过程：image [B,3,H,W] -> encoder -> quant_conv -> 取均值 -> 归一化
    输出 latent 形状：[B, 16, H/8, W/8]（16 通道，空间下采样 8 倍）

    Args:
        vae: VAE 模型（encoder 部分）
        images: 输入图片张量 [B, 3, H, W]，范围 [-1, 1]
        scaling_factor: latent 缩放因子
        shift_factor: latent 偏移因子
    Returns:
        latents: 编码后的 latent [B, 16, H/8, W/8]
    """
    with torch.no_grad():
        # 通过 VAE encoder 得到分布参数
        h = vae.encoder(images)
        if vae.quant_conv is not None:
            h = vae.quant_conv(h)
        # 取高斯分布的均值（不采样，确定性编码）
        mean, _ = h.chunk(2, dim=1)
        # 应用归一化：latent = (mean - shift) * scale
        latents = (mean - shift_factor) * scaling_factor
    return latents


def encode_text(tokenizer, text_encoder, prompts, max_length, device):
    """
    将文本 prompt 编码为 embedding 向量序列。

    使用 chat template 格式化 prompt（与推理 pipeline 保持一致），
    取 text encoder 倒数第二层 hidden states 作为文本表示，
    并移除 padding 位置，只保留有效 token 的 embedding。

    Args:
        tokenizer: 分词器
        text_encoder: 文本编码器
        prompts: 文本 prompt 列表
        max_length: 最大 token 长度
        device: 计算设备
    Returns:
        embeddings: 列表，每个元素为 [seq_len, hidden_dim] 的 embedding（已去除 padding）
    """
    # 使用 chat template 格式化（与推理时一致，确保 embedding 匹配）
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        formatted_prompts.append(formatted)

    # 分词
    text_inputs = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()

    # 编码：取倒数第二层 hidden states（比最后一层更通用）
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-2]

    # 提取有效 token 的 embedding（去掉 padding 部分）
    embeddings = []
    for i in range(len(hidden_states)):
        embeddings.append(hidden_states[i][attention_mask[i]].cpu())

    return embeddings


# ============================================================================
# 预计算阶段
# ============================================================================

def precompute(config: TrainConfig):
    """
    预计算 VAE latents 和 text embeddings，保存到磁盘。

    这一步将数据集中所有图片和文本提前编码好，避免训练时重复加载 VAE 和 text encoder，
    大幅减少训练时的显存占用和计算开销。

    输出文件：
        - cache_dir/latents.pt:    所有图片的 VAE latent 列表
        - cache_dir/embeddings.pt: 所有文本的 embedding 列表
        - cache_dir/prompts.json:  所有 prompt 文本（用于推理时的全集测试）
    """
    from datasets import load_dataset
    from utils import load_from_local_dir

    device = torch.device("cuda")
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型组件（VAE + text encoder）
    logger.info("Loading models for precomputation...")
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    vae = components["vae"]  # VAE 始终使用 float32 精度
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    # 加载数据集
    logger.info("Loading dataset...")
    ds = load_dataset(config.dataset_name, split="train")

    # 图片预处理：Resize -> CenterCrop -> ToTensor -> Normalize 到 [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
    ])

    logger.info(f"Precomputing {len(ds)} samples...")
    all_latents = []
    all_embeddings = []
    all_prompts = []

    for i, sample in enumerate(tqdm(ds, desc="Encoding")):
        img = sample["image"].convert("RGB")
        prompt = sample["prompt"]
        all_prompts.append(prompt)

        # 图片 -> VAE latent
        img_tensor = transform(img).unsqueeze(0).to(device, dtype=torch.float32)
        latent = vae_encode(vae, img_tensor, config.vae_scaling_factor, config.vae_shift_factor)
        all_latents.append(latent.squeeze(0).cpu())

        # 文本 -> embedding
        emb = encode_text(tokenizer, text_encoder, [prompt], config.max_sequence_length, device)
        all_embeddings.append(emb[0])

    # 保存到磁盘
    torch.save(all_latents, cache_dir / "latents.pt")
    torch.save(all_embeddings, cache_dir / "embeddings.pt")
    with open(cache_dir / "prompts.json", "w") as f:
        json.dump(all_prompts, f, ensure_ascii=False)

    logger.info(f"Saved to {cache_dir}/")
    logger.info(f"  latents.pt: {len(all_latents)} items, shape={all_latents[0].shape}")
    logger.info(f"  embeddings.pt: {len(all_embeddings)} items, shape[0]={all_embeddings[0].shape}")

    # 释放 GPU 显存
    del vae, text_encoder, tokenizer, components
    torch.cuda.empty_cache()


# ============================================================================
# 数据集
# ============================================================================

class PrecomputedDataset(Dataset):
    """
    预计算数据集：直接加载磁盘上的 latents 和 embeddings。

    避免训练时重复编码，每个样本包含：
    - latent: VAE 编码后的图片 latent [16, 64, 64]（512px 对应 64x64 latent）
    - embedding: 文本 embedding [seq_len, 2560]（变长，由 collate_fn 处理）
    - prompt: 原始文本（仅用于日志）
    """

    def __init__(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        self.latents = torch.load(cache_dir / "latents.pt", weights_only=True)
        self.embeddings = torch.load(cache_dir / "embeddings.pt", weights_only=True)
        with open(cache_dir / "prompts.json") as f:
            self.prompts = json.load(f)
        logger.info(f"Loaded {len(self.latents)} precomputed samples")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {
            "latent": self.latents[idx],
            "embedding": self.embeddings[idx],
            "prompt": self.prompts[idx],
        }


def collate_fn(batch):
    """
    自定义 collate 函数：处理变长 text embeddings。

    latents 可以直接 stack（形状固定），但 embeddings 长度不同（每个 prompt token 数不同），
    所以用 list 保持原样，在训练循环中逐个传给模型。
    """
    latents = torch.stack([item["latent"] for item in batch])
    embeddings = [item["embedding"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    return {"latents": latents, "embeddings": embeddings, "prompts": prompts}


# ============================================================================
# 训练阶段
# ============================================================================

def train(config: TrainConfig):
    """
    LoRA 训练主循环。

    训练流程：
    1. 加载基础 Transformer 模型，注入 LoRA 适配器
    2. 加载预计算的数据集
    3. 加载验证用的 VAE/text_encoder/scheduler（用于定期生成验证图片）
    4. Flow Matching 训练循环：
       - 随机采样时间步 sigma ~ sigmoid(N(0,1))
       - 构造加噪 latent: x_t = (1-sigma)*x_0 + sigma*noise
       - 模型预测 target = x_0 - noise
       - MSE loss 优化
    5. 定期记录 loss、生成验证图片、保存 checkpoint
    """
    from peft import LoraConfig, get_peft_model
    from utils.loader import load_sharded_safetensors
    from utils import load_from_local_dir
    from zimage.transformer import ZImageTransformer2DModel
    from zimage.pipeline import generate as zimage_generate
    from torch.utils.tensorboard import SummaryWriter
    import bitsandbytes as bnb
    import numpy as np

    device = torch.device("cuda")
    torch.manual_seed(config.seed)

    # ========== 1. 加载 Transformer 模型 ==========
    logger.info("Loading transformer...")
    model_dir = Path(config.model_path)
    transformer_dir = model_dir / "transformer"

    # 读取模型结构配置
    with open(transformer_dir / "config.json") as f:
        transformer_config = json.load(f)

    # 在 meta device 上实例化模型（不分配实际显存），再加载权重
    # 这比直接在 GPU 上实例化更省显存（避免同时存在随机初始化 + 权重两份参数）
    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            in_channels=transformer_config.get("in_channels", 16),
            dim=transformer_config.get("dim", 3840),
            n_layers=transformer_config.get("n_layers", 30),
            n_refiner_layers=transformer_config.get("n_refiner_layers", 2),
            n_heads=transformer_config.get("n_heads", 30),
            n_kv_heads=transformer_config.get("n_kv_heads", 30),
            cap_feat_dim=transformer_config.get("cap_feat_dim", 2560),
            all_patch_size=tuple(transformer_config.get("all_patch_size", [2])),
            all_f_patch_size=tuple(transformer_config.get("all_f_patch_size", [1])),
            norm_eps=transformer_config.get("norm_eps", 1e-5),
            qk_norm=transformer_config.get("qk_norm", True),
            rope_theta=transformer_config.get("rope_theta", 256.0),
            t_scale=transformer_config.get("t_scale", 1000.0),
            axes_dims=transformer_config.get("axes_dims", [32, 48, 48]),
            axes_lens=transformer_config.get("axes_lens", [1536, 512, 512]),
        )

    # 加载分片的 safetensors 权重，移动到 GPU
    state_dict = load_sharded_safetensors(transformer_dir)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    transformer.eval()

    # ========== 2. 注入 LoRA 适配器 ==========
    logger.info("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=config.lora_rank,                         # LoRA 秩（低秩分解的维度）
        lora_alpha=config.lora_alpha,               # 缩放因子（实际缩放 = alpha/rank）
        target_modules=config.lora_target_modules,  # 注入 LoRA 的目标线性层
        lora_dropout=0.0,                           # 不使用 dropout（数据集小，需要充分拟合）
        bias="none",                                # 不训练 bias
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()  # 打印可训练参数量（通常 < 1% 总参数）

    # 冻结所有非 LoRA 参数，只训练 LoRA 的 A/B 矩阵
    transformer.get_input_embeddings = lambda: None  # PEFT 兼容性 hack
    for param in transformer.parameters():
        if not param.requires_grad:
            param.requires_grad_(False)

    transformer.train()

    # ========== 3. 加载数据集 ==========
    dataset = PrecomputedDataset(config.cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,           # 打乱数据顺序
        collate_fn=collate_fn,  # 自定义 collate 处理变长 embedding
        num_workers=0,          # 数据已在内存中，不需要多进程
        drop_last=True,         # 丢弃不完整的最后一个 batch
    )

    # ========== 4. 加载验证组件 ==========
    # 训练中定期生成图片需要完整的推理 pipeline（VAE 解码 + text encode + scheduler）
    logger.info("Loading validation components (VAE, text_encoder, scheduler)...")
    val_components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    val_vae = val_components["vae"]
    val_text_encoder = val_components["text_encoder"]
    val_tokenizer = val_components["tokenizer"]
    val_scheduler = val_components["scheduler"]
    # 验证时使用训练中的 LoRA transformer，不需要额外加载
    del val_components["transformer"]
    torch.cuda.empty_cache()

    # ========== 5. 配置优化器和学习率调度 ==========
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    # 使用 8-bit AdamW 节省显存（bitsandbytes 实现）
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config.learning_rate, weight_decay=0.01)

    # 计算总步数和有效步数
    total_steps = config.epochs * math.ceil(len(dataset) / config.batch_size)
    effective_steps = total_steps // config.gradient_accumulation_steps

    # 学习率调度：线性 warmup + cosine decay
    def lr_lambda(step):
        if step < config.warmup_steps:
            # Warmup 阶段：从 0 线性增长到 1
            return step / max(1, config.warmup_steps)
        # Cosine decay 阶段：从 1 余弦衰减到 0
        progress = (step - config.warmup_steps) / max(1, effective_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========== 6. 训练循环 ==========
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    running_loss = 0.0

    # TensorBoard 记录器
    tb_log_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    logger.info(f"TensorBoard logs: {tb_log_dir}")
    logger.info(f"  Run: tensorboard --logdir {tb_log_dir}")

    logger.info(f"Starting training: {config.epochs} epochs, {total_steps} total steps")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
    logger.info(f"  Learning rate: {config.learning_rate}")

    for epoch in range(config.epochs):
        for step, batch in enumerate(dataloader):
            # --- 准备数据 ---
            latents = batch["latents"].to(device, dtype=torch.bfloat16)      # [B, 16, 64, 64]
            embeddings = [e.to(device, dtype=torch.bfloat16) for e in batch["embeddings"]]  # 变长列表
            B = latents.shape[0]

            # --- CFG Dropout ---
            # 以 cfg_dropout_prob 的概率将文本条件替换为零向量
            # 这使得模型在推理时可以使用 classifier-free guidance:
            #   output = uncond_pred + scale * (cond_pred - uncond_pred)
            if config.cfg_dropout_prob > 0:
                for i in range(B):
                    if torch.rand(1).item() < config.cfg_dropout_prob:
                        embeddings[i] = torch.zeros(1, embeddings[i].shape[-1], device=device, dtype=torch.bfloat16)

            # --- 采样时间步 ---
            # 使用 logit-normal 分布：sigma = sigmoid(N(0,1))
            # 这使得 sigma 集中在 0.5 附近，比均匀分布更关注中间噪声水平
            # sigma 表示噪声水平：0 = 干净图片，1 = 纯噪声
            sigma = torch.sigmoid(torch.randn(B, device=device, dtype=torch.bfloat16))

            # --- 构造加噪 latent（Flow Matching 线性插值）---
            # x_t = (1 - sigma) * x_0 + sigma * noise
            # 当 sigma=0: x_t = x_0（干净图片）
            # 当 sigma=1: x_t = noise（纯噪声）
            noise = torch.randn_like(latents)
            sigma_expand = sigma[:, None, None, None]  # [B, 1, 1, 1] 用于广播
            noisy_latents = (1 - sigma_expand) * latents + sigma_expand * noise

            # --- 训练目标 ---
            # 模型学习预测 (x_0 - noise) = (latents - noise)
            # 推理 pipeline 会对模型输出取反后作为 velocity 传给 scheduler
            target = latents - noise

            # --- 时间步转换（关键！）---
            # 推理 pipeline 传给模型的时间步为 (1000-t)/1000：
            #   - 纯噪声时 t=1000 -> 模型收到 0
            #   - 干净时 t=0    -> 模型收到 1
            # 训练中 sigma 表示噪声水平（1=噪声, 0=干净），语义相反
            # 所以传 (1 - sigma) 给模型，确保训练和推理约定一致
            model_timestep = 1 - sigma

            # --- 模型前向传播 ---
            # 输入格式：每个 latent 加一个 frame 维度 [C, 1, H, W]（兼容视频模型接口）
            x_list = [noisy_latents[i].unsqueeze(1) for i in range(B)]
            cap_feats_list = embeddings

            # 通过 PEFT 包装后的模型前向传播（自动应用 LoRA）
            pred_list, _ = transformer(x_list, model_timestep, cap_feats_list)
            pred = torch.stack([p.squeeze(1) for p in pred_list])  # [B, 16, 64, 64]

            # --- 计算 Loss ---
            # MSE loss：衡量模型预测与目标 (latents - noise) 的差距
            # 使用 float32 计算 loss 以避免 bfloat16 精度问题
            loss = F.mse_loss(pred.float(), target.float())
            loss = loss / config.gradient_accumulation_steps  # 梯度累积：loss 除以累积步数
            loss.backward()

            running_loss += loss.item()

            # --- 梯度累积 & 参数更新 ---
            # 每 gradient_accumulation_steps 个 micro-step 做一次实际的参数更新
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- 日志记录 ---
                if global_step % config.log_every_steps == 0:
                    avg_loss = running_loss / config.log_every_steps
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(f"Step {global_step} | Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running_loss = 0.0

                # --- 验证：生成样本图片 ---
                if global_step % config.validate_every_steps == 0:
                    logger.info(f"[Validate] Generating sample at step {global_step}...")
                    transformer.eval()  # 切换到 eval 模式（关闭 dropout 等）
                    with torch.no_grad():
                        val_generator = torch.Generator(device).manual_seed(config.seed)
                        val_images = zimage_generate(
                            transformer=transformer,
                            vae=val_vae,
                            text_encoder=val_text_encoder,
                            tokenizer=val_tokenizer,
                            scheduler=val_scheduler,
                            prompt=config.validate_prompt,
                            height=config.resolution,
                            width=config.resolution,
                            num_inference_steps=30,
                            guidance_scale=3.5,
                            generator=val_generator,
                        )
                    # 记录到 TensorBoard（可在 Images tab 中查看）
                    val_img = val_images[0]
                    val_img_np = np.array(val_img).transpose(2, 0, 1)  # HWC -> CHW (TensorBoard 格式)
                    writer.add_image("validation/sample", val_img_np, global_step)
                    # 同时保存到磁盘（方便直接查看）
                    val_dir = output_dir / "validation"
                    val_dir.mkdir(parents=True, exist_ok=True)
                    val_img.save(val_dir / f"step_{global_step:06d}.png")
                    logger.info(f"[Validate] Saved: {val_dir}/step_{global_step:06d}.png")
                    transformer.train()  # 切回训练模式

                # --- 保存 checkpoint ---
                if global_step % config.save_every_steps == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    transformer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

    # ========== 7. 保存最终权重 ==========
    final_path = output_dir / "lora_weights"
    transformer.save_pretrained(final_path)
    writer.close()
    logger.info(f"Training complete! LoRA weights saved to {final_path}")
    logger.info(f"TensorBoard logs: tensorboard --logdir {tb_log_dir}")


# ============================================================================
# 推理阶段
# ============================================================================

def inference(config: TrainConfig, lora_path: str, prompt: str = None):
    """
    使用基础模型 + LoRA 权重生成图片。

    流程：
    1. 加载基础模型全部组件
    2. 加载 LoRA 权重并合并到基础模型（merge_and_unload 后不再需要 PEFT）
    3. 对指定 prompt（或全部训练集 prompt）逐个生成图片

    Args:
        config: 训练配置（用于获取模型路径、分辨率等）
        lora_path: LoRA 权重路径
        prompt: 单个 prompt（为 None 则使用训练集全部 prompt 作为测试集）
    """
    from peft import PeftModel, LoraConfig
    from utils import load_from_local_dir
    from zimage.pipeline import generate

    device = torch.device("cuda")

    # 加载基础模型全部组件
    logger.info("Loading base model...")
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    transformer = components["transformer"]
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    scheduler = components["scheduler"]

    # 加载 LoRA 权重并合并到基础模型
    # merge_and_unload() 将 LoRA 的 delta 矩阵 (B@A * alpha/rank) 加到原始权重上
    # 合并后模型与普通模型完全相同，推理时无额外开销
    logger.info(f"Loading LoRA weights from {lora_path}...")
    transformer = PeftModel.from_pretrained(transformer, lora_path)
    transformer = transformer.merge_and_unload()
    transformer.eval()

    # 确定要生成的 prompt 列表
    if prompt is not None:
        # 指定了单个 prompt
        prompts = [prompt]
    else:
        # 未指定 prompt：使用训练集全部 prompt 作为测试集
        prompts_file = Path(config.cache_dir) / "prompts.json"
        if prompts_file.exists():
            with open(prompts_file) as f:
                prompts = json.load(f)
            logger.info(f"Using all {len(prompts)} training prompts as test set")
        else:
            prompts = ["a 3dicon, a cute cat icon on a purple background"]

    # 逐个 prompt 生成图片
    output_dir = Path(config.output_dir) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] Generating: {p}")
        # 使用固定种子确保可复现
        generator = torch.Generator(device).manual_seed(config.seed)

        images = generate(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=p,
            height=config.resolution,
            width=config.resolution,
            num_inference_steps=30,   # Z-Image 基础模型使用 30 步
            guidance_scale=3.5,       # CFG 引导强度
            generator=generator,
        )

        # 保存图片（文件名包含 prompt 关键词，方便识别）
        short_name = p.replace("a 3dicon, ", "").replace(" ", "_")[:40]
        save_path = output_dir / f"{i:02d}_{short_name}.png"
        images[0].save(save_path)
        logger.info(f"  Saved: {save_path}")

    logger.info(f"All samples saved to {output_dir}/")


# ============================================================================
# 入口
# ============================================================================

def main():
    """命令行入口，支持 --precompute / --train / --inference 三种模式。"""
    parser = argparse.ArgumentParser(description="Z-Image LoRA Fine-tuning")
    # 运行模式（三选一）
    parser.add_argument("--precompute", action="store_true", help="预计算 VAE latents 和 text embeddings")
    parser.add_argument("--train", action="store_true", help="运行 LoRA 训练")
    parser.add_argument("--inference", action="store_true", help="使用 LoRA 权重生成图片")
    parser.add_argument("--lora_path", type=str, default="output/lora_3d_icon/lora_weights",
                        help="LoRA 权重路径（推理时使用）")
    parser.add_argument("--prompt", type=str, default=None,
                        help="推理 prompt（不指定则对训练集全部 prompt 生成）")

    # 超参数覆盖（可选，用于快速实验）
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--rank", type=int, default=None, help="覆盖 LoRA rank（同时设置 alpha=rank）")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--resolution", type=int, default=None, help="覆盖分辨率")

    args = parser.parse_args()
    config = TrainConfig()

    # 应用命令行覆盖
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.rank:
        config.lora_rank = args.rank
        config.lora_alpha = args.rank  # 保持 alpha/rank = 1
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.resolution:
        config.resolution = args.resolution

    # 执行对应模式
    if args.precompute:
        precompute(config)
    elif args.train:
        train(config)
    elif args.inference:
        inference(config, args.lora_path, args.prompt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
