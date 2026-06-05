"""
Z-Image-Edit LoRA 训练脚本
基于 Z-Image 的图像编辑能力训练：源图 + 编辑指令 → 编辑后图片

架构设计（参考 Z-Image 论文 Figure 10）：
1. SigLip-2（frozen 视觉编码器）→ Semantic Processor（投影层）→ 语义 tokens
2. 源图 VAE latent → 作为 frame 1 拼接到 noised target（frame 0）的 T 维度

统一序列组成：
  [noised_target_tokens, source_vae_tokens, text_tokens, semantic_tokens]
  其中 noised_target + source_vae 经过 noise_refiner（有 timestep 调制）
  text + semantic 经过 context_refiner（无 timestep）

Usage:
    # Step 1: 预计算 latents、text embeddings 和 SigLip-2 特征
    CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --precompute --data_dir edit_data/

    # Step 2: 训练
    CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --train

    # Step 3: 推理
    python train_edit_lora.py --inference --source input.png --prompt "turn the logo green"
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 训练配置
# ============================================================================


class EditTrainConfig:
    """图像编辑 LoRA 训练超参数配置。"""

    # --- 模型与数据路径 ---
    model_path = "ckpts/Z-Image"
    data_dir = "edit_data"                          # 编辑数据集目录
    output_dir = "output/edit_lora"                 # 训练输出目录
    cache_dir = "output/edit_precomputed"           # 预计算缓存目录

    # --- 数据集选项 ---
    prompt_level = "medium"                         # TSV 数据集 prompt 级别: short/medium/long
    max_samples = None                              # 最大预计算样本数（None=全部）

    # --- SigLip-2 ---
    siglip_model_name = "google/siglip2-large-patch16-384"
    siglip_dim = 1024                               # siglip2-large-patch16 hidden dim
    siglip_num_tokens = 576                         # 384px / 16 patch = 24×24 = 576

    # --- 训练超参数 ---
    resolution = 512
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    epochs = 500
    warmup_steps = 100
    save_every_steps = 500
    log_every_steps = 10
    validate_every_steps = 200
    val_samples_per_run = 10                        # 每次验证推理多少样本（None=全部验证集）
    seed = 42

    # --- LoRA 配置 ---
    lora_rank = 64
    lora_alpha = 64
    lora_target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "w1", "w2", "w3",
    ]

    # --- Flow Matching ---
    cfg_dropout_prob = 0.1

    # --- VAE 常量 ---
    vae_scaling_factor = 0.3611
    vae_shift_factor = 0.1159

    # --- 文本编码 ---
    max_sequence_length = 512

    # --- 推理参数 ---
    inference_steps = 50
    guidance_scale = 5.0


# ============================================================================
# Semantic Processor
# ============================================================================


class SemanticProcessor(nn.Module):
    """
    SigLip-2 视觉特征 → cap_feat_dim (2560) 的投影层。

    输出 2560 维，与 text embedding 拼接后一起经过 transformer 内部的
    cap_embedder（RMSNorm + Linear 2560→3840）和 context_refiner，
    无需修改 transformer 源码。
    """

    def __init__(self, siglip_dim=1024, output_dim=2560):
        super().__init__()
        self.norm = nn.LayerNorm(siglip_dim)
        self.proj = nn.Linear(siglip_dim, output_dim)

    def forward(self, x):
        # x: [B, N_tokens, siglip_dim] -> [B, N_tokens, output_dim]
        return self.proj(self.norm(x))


# ============================================================================
# 工具函数
# ============================================================================


def vae_encode(vae, images, scaling_factor, shift_factor):
    """将图片编码到 VAE latent 空间。输出 [B, 16, H/8, W/8]。"""
    with torch.no_grad():
        h = vae.encoder(images)
        if vae.quant_conv is not None:
            h = vae.quant_conv(h)
        mean, _ = h.chunk(2, dim=1)
        latents = (mean - shift_factor) * scaling_factor
    return latents


def encode_text(tokenizer, text_encoder, prompts, max_length, device):
    """将文本编码为 embedding 列表，每个元素 [seq_len, 2560]（去除 padding）。"""
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        formatted_prompts.append(formatted)

    text_inputs = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-2]

    embeddings = []
    for i in range(len(hidden_states)):
        embeddings.append(hidden_states[i][attention_mask[i]].cpu())

    return embeddings


def extract_siglip_features(siglip_model, siglip_processor, images, device):
    """
    提取 SigLip-2 视觉特征。

    Args:
        siglip_model: SigLip-2 模型（frozen）
        siglip_processor: SigLip-2 processor
        images: PIL Image 列表
        device: 计算设备
    Returns:
        features: [B, N_tokens, siglip_dim] 视觉特征
    """
    with torch.no_grad():
        inputs = siglip_processor(images=images, return_tensors="pt").to(device)
        outputs = siglip_model.vision_model(pixel_values=inputs["pixel_values"])
        # 使用所有 patch tokens（不含 CLS token，SigLip-2 没有 CLS）
        features = outputs.last_hidden_state  # [B, N_tokens, hidden_dim]
    return features.cpu()


# ============================================================================
# 预计算阶段
# ============================================================================


def load_tsv_dataset(data_dir, prompt_level="short", max_samples=None, shard_id=None, num_shards=None, start_idx=0):
    """
    加载 TSV 格式的编辑数据集。

    TSV 格式（8 字段，tab 分隔）：
        field 0: md5 ID
        field 1: 数据集标签
        field 2: JSON metadata
        field 3: 源图 base64 JPEG
        field 4: 目标图 base64 JPEG
        field 5: 短 prompt（中文）
        field 6: 中等 prompt（中文）
        field 7: 详细 prompt（中文）

    Args:
        data_dir: 数据集根目录（含 part-XX 文件）
        prompt_level: 使用哪个级别的 prompt ("short"=field5, "medium"=field6, "long"=field7)
        max_samples: 最大样本数（None=全部加载）
        shard_id: 当前分片 ID（None=不分片）
        num_shards: 总分片数
        start_idx: 全局起始索引（用于跨子目录连续编号）
    Returns:
        生成器，每次 yield (source_img: PIL.Image, target_img: PIL.Image, prompt: str, next_idx: int)
        最后一个 yield 后可通过 .gi_frame 无法获取 next_idx，所以改为返回 tuple
    """
    import base64
    from io import BytesIO

    prompt_field_map = {"short": 5, "medium": 6, "long": 7}
    prompt_idx = prompt_field_map.get(prompt_level, 5)

    data_dir = Path(data_dir)
    part_files = sorted(data_dir.glob("part-*"))
    if not part_files:
        raise FileNotFoundError(f"No part-* files found in {data_dir}")

    count = 0
    global_idx = start_idx
    for part_file in part_files:
        with open(part_file) as f:
            for line in f:
                if max_samples and count >= max_samples:
                    return
                line = line.strip()
                if not line:
                    continue

                # 分片过滤：在 base64 解码之前跳过不属于本 shard 的样本
                if shard_id is not None and global_idx % num_shards != shard_id:
                    global_idx += 1
                    continue
                global_idx += 1

                parts = line.split("\t")
                if len(parts) < 8:
                    continue

                try:
                    source_img = Image.open(BytesIO(base64.b64decode(parts[3]))).convert("RGB")
                    target_img = Image.open(BytesIO(base64.b64decode(parts[4]))).convert("RGB")
                    prompt = parts[prompt_idx]
                    if not prompt:
                        prompt = parts[5]  # fallback to short
                    yield source_img, target_img, prompt
                    count += 1
                except Exception as e:
                    logger.warning(f"Skipping sample {parts[0]}: {e}")
                    continue


def load_jsonl_dataset(data_dir):
    """加载 metadata.jsonl + source/target 目录格式的数据集。"""
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.jsonl"
    samples = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    for sample in samples:
        source_path = data_dir / "source" / sample["source"]
        target_path = data_dir / "target" / sample["target"]
        source_img = Image.open(source_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        yield source_img, target_img, sample["prompt"]


def detect_dataset_format(data_dir):
    """检测数据集格式：TSV (part-XX files) 或 JSONL (metadata.jsonl)。"""
    data_dir = Path(data_dir)
    if (data_dir / "metadata.jsonl").exists():
        return "jsonl"
    if list(data_dir.glob("part-*")):
        return "tsv"
    # 检查是否是包含多个子目录的根目录
    subdirs = [d for d in data_dir.iterdir() if d.is_dir() and list(d.glob("part-*"))]
    if subdirs:
        return "tsv_multi"
    raise ValueError(f"Cannot detect dataset format in {data_dir}. "
                     f"Expected metadata.jsonl or part-* files.")


def precompute(config: EditTrainConfig, shard_id=None, num_shards=None):
    """
    预计算 VAE latents、text embeddings 和 SigLip-2 特征，保存到磁盘。

    支持两种数据集格式：
    1. TSV 格式（百度内部）：part-XX 文件，每行 8 个 tab 分隔字段，图片为 base64
    2. JSONL 格式：metadata.jsonl + source/target 图片目录

    对于 TSV 多子目录格式（如 apple_full/ + log_full/ + text_data/），
    会自动扫描所有子目录。
    """
    from transformers import AutoModel, AutoProcessor
    from utils import load_from_local_dir

    device = torch.device("cuda")
    data_dir = Path(config.data_dir)
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 检测数据集格式并加载样本迭代器
    fmt = detect_dataset_format(data_dir)
    if fmt == "jsonl":
        samples_iter = list(load_jsonl_dataset(data_dir))
        total_count = len(samples_iter)
    elif fmt == "tsv":
        # 先数一下总数
        total_count = 0
        count_file = data_dir / "count.txt"
        if count_file.exists():
            with open(count_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        total_count += int(parts[1])
        else:
            total_count = sum(1 for _ in load_tsv_dataset(data_dir, config.prompt_level, config.max_samples))
        samples_iter = None  # will iterate lazily
    elif fmt == "tsv_multi":
        # 多子目录：合并所有含 part-* 文件的子目录
        subdirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and list(d.glob("part-*")))
        total_count = 0
        for sd in subdirs:
            count_file = sd / "count.txt"
            if count_file.exists():
                with open(count_file) as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 2:
                            total_count += int(parts[1])
        samples_iter = None
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if config.max_samples:
        total_count = min(total_count, config.max_samples)
    logger.info(f"Dataset format: {fmt}, estimated {total_count} samples")

    # 加载 Z-Image 模型组件
    logger.info("Loading Z-Image models (VAE + text encoder)...")
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    # 加载 SigLip-2
    logger.info(f"Loading SigLip-2: {config.siglip_model_name}...")
    siglip_processor = AutoProcessor.from_pretrained(config.siglip_model_name)
    siglip_model = AutoModel.from_pretrained(config.siglip_model_name).to(device)
    siglip_model.eval()

    # 图片预处理 transform（用于 VAE 编码）
    transform = transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
    ])

    # 构建样本迭代器（分片过滤已在 load_tsv_dataset 内部完成，按行号跳过不解码）
    def get_samples():
        if fmt == "jsonl":
            for idx, item in enumerate(samples_iter):
                if shard_id is not None and idx % num_shards != shard_id:
                    continue
                yield item
        elif fmt == "tsv":
            yield from load_tsv_dataset(data_dir, config.prompt_level, config.max_samples,
                                        shard_id=shard_id, num_shards=num_shards)
        elif fmt == "tsv_multi":
            count = 0
            cumulative_idx = 0
            subdirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and list(d.glob("part-*")))
            for sd in subdirs:
                logger.info(f"  Processing subdir: {sd.name}")
                # 获取该子目录的样本数以计算 cumulative offset
                count_file = sd / "count.txt"
                subdir_total = 0
                if count_file.exists():
                    with open(count_file) as f:
                        for line in f:
                            p = line.strip().split(",")
                            if len(p) == 2:
                                subdir_total += int(p[1])

                for item in load_tsv_dataset(sd, config.prompt_level,
                                             shard_id=shard_id, num_shards=num_shards,
                                             start_idx=cumulative_idx):
                    if config.max_samples and count >= config.max_samples:
                        return
                    yield item
                    count += 1
                cumulative_idx += subdir_total

    shard_info = f" (shard {shard_id}/{num_shards})" if shard_id is not None else ""
    shard_count = (total_count + num_shards - 1) // num_shards if num_shards else total_count
    logger.info(f"Precomputing samples{shard_info} (resolution={config.resolution})...")
    all_source_latents = []
    all_target_latents = []
    all_semantic_features = []
    all_text_embeddings = []
    all_prompts = []

    # Batch 化处理提高 GPU 利用率
    BATCH_SIZE = 8  # 每次攒够 8 个样本统一送 GPU
    batch_sources = []
    batch_targets = []
    batch_source_imgs = []
    batch_prompts_buf = []

    def flush_batch():
        """将攒好的 batch 统一送 GPU 处理。"""
        if not batch_sources:
            return
        B = len(batch_sources)

        # VAE batch encode 源图
        src_batch = torch.stack(batch_sources).to(device, dtype=torch.float32)
        src_latents = vae_encode(vae, src_batch, config.vae_scaling_factor, config.vae_shift_factor)
        for i in range(B):
            all_source_latents.append(src_latents[i].cpu())

        # VAE batch encode 目标图
        tgt_batch = torch.stack(batch_targets).to(device, dtype=torch.float32)
        tgt_latents = vae_encode(vae, tgt_batch, config.vae_scaling_factor, config.vae_shift_factor)
        for i in range(B):
            all_target_latents.append(tgt_latents[i].cpu())

        # SigLip-2 batch 特征提取
        sem_feats = extract_siglip_features(siglip_model, siglip_processor, batch_source_imgs, device)
        for i in range(B):
            all_semantic_features.append(sem_feats[i])

        # 文本 batch 编码
        embs = encode_text(tokenizer, text_encoder, batch_prompts_buf, config.max_sequence_length, device)
        all_text_embeddings.extend(embs)

        batch_sources.clear()
        batch_targets.clear()
        batch_source_imgs.clear()
        batch_prompts_buf.clear()

    processed = 0
    for source_img, target_img, prompt in tqdm(get_samples(), desc=f"Encoding{shard_info}", total=shard_count):
        all_prompts.append(prompt)

        # CPU 预处理（transform + 攒 batch）
        batch_sources.append(transform(source_img))
        batch_targets.append(transform(target_img))
        batch_source_imgs.append(source_img)
        batch_prompts_buf.append(prompt)

        if len(batch_sources) >= BATCH_SIZE:
            flush_batch()

        processed += 1

    # 处理剩余不足一个 batch 的样本
    flush_batch()

    logger.info(f"Processed {processed} samples{shard_info}")

    # 保存到磁盘（分片模式保存带 shard 后缀的文件）
    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    torch.save(all_source_latents, cache_dir / f"source_latents{suffix}.pt")
    torch.save(all_target_latents, cache_dir / f"target_latents{suffix}.pt")
    torch.save(all_semantic_features, cache_dir / f"semantic_features{suffix}.pt")
    torch.save(all_text_embeddings, cache_dir / f"text_embeddings{suffix}.pt")
    with open(cache_dir / f"prompts{suffix}.json", "w") as f:
        json.dump(all_prompts, f, ensure_ascii=False)

    logger.info(f"Saved to {cache_dir}/ (suffix='{suffix}')")
    logger.info(f"  source_latents{suffix}.pt: {len(all_source_latents)} items, shape={all_source_latents[0].shape}")
    logger.info(f"  target_latents{suffix}.pt: {len(all_target_latents)} items, shape={all_target_latents[0].shape}")
    logger.info(f"  semantic_features{suffix}.pt: {len(all_semantic_features)} items, shape={all_semantic_features[0].shape}")
    logger.info(f"  text_embeddings{suffix}.pt: {len(all_text_embeddings)} items, shape[0]={all_text_embeddings[0].shape}")

    del vae, text_encoder, tokenizer, siglip_model, siglip_processor, components
    torch.cuda.empty_cache()


def merge_shards(config: EditTrainConfig, num_shards: int):
    """合并多个 shard 的预计算结果为统一文件。"""
    cache_dir = Path(config.cache_dir)
    logger.info(f"Merging {num_shards} shards from {cache_dir}...")

    all_source_latents = []
    all_target_latents = []
    all_semantic_features = []
    all_text_embeddings = []
    all_prompts = []

    for i in range(num_shards):
        suffix = f"_shard{i}"
        sl = torch.load(cache_dir / f"source_latents{suffix}.pt", weights_only=True)
        tl = torch.load(cache_dir / f"target_latents{suffix}.pt", weights_only=True)
        sf = torch.load(cache_dir / f"semantic_features{suffix}.pt", weights_only=True)
        te = torch.load(cache_dir / f"text_embeddings{suffix}.pt", weights_only=True)
        with open(cache_dir / f"prompts{suffix}.json") as f:
            p = json.load(f)
        all_source_latents.extend(sl)
        all_target_latents.extend(tl)
        all_semantic_features.extend(sf)
        all_text_embeddings.extend(te)
        all_prompts.extend(p)
        logger.info(f"  Shard {i}: {len(sl)} samples")

    torch.save(all_source_latents, cache_dir / "source_latents.pt")
    torch.save(all_target_latents, cache_dir / "target_latents.pt")
    torch.save(all_semantic_features, cache_dir / "semantic_features.pt")
    torch.save(all_text_embeddings, cache_dir / "text_embeddings.pt")
    with open(cache_dir / "prompts.json", "w") as f:
        json.dump(all_prompts, f, ensure_ascii=False)

    logger.info(f"Merged {len(all_source_latents)} total samples → {cache_dir}/")

    # 清理 shard 文件
    for i in range(num_shards):
        suffix = f"_shard{i}"
        for name in ["source_latents", "target_latents", "semantic_features", "text_embeddings"]:
            (cache_dir / f"{name}{suffix}.pt").unlink(missing_ok=True)
        (cache_dir / f"prompts{suffix}.json").unlink(missing_ok=True)
    logger.info("Cleaned up shard files.")


# ============================================================================
# 数据集
# ============================================================================


class EditPrecomputedDataset(Dataset):
    """
    预计算数据集，包含：
    - source_latent: 源图 VAE latent [16, H/8, W/8]
    - target_latent: 目标图 VAE latent [16, H/8, W/8]
    - semantic_feature: SigLip-2 特征 [N_tokens, siglip_dim]
    - text_embedding: 文本 embedding [seq_len, 2560]
    """

    def __init__(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        self.source_latents = torch.load(cache_dir / "source_latents.pt", weights_only=True)
        self.target_latents = torch.load(cache_dir / "target_latents.pt", weights_only=True)
        self.semantic_features = torch.load(cache_dir / "semantic_features.pt", weights_only=True)
        self.text_embeddings = torch.load(cache_dir / "text_embeddings.pt", weights_only=True)
        with open(cache_dir / "prompts.json") as f:
            self.prompts = json.load(f)
        logger.info(f"Loaded {len(self.source_latents)} precomputed edit samples")

    def __len__(self):
        return len(self.source_latents)

    def __getitem__(self, idx):
        return {
            "source_latent": self.source_latents[idx],
            "target_latent": self.target_latents[idx],
            "semantic_feature": self.semantic_features[idx],
            "text_embedding": self.text_embeddings[idx],
            "prompt": self.prompts[idx],
        }


def edit_collate_fn(batch):
    """处理变长 text embeddings 和 semantic features 的 collate。"""
    source_latents = torch.stack([item["source_latent"] for item in batch])
    target_latents = torch.stack([item["target_latent"] for item in batch])
    semantic_features = [item["semantic_feature"] for item in batch]
    text_embeddings = [item["text_embedding"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    return {
        "source_latents": source_latents,
        "target_latents": target_latents,
        "semantic_features": semantic_features,
        "text_embeddings": text_embeddings,
        "prompts": prompts,
    }


# ============================================================================
# 验证：完整去噪推理
# ============================================================================


def validate_full(
    transformer,
    semantic_processor,
    val_dataset,
    vae,
    scheduler,
    config,
    global_step,
    output_dir,
    device,
    num_val_steps=20,
    max_val_samples=None,
):
    """
    在验证集上运行完整去噪推理，保存 source / generated / target 对比图。

    Args:
        val_dataset: EditPrecomputedDataset（验证子集）
        vae: VAE 模型（懒加载）
        scheduler: FlowMatchEulerDiscreteScheduler
        num_val_steps: 验证推理步数（默认 20，快速预览）
        max_val_samples: 每次最多推理多少张（None=全部）
    """
    import inspect
    from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT

    transformer.eval()
    semantic_processor.eval()

    val_dir = Path(output_dir) / "val_samples" / f"step_{global_step:06d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(val_dataset) if max_val_samples is None else min(max_val_samples, len(val_dataset))

    # 设置 scheduler 的 mu 参数
    height_latent = config.resolution // 8
    width_latent = config.resolution // 8
    image_seq_len = (height_latent // 2) * (width_latent // 2)
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
    mu = image_seq_len * m + b

    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(num_val_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps

    total_mse = 0.0
    logger.info(f"[Validate] Running full inference on {n_samples} samples ({num_val_steps} steps)...")

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc=f"Val step {global_step}"):
            sample = val_dataset[idx]
            source_latent = sample["source_latent"].to(device, dtype=torch.bfloat16)   # [16, H, W]
            target_latent = sample["target_latent"].to(device, dtype=torch.bfloat16)   # [16, H, W]
            semantic_feat = sample["semantic_feature"].to(device, dtype=torch.bfloat16)  # [N, dim]
            text_emb = sample["text_embedding"].to(device, dtype=torch.bfloat16)        # [seq, 2560]
            prompt_text = sample["prompt"]

            # Semantic Processor
            sem_embed = semantic_processor(semantic_feat.unsqueeze(0)).squeeze(0)  # [N, 2560]
            cap_feats = torch.cat([text_emb, sem_embed], dim=0)  # [seq+N, 2560]

            # 去噪
            scheduler._step_index = None
            source_5d = source_latent.unsqueeze(1)  # [16, 1, H, W]
            latents = torch.randn(1, 16, 1, height_latent, width_latent, device=device, dtype=torch.float32)

            for i, t in enumerate(timesteps):

                combined = torch.cat([latents, source_5d.unsqueeze(0).to(latents.dtype)], dim=2)  # [1,16,2,H,W]
                timestep_model = (1000 - t.expand(1)) / 1000

                x_list = [combined[0].to(torch.bfloat16)]  # [16, 2, H, W]
                model_out_list = transformer(x_list, timestep_model, [cap_feats])[0]

                noise_pred = model_out_list[0][:, :1, :, :]  # [16, 1, H, W] -> frame 0
                noise_pred = -noise_pred.float().unsqueeze(0).squeeze(2)  # [1, 16, H, W]
                latents_2d = latents.squeeze(2)
                latents_2d = scheduler.step(noise_pred.float(), t, latents_2d, return_dict=False)[0]
                latents = latents_2d.unsqueeze(2)

            # VAE decode generated
            shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
            decode_latents = (latents.squeeze(2).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
            decoded_gen = vae.decode(decode_latents, return_dict=False)[0]
            decoded_gen = (decoded_gen / 2 + 0.5).clamp(0, 1)

            # VAE decode target (ground truth)
            target_for_decode = (target_latent.unsqueeze(0).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
            decoded_tgt = vae.decode(target_for_decode, return_dict=False)[0]
            decoded_tgt = (decoded_tgt / 2 + 0.5).clamp(0, 1)

            # VAE decode source
            source_for_decode = (source_latent.unsqueeze(0).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
            decoded_src = vae.decode(source_for_decode, return_dict=False)[0]
            decoded_src = (decoded_src / 2 + 0.5).clamp(0, 1)

            # 计算 latent 空间 MSE
            gen_latent = (latents.squeeze(2).to(torch.float32) - config.vae_shift_factor) * config.vae_scaling_factor
            # Actually just compare generated latent with target latent directly
            total_mse += F.mse_loss(latents.squeeze(2).squeeze(0).float(),
                                     target_latent.float()).item()

            # 保存对比图: source | generated | target
            def to_pil(tensor):
                img = tensor[0].cpu().permute(1, 2, 0).float().numpy()
                img = (img * 255).round().clip(0, 255).astype("uint8")
                return Image.fromarray(img)

            src_img = to_pil(decoded_src)
            gen_img = to_pil(decoded_gen)
            tgt_img = to_pil(decoded_tgt)

            # 拼接三张图
            w, h = src_img.size
            comparison = Image.new("RGB", (w * 3, h))
            comparison.paste(src_img, (0, 0))
            comparison.paste(gen_img, (w, 0))
            comparison.paste(tgt_img, (w * 2, 0))
            comparison.save(val_dir / f"{idx:03d}.jpg", quality=90)

            # 保存 prompt 文本
            if idx == 0:
                prompt_file = open(val_dir / "prompts.txt", "w")
            else:
                prompt_file = open(val_dir / "prompts.txt", "a")
            prompt_file.write(f"{idx:03d}: {prompt_text}\n")
            prompt_file.close()

    avg_mse = total_mse / n_samples
    logger.info(f"[Validate] Done. Avg latent MSE: {avg_mse:.6f}. Images saved to {val_dir}")

    transformer.train()
    semantic_processor.train()
    return avg_mse


# ============================================================================
# 训练阶段
# ============================================================================


def train(config: EditTrainConfig):
    """
    图像编辑 LoRA 训练主循环。

    训练流程：
    1. 加载 Transformer 模型 + 注入 LoRA
    2. 初始化 Semantic Processor（全量训练）
    3. Flow Matching 训练：
       - 对 target latent 加噪
       - 将 noised target 和 source latent 沿 T 维拼接
       - text embedding 和 semantic embedding 拼接作为 cap_feats
       - 模型预测 target = x_0 - noise
       - 取 frame 0 输出计算 MSE loss
    """
    from peft import LoraConfig, get_peft_model
    from torch.utils.tensorboard import SummaryWriter
    from utils import load_from_local_dir
    from utils.loader import load_sharded_safetensors
    from zimage.pipeline import generate as zimage_generate
    from zimage.transformer import ZImageTransformer2DModel

    import bitsandbytes as bnb
    import numpy as np

    # --- DDP 初始化 ---
    distributed = dist.is_initialized()
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda")

    is_main = (rank == 0)
    torch.manual_seed(config.seed + rank)  # 不同 rank 用不同种子增加多样性

    # ========== 1. 加载 Transformer 模型 ==========
    if is_main:
        logger.info("Loading transformer...")
    model_dir = Path(config.model_path)
    transformer_dir = model_dir / "transformer"

    with open(transformer_dir / "config.json") as f:
        transformer_config = json.load(f)

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

    state_dict = load_sharded_safetensors(transformer_dir, device="cpu")
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    transformer.eval()

    # ========== 2. 注入 LoRA ==========
    if is_main:
        logger.info("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    if is_main:
        transformer.print_trainable_parameters()

    transformer.get_input_embeddings = lambda: None
    for param in transformer.parameters():
        if not param.requires_grad:
            param.requires_grad_(False)

    # 启用 gradient checkpointing 减少显存（在每个 transformer layer 上）
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    base_model = transformer.get_base_model()
    if hasattr(base_model, 'layers'):
        for layer in base_model.layers:
            layer._original_forward = layer.forward
            def make_ckpt_forward(mod):
                def ckpt_forward(*args, **kwargs):
                    return torch_checkpoint(mod._original_forward, *args, use_reentrant=False, **kwargs)
                return ckpt_forward
            layer.forward = make_ckpt_forward(layer)
        if is_main:
            logger.info(f"Enabled gradient checkpointing on {len(base_model.layers)} layers")

    transformer.train()

    # ========== 3. 初始化 Semantic Processor ==========
    if is_main:
        logger.info("Initializing Semantic Processor...")
    semantic_processor = SemanticProcessor(
        siglip_dim=config.siglip_dim,
        output_dim=2560,
    ).to(device=device, dtype=torch.bfloat16)
    semantic_processor.train()
    if is_main:
        logger.info(f"  Semantic Processor params: {sum(p.numel() for p in semantic_processor.parameters()):,}")

    # ========== 4. 加载数据集（拆分 train/val） ==========
    full_dataset = EditPrecomputedDataset(config.cache_dir)

    # 随机抽取 100 个样本作为验证集（固定种子保证可复现）
    n_val = min(100, len(full_dataset) // 10)
    rng = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(len(full_dataset), generator=rng).tolist()
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    if is_main:
        logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val")

    if distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=edit_collate_fn,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=edit_collate_fn,
            num_workers=0,
            drop_last=True,
        )

    # ========== 5. 验证组件（延迟加载，节省显存） ==========
    # VAE 和 scheduler 在首次验证时加载到 rank 0 设备
    val_vae = None
    val_scheduler = None

    # ========== 6. 配置优化器 ==========
    # 收集所有可训练参数：LoRA 参数 + Semantic Processor 参数
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    semantic_params = list(semantic_processor.parameters())
    all_trainable_params = lora_params + semantic_params

    optimizer = bnb.optim.AdamW8bit(all_trainable_params, lr=config.learning_rate, weight_decay=0.01)

    # DDP: 每个 epoch 每张卡看 len(dataset)/world_size 个样本
    steps_per_epoch = math.ceil(len(train_dataset) / (config.batch_size * world_size))
    total_steps = config.epochs * steps_per_epoch
    effective_steps = total_steps // config.gradient_accumulation_steps

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, effective_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========== 7. 训练循环 ==========
    output_dir = Path(config.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    running_loss = 0.0

    writer = None
    if is_main:
        tb_log_dir = output_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard logs: {tb_log_dir}")
        logger.info(f"Starting training: {config.epochs} epochs, {total_steps} total steps")
        logger.info(f"  World size: {world_size}, Effective batch size: {config.batch_size * config.gradient_accumulation_steps * world_size}")
        logger.info(f"  LoRA rank: {config.lora_rank}, Semantic Processor: {config.siglip_dim}→2560")
        logger.info(f"  Learning rate: {config.learning_rate}")

    if distributed:
        dist.barrier()

    for epoch in range(config.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            # --- 准备数据 ---
            target_latents = batch["target_latents"].to(device, dtype=torch.bfloat16)  # [B, 16, H, W]
            source_latents = batch["source_latents"].to(device, dtype=torch.bfloat16)  # [B, 16, H, W]
            text_embeddings = [e.to(device, dtype=torch.bfloat16) for e in batch["text_embeddings"]]
            semantic_features = [sf.to(device, dtype=torch.bfloat16) for sf in batch["semantic_features"]]
            B = target_latents.shape[0]

            # --- Semantic Processor: SigLip-2 特征 → 2560 维 ---
            # Stack semantic features for batch processing
            semantic_stacked = torch.stack(semantic_features)  # [B, N_tokens, siglip_dim]
            semantic_embeds = semantic_processor(semantic_stacked)  # [B, N_tokens, 2560]

            # --- 拼接 text + semantic 作为 cap_feats ---
            # 每个样本的 cap_feats = [text_embedding; semantic_embedding] 沿 seq_len 维度
            cap_feats_list = []
            for i in range(B):
                cap = torch.cat([text_embeddings[i], semantic_embeds[i]], dim=0)  # [seq+N_tokens, 2560]
                cap_feats_list.append(cap)

            # --- CFG Dropout ---
            if config.cfg_dropout_prob > 0:
                for i in range(B):
                    if torch.rand(1).item() < config.cfg_dropout_prob:
                        # 丢弃文本和语义条件，用零向量替换
                        cap_feats_list[i] = torch.zeros(
                            1, cap_feats_list[i].shape[-1],
                            device=device, dtype=torch.bfloat16
                        )

            # --- 采样时间步 ---
            sigma = torch.sigmoid(torch.randn(B, device=device, dtype=torch.bfloat16))

            # --- 构造加噪 target ---
            noise = torch.randn_like(target_latents)
            sigma_expand = sigma[:, None, None, None]
            noisy_target = (1 - sigma_expand) * target_latents + sigma_expand * noise

            # --- 训练目标 ---
            target = target_latents - noise

            # --- 时间步转换 ---
            model_timestep = 1 - sigma

            # --- 拼接 source + noised_target 在 T 维度 ---
            # noisy_target: [B, 16, H, W] -> [B, 16, 1, H, W]
            # source: [B, 16, H, W] -> [B, 16, 1, H, W]
            # combined: [B, 16, 2, H, W] (frame 0 = noised target, frame 1 = source)
            noisy_target_5d = noisy_target.unsqueeze(2)
            source_5d = source_latents.unsqueeze(2)
            x_combined = torch.cat([noisy_target_5d, source_5d], dim=2)  # [B, 16, 2, H, W]

            # --- 模型前向传播 ---
            x_list = [x_combined[i] for i in range(B)]  # List of [16, 2, H, W]
            pred_list, _ = transformer(x_list, model_timestep, cap_feats_list)

            # --- 取 frame 0（target 部分）计算 loss ---
            # pred_list 每个元素为 [16, 2, H, W]，取 [:, 0, :, :] 即 frame 0
            pred = torch.stack([p[:, 0, :, :] for p in pred_list])  # [B, 16, H, W]

            # --- 计算 Loss ---
            loss = F.mse_loss(pred.float(), target.float())
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()

            # --- 梯度累积 & 参数更新 ---
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # DDP: 同步所有 rank 的梯度（flatten 后一次 all-reduce，高效）
                if distributed:
                    # 收集所有有梯度的参数到 flat buffer
                    grads = []
                    for param in all_trainable_params:
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        grads.append(param.grad.flatten())
                    flat_grads = torch.cat(grads)
                    dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG)
                    # 写回
                    offset = 0
                    for param in all_trainable_params:
                        numel = param.grad.numel()
                        param.grad.copy_(flat_grads[offset:offset + numel].view_as(param.grad))
                        offset += numel

                torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- 日志记录（仅 rank 0） ---
                if is_main and global_step % config.log_every_steps == 0:
                    avg_loss = running_loss / config.log_every_steps
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(f"Step {global_step} | Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running_loss = 0.0

                # --- 验证（仅 rank 0）：完整推理 100 个验证样本 ---
                if is_main and (global_step == 1 or global_step % config.validate_every_steps == 0):
                    # 懒加载 VAE 和 scheduler（首次验证时）
                    if val_vae is None:
                        from diffusers import AutoencoderKL
                        from zimage.scheduler import FlowMatchEulerDiscreteScheduler
                        logger.info("[Validate] Loading VAE and scheduler for validation...")
                        model_dir = Path(config.model_path)
                        val_vae = AutoencoderKL.from_pretrained(
                            str(model_dir / "vae"), torch_dtype=torch.bfloat16
                        ).to(device)
                        val_vae.eval()
                        val_scheduler = FlowMatchEulerDiscreteScheduler()

                    val_mse = validate_full(
                        transformer=transformer,
                        semantic_processor=semantic_processor,
                        val_dataset=val_dataset,
                        vae=val_vae,
                        scheduler=val_scheduler,
                        config=config,
                        global_step=global_step,
                        output_dir=output_dir,
                        device=device,
                        num_val_steps=10,
                        max_val_samples=config.val_samples_per_run,
                    )
                    writer.add_scalar("validation/latent_mse", val_mse, global_step)

                # --- 保存 checkpoint（仅 rank 0） ---
                if is_main and global_step % config.save_every_steps == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    transformer.save_pretrained(save_path / "lora")
                    torch.save(semantic_processor.state_dict(), save_path / "semantic_processor.pt")
                    logger.info(f"Saved checkpoint to {save_path}")

    # ========== 8. 保存最终权重 ==========
    if is_main:
        final_path = output_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(final_path / "lora")
        torch.save(semantic_processor.state_dict(), final_path / "semantic_processor.pt")
        if writer:
            writer.close()
        logger.info(f"Training complete! Weights saved to {final_path}")
        logger.info(f"TensorBoard logs: tensorboard --logdir {output_dir / 'tensorboard'}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# ============================================================================
# 推理阶段
# ============================================================================


def inference(config: EditTrainConfig, weights_path: str, source_path: str, prompt: str):
    """
    使用训练好的 Edit LoRA 进行图像编辑推理。

    流程：
    1. 加载 SigLip-2 (frozen) + Semantic Processor (trained) + Transformer (base + LoRA)
    2. 源图 → SigLip-2 → Semantic Processor → semantic_tokens
    3. 源图 → VAE encode → source_latents
    4. 编辑指令 → Qwen3 → text_tokens
    5. 纯噪声 → noise_latents
    6. 去噪循环，每步拼接 [noise, source] 在 T 维，[text, semantic] 在 seq 维
    7. VAE decode → 编辑后图片
    """
    import inspect

    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor
    from utils import load_from_local_dir

    device = torch.device("cuda")
    weights_path = Path(weights_path)

    # 加载基础模型
    logger.info("Loading base model...")
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    transformer = components["transformer"]
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    scheduler = components["scheduler"]

    # 加载 LoRA
    logger.info(f"Loading LoRA from {weights_path / 'lora'}...")
    transformer = PeftModel.from_pretrained(transformer, str(weights_path / "lora"))
    transformer = transformer.merge_and_unload()
    transformer.eval()

    # 加载 Semantic Processor
    logger.info("Loading Semantic Processor...")
    semantic_processor = SemanticProcessor(
        siglip_dim=config.siglip_dim,
        output_dim=2560,
    ).to(device=device, dtype=torch.bfloat16)
    sp_state = torch.load(weights_path / "semantic_processor.pt", map_location=device, weights_only=True)
    semantic_processor.load_state_dict(sp_state)
    semantic_processor.eval()

    # 加载 SigLip-2
    logger.info(f"Loading SigLip-2: {config.siglip_model_name}...")
    siglip_processor = AutoProcessor.from_pretrained(config.siglip_model_name)
    siglip_model = AutoModel.from_pretrained(config.siglip_model_name).to(device)
    siglip_model.eval()

    # 加载源图
    source_img = Image.open(source_path).convert("RGB")
    logger.info(f"Source image: {source_path} ({source_img.width}x{source_img.height})")

    # --- 1. 源图 → SigLip-2 → Semantic Processor → semantic tokens ---
    with torch.no_grad():
        semantic_feat = extract_siglip_features(siglip_model, siglip_processor, [source_img], device)
        semantic_feat = semantic_feat.to(device=device, dtype=torch.bfloat16)
        semantic_embeds = semantic_processor(semantic_feat).squeeze(0)  # [N_tokens, 2560]

    # --- 2. 源图 → VAE encode → source_latents ---
    transform = transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    source_tensor = transform(source_img).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        source_latents = vae_encode(vae, source_tensor, config.vae_scaling_factor, config.vae_shift_factor)
    # [1, 16, H/8, W/8] -> [1, 16, 1, H/8, W/8]
    source_latents = source_latents.unsqueeze(2).to(torch.bfloat16)

    # --- 3. 编辑指令 → Qwen3 → text tokens ---
    with torch.no_grad():
        text_emb = encode_text(tokenizer, text_encoder, [prompt], config.max_sequence_length, device)
        text_emb = text_emb[0].to(device=device, dtype=torch.bfloat16)  # [seq_len, 2560]

    # --- 4. 拼接 cap_feats ---
    cap_feats = torch.cat([text_emb, semantic_embeds], dim=0)  # [seq+N_tokens, 2560]

    # --- 5. Prepare for negative prompt (CFG) ---
    neg_text_emb = encode_text(tokenizer, text_encoder, [""], config.max_sequence_length, device)
    neg_text_emb = neg_text_emb[0].to(device=device, dtype=torch.bfloat16)
    # 负条件不含 semantic（或用零向量）
    neg_semantic = torch.zeros_like(semantic_embeds)
    neg_cap_feats = torch.cat([neg_text_emb, neg_semantic], dim=0)

    # --- 6. 去噪循环 ---
    height_latent = config.resolution // 8
    width_latent = config.resolution // 8
    noise_shape = (1, 16, 1, height_latent, width_latent)
    latents = torch.randn(noise_shape, device=device, dtype=torch.float32)

    # 设置 scheduler
    from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT

    image_seq_len = (height_latent // 2) * (width_latent // 2)
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
    mu = image_seq_len * m + b

    scheduler.sigma_min = 0.0
    sigmas_kwargs = {"mu": mu}

    # retrieve_timesteps
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    scheduler.set_timesteps(config.inference_steps, device=device, **sigmas_kwargs)
    timesteps = scheduler.timesteps

    logger.info(f"Denoising: {config.inference_steps} steps, cfg={config.guidance_scale}")

    for i, t in enumerate(tqdm(timesteps, desc="Edit denoising")):
        if t == 0 and i == len(timesteps) - 1:
            continue

        # 拼接 noise + source 在 T 维度
        combined = torch.cat([latents, source_latents.to(latents.dtype)], dim=2)  # [1, 16, 2, H, W]

        timestep = t.expand(1)
        timestep_model = (1000 - timestep) / 1000

        # CFG: 正条件 + 负条件
        if config.guidance_scale > 1.0:
            combined_typed = combined.to(torch.bfloat16)
            latent_input = combined_typed.repeat(2, 1, 1, 1, 1)  # [2, 16, 2, H, W]
            cap_input = [cap_feats, neg_cap_feats]
            t_input = timestep_model.repeat(2)
        else:
            latent_input = combined.to(torch.bfloat16)
            cap_input = [cap_feats]
            t_input = timestep_model

        x_list = list(latent_input.unbind(dim=0))
        model_out_list = transformer(x_list, t_input, cap_input)[0]

        # 取 frame 0（target 部分）
        model_out_list = [out[:, :1, :, :] for out in model_out_list]

        if config.guidance_scale > 1.0:
            pos_out = model_out_list[0].float()  # [16, 1, H, W]
            neg_out = model_out_list[1].float()  # [16, 1, H, W]
            noise_pred = pos_out + config.guidance_scale * (pos_out - neg_out)
        else:
            noise_pred = model_out_list[0].float()  # [16, 1, H, W]

        # noise_pred: [C, 1, H, W] -> add batch dim -> [1, C, 1, H, W] -> squeeze T -> [1, C, H, W]
        noise_pred = -noise_pred.unsqueeze(0).squeeze(2)  # [1, 16, H, W]
        latents_2d = latents.squeeze(2)  # [1, 16, H, W]
        latents_2d = scheduler.step(noise_pred.to(torch.float32), t, latents_2d, return_dict=False)[0]
        latents = latents_2d.unsqueeze(2)

    # --- 7. VAE decode ---
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    decode_latents = (latents.squeeze(2).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
    with torch.no_grad():
        decoded = vae.decode(decode_latents, return_dict=False)[0]

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
    decoded = (decoded * 255).round().astype("uint8")
    result_img = Image.fromarray(decoded[0])

    # 保存结果
    output_path = Path(config.output_dir) / "edit_result.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    logger.info(f"Saved edit result to {output_path}")

    return result_img


# ============================================================================
# 入口
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Edit LoRA Training")
    # 运行模式
    parser.add_argument("--precompute", action="store_true", help="预计算 VAE latents、text embeddings 和 SigLip-2 特征")
    parser.add_argument("--merge_shards", type=int, default=None, help="合并 N 个 shard 的预计算结果")
    parser.add_argument("--train", action="store_true", help="运行 Edit LoRA 训练")
    parser.add_argument("--inference", action="store_true", help="使用训练好的权重进行图像编辑推理")

    # 路径参数
    parser.add_argument("--data_dir", type=str, default=None, help="编辑数据集目录")
    parser.add_argument("--weights_path", type=str, default=None, help="推理时的权重路径")
    parser.add_argument("--source", type=str, default=None, help="推理时的源图路径")
    parser.add_argument("--prompt", type=str, default=None, help="推理时的编辑指令")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--prompt_level", type=str, default=None, choices=["short", "medium", "long"],
                        help="TSV 数据集 prompt 级别")
    parser.add_argument("--max_samples", type=int, default=None, help="预计算最大样本数")
    parser.add_argument("--shard", type=str, default=None,
                        help="预计算分片，格式 'i/N'（如 '0/8' 表示 8 卡中的第 0 卡）")

    # 超参数覆盖
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument("--rank", type=int, default=None, help="覆盖 LoRA rank")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="覆盖梯度累积步数")
    parser.add_argument("--val_every", type=int, default=None, help="覆盖验证间隔步数")
    parser.add_argument("--val_samples", type=int, default=None, help="每次验证的样本数（默认全部100）")
    parser.add_argument("--resolution", type=int, default=None, help="覆盖分辨率")
    parser.add_argument("--steps", type=int, default=None, help="覆盖推理步数")
    parser.add_argument("--cfg", type=float, default=None, help="覆盖 guidance scale")

    args = parser.parse_args()
    config = EditTrainConfig()

    # 应用命令行覆盖
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.prompt_level:
        config.prompt_level = args.prompt_level
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.rank:
        config.lora_rank = args.rank
        config.lora_alpha = args.rank
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.grad_accum:
        config.gradient_accumulation_steps = args.grad_accum
    if args.val_every:
        config.validate_every_steps = args.val_every
    if args.val_samples:
        config.val_samples_per_run = args.val_samples
    if args.resolution:
        config.resolution = args.resolution
    if args.steps:
        config.inference_steps = args.steps
    if args.cfg:
        config.guidance_scale = args.cfg

    # 执行对应模式
    if args.precompute:
        shard_id, num_shards = None, None
        if args.shard:
            parts = args.shard.split("/")
            shard_id, num_shards = int(parts[0]), int(parts[1])
        precompute(config, shard_id=shard_id, num_shards=num_shards)
    elif args.merge_shards:
        merge_shards(config, args.merge_shards)
    elif args.train:
        # DDP 初始化：如果通过 torchrun 启动，环境变量会自动设置
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            if int(os.environ.get("RANK", 0)) == 0:
                logger.info(f"DDP initialized: world_size={dist.get_world_size()}")
        train(config)
    elif args.inference:
        if args.source is None:
            parser.error("--inference requires --source (source image path)")
        if args.prompt is None:
            parser.error("--inference requires --prompt (edit instruction)")
        weights_path = args.weights_path or str(Path(config.output_dir) / "final")
        inference(config, weights_path, args.source, args.prompt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
