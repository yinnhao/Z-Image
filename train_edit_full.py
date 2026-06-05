"""
Z-Image-Edit 全参数微调训练脚本 (FSDP)

与 train_edit_lora.py 的区别：
- 全参数训练 ~6B transformer（非 LoRA）
- 使用 FSDP (ZeRO-3) 做模型/梯度/优化器分片
- 8×A100-40GB 分布式训练
- 学习率 5e-6，30 epochs

Usage:
    # 预计算（复用 train_edit_lora.py 的预计算结果）
    # 训练（8卡）
    torchrun --nproc_per_node=8 train_edit_full.py --train
    # 推理
    python train_edit_full.py --inference --source input.png --prompt "把 logo 改成绿色"
"""

import argparse
import json
import logging
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================


class FullParamConfig:
    model_path = "ckpts/Z-Image"
    cache_dir = "output/edit_precomputed"
    output_dir = "output/edit_full_param"

    resolution = 512
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 5e-6
    weight_decay = 0.01
    epochs = 30
    warmup_steps = 500
    max_grad_norm = 1.0
    save_every_steps = 200
    log_every_steps = 10
    validate_every_steps = 100
    val_samples_per_run = 10
    seed = 42

    siglip_dim = 1024
    cfg_dropout_prob = 0.1
    vae_scaling_factor = 0.3611
    vae_shift_factor = 0.1159
    max_sequence_length = 512
    inference_steps = 50
    guidance_scale = 5.0

    siglip_model_name = "google/siglip2-large-patch16-384"


# ============================================================================
# SemanticProcessor（同 train_edit_lora.py）
# ============================================================================


class SemanticProcessor(nn.Module):
    def __init__(self, siglip_dim=1024, output_dim=2560):
        super().__init__()
        self.norm = nn.LayerNorm(siglip_dim)
        self.proj = nn.Linear(siglip_dim, output_dim)

    def forward(self, x):
        return self.proj(self.norm(x))


# ============================================================================
# Dataset / Collate（同 train_edit_lora.py）
# ============================================================================


class EditPrecomputedDataset(Dataset):
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
    return {
        "source_latents": torch.stack([item["source_latent"] for item in batch]),
        "target_latents": torch.stack([item["target_latent"] for item in batch]),
        "semantic_features": [item["semantic_feature"] for item in batch],
        "text_embeddings": [item["text_embedding"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
    }


# ============================================================================
# 验证
# ============================================================================


def validate_full(
    transformer, semantic_processor, val_dataset, vae, scheduler, config,
    global_step, output_dir, device, num_val_steps=20, max_val_samples=None,
):
    """在验证集上运行完整去噪推理（仅 rank 0 调用）。"""
    from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT

    transformer.eval()
    semantic_processor.eval()

    val_dir = Path(output_dir) / "val_samples" / f"step_{global_step:06d}"
    val_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(val_dataset) if max_val_samples is None else min(max_val_samples, len(val_dataset))

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
    logger.info(f"[Validate] Running inference on {n_samples} samples ({num_val_steps} steps)...")

    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc=f"Val step {global_step}"):
            sample = val_dataset[idx]
            source_latent = sample["source_latent"].to(device, dtype=torch.bfloat16)
            target_latent = sample["target_latent"].to(device, dtype=torch.bfloat16)
            semantic_feat = sample["semantic_feature"].to(device, dtype=torch.bfloat16)
            text_emb = sample["text_embedding"].to(device, dtype=torch.bfloat16)

            sem_embed = semantic_processor(semantic_feat.unsqueeze(0)).squeeze(0)
            cap_feats = torch.cat([text_emb, sem_embed], dim=0)

            scheduler._step_index = None
            source_5d = source_latent.unsqueeze(1)
            latents = torch.randn(1, 16, 1, height_latent, width_latent, device=device, dtype=torch.float32)

            for i, t in enumerate(timesteps):
                combined = torch.cat([latents, source_5d.unsqueeze(0).to(latents.dtype)], dim=2)
                timestep_model = (1000 - t.expand(1)) / 1000
                x_list = [combined[0].to(torch.bfloat16)]
                model_out_list = transformer(x_list, timestep_model, [cap_feats])[0]
                noise_pred = model_out_list[0][:, :1, :, :]
                noise_pred = -noise_pred.float().unsqueeze(0).squeeze(2)
                latents_2d = latents.squeeze(2)
                latents_2d = scheduler.step(noise_pred.float(), t, latents_2d, return_dict=False)[0]
                latents = latents_2d.unsqueeze(2)

            # Decode and save comparison
            shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
            decode_latents = (latents.squeeze(2).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
            decoded_gen = (vae.decode(decode_latents, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)

            target_for_decode = (target_latent.unsqueeze(0).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
            decoded_tgt = (vae.decode(target_for_decode, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)

            source_for_decode = (source_latent.unsqueeze(0).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
            decoded_src = (vae.decode(source_for_decode, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)

            total_mse += F.mse_loss(latents.squeeze(2).squeeze(0).float(), target_latent.float()).item()

            def to_pil(tensor):
                img = tensor[0].cpu().permute(1, 2, 0).float().numpy()
                return Image.fromarray((img * 255).round().clip(0, 255).astype("uint8"))

            src_img, gen_img, tgt_img = to_pil(decoded_src), to_pil(decoded_gen), to_pil(decoded_tgt)
            w, h = src_img.size
            comparison = Image.new("RGB", (w * 3, h))
            comparison.paste(src_img, (0, 0))
            comparison.paste(gen_img, (w, 0))
            comparison.paste(tgt_img, (w * 2, 0))
            comparison.save(val_dir / f"{idx:03d}.jpg", quality=90)

            with open(val_dir / "prompts.txt", "a" if idx > 0 else "w") as f:
                f.write(f"{idx:03d}: {sample['prompt']}\n")

    avg_mse = total_mse / n_samples
    logger.info(f"[Validate] Avg latent MSE: {avg_mse:.6f}. Images saved to {val_dir}")
    transformer.train()
    semantic_processor.train()
    return avg_mse


# ============================================================================
# 训练
# ============================================================================


def train(config: FullParamConfig):
    """FSDP 全参数微调训练主函数。"""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    from torch.utils.tensorboard import SummaryWriter
    from utils.loader import load_sharded_safetensors
    from zimage.transformer import ZImageTransformer2DModel, ZImageTransformerBlock

    # --- 分布式初始化 ---
    import datetime
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0
    torch.manual_seed(config.seed + rank)

    # ========== 1. 加载 Transformer ==========
    if is_main:
        logger.info("Loading transformer on meta device...")
    model_dir = Path(config.model_path)
    transformer_dir = model_dir / "transformer"
    with open(transformer_dir / "config.json") as f:
        tc = json.load(f)

    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            in_channels=tc.get("in_channels", 16),
            dim=tc.get("dim", 3840),
            n_layers=tc.get("n_layers", 30),
            n_refiner_layers=tc.get("n_refiner_layers", 2),
            n_heads=tc.get("n_heads", 30),
            n_kv_heads=tc.get("n_kv_heads", 30),
            cap_feat_dim=tc.get("cap_feat_dim", 2560),
            all_patch_size=tuple(tc.get("all_patch_size", [2])),
            all_f_patch_size=tuple(tc.get("all_f_patch_size", [1])),
            norm_eps=tc.get("norm_eps", 1e-5),
            qk_norm=tc.get("qk_norm", True),
            rope_theta=tc.get("rope_theta", 256.0),
            t_scale=tc.get("t_scale", 1000.0),
            axes_dims=tc.get("axes_dims", [32, 48, 48]),
            axes_lens=tc.get("axes_lens", [1536, 512, 512]),
        )

    # Rank 0 加载权重，其他 rank 也从磁盘加载（避免 broadcast 超时）
    if is_main:
        logger.info("Loading transformer weights...")
    state_dict = load_sharded_safetensors(transformer_dir, device="cpu", dtype=torch.bfloat16)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    if is_main:
        logger.info("Transformer weights loaded on all ranks")

    # 同步：确保所有 rank 都加载完毕再进入 FSDP
    dist.barrier()

    # ========== 2. FSDP 包裹 Transformer ==========
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ZImageTransformerBlock},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    transformer = FSDP(
        transformer,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,
        sync_module_states=False,
        limit_all_gathers=True,
    )

    # Activation checkpointing
    apply_activation_checkpointing(
        transformer,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda m: isinstance(m, ZImageTransformerBlock),
    )
    if is_main:
        total_params = sum(p.numel() for p in transformer.parameters())
        logger.info(f"Transformer params: {total_params:,} (FSDP sharded across {world_size} GPUs)")

    transformer.train()

    # ========== 3. SemanticProcessor (DDP) ==========
    semantic_processor = SemanticProcessor(siglip_dim=config.siglip_dim, output_dim=2560).to(device=device, dtype=torch.bfloat16)
    semantic_processor = DDP(semantic_processor, device_ids=[local_rank])
    semantic_processor.train()
    if is_main:
        sp_params = sum(p.numel() for p in semantic_processor.parameters())
        logger.info(f"SemanticProcessor params: {sp_params:,}")

    # ========== 4. Dataset ==========
    full_dataset = EditPrecomputedDataset(config.cache_dir)
    n_val = min(100, len(full_dataset) // 10)
    rng = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(len(full_dataset), generator=rng).tolist()
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    if is_main:
        logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=sampler,
        collate_fn=edit_collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    # ========== 5. Optimizer + Scheduler ==========
    all_params = list(transformer.parameters()) + list(semantic_processor.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    steps_per_epoch = math.ceil(len(train_dataset) / (config.batch_size * world_size))
    total_steps = config.epochs * steps_per_epoch
    effective_steps = total_steps // config.gradient_accumulation_steps

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, effective_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========== 6. Resume ==========
    start_epoch = 0
    global_step = 0
    resume_path = Path(config.output_dir) / "latest_checkpoint"
    if resume_path.exists():
        ckpt = torch.load(resume_path / "training_state.pt", map_location="cpu", weights_only=True)
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        # Load semantic processor
        sp_state = torch.load(resume_path / "semantic_processor.pt", map_location=device, weights_only=True)
        semantic_processor.module.load_state_dict(sp_state)
        if is_main:
            logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")

    # ========== 7. Training Loop ==========
    output_dir = Path(config.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    running_loss = 0.0
    writer = None
    if is_main:
        tb_dir = output_dir / "runs"
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"Training: {config.epochs} epochs, effective batch={config.batch_size * config.gradient_accumulation_steps * world_size}")

    val_vae = None
    val_scheduler = None
    dist.barrier()

    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            target_latents = batch["target_latents"].to(device, dtype=torch.bfloat16)
            source_latents = batch["source_latents"].to(device, dtype=torch.bfloat16)
            text_embeddings = [e.to(device, dtype=torch.bfloat16) for e in batch["text_embeddings"]]
            semantic_features = [sf.to(device, dtype=torch.bfloat16) for sf in batch["semantic_features"]]
            B = target_latents.shape[0]

            # Semantic processing
            semantic_stacked = torch.stack(semantic_features)
            semantic_embeds = semantic_processor(semantic_stacked)

            # Cap feats = text + semantic
            cap_feats_list = []
            for i in range(B):
                cap_feats_list.append(torch.cat([text_embeddings[i], semantic_embeds[i]], dim=0))

            # CFG dropout
            if config.cfg_dropout_prob > 0:
                for i in range(B):
                    if torch.rand(1).item() < config.cfg_dropout_prob:
                        cap_feats_list[i] = torch.zeros(1, cap_feats_list[i].shape[-1], device=device, dtype=torch.bfloat16)

            # Flow matching: sample timestep, add noise
            sigma = torch.sigmoid(torch.randn(B, device=device, dtype=torch.bfloat16))
            noise = torch.randn_like(target_latents)
            sigma_expand = sigma[:, None, None, None]
            noisy_target = (1 - sigma_expand) * target_latents + sigma_expand * noise
            target = target_latents - noise
            model_timestep = 1 - sigma

            # Concat source + noised target in T dim
            x_combined = torch.cat([noisy_target.unsqueeze(2), source_latents.unsqueeze(2)], dim=2)
            x_list = [x_combined[i] for i in range(B)]

            # Forward + loss (no_sync avoids redundant all-reduce during accumulation)
            is_accumulating = (step + 1) % config.gradient_accumulation_steps != 0
            if is_accumulating:
                with transformer.no_sync():
                    pred_list, _ = transformer(x_list, model_timestep, cap_feats_list)
                    pred = torch.stack([p[:, 0, :, :] for p in pred_list])
                    loss = F.mse_loss(pred.float(), target.float()) / config.gradient_accumulation_steps
                    loss.backward()
            else:
                pred_list, _ = transformer(x_list, model_timestep, cap_feats_list)
                pred = torch.stack([p[:, 0, :, :] for p in pred_list])
                loss = F.mse_loss(pred.float(), target.float()) / config.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item()

            # Optimizer step on accumulation boundary
            if not is_accumulating:
                torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if is_main and global_step % config.log_every_steps == 0:
                    avg_loss = running_loss / config.log_every_steps
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(f"Step {global_step} | Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running_loss = 0.0

                # Validation (rank 0 only, others barrier wait)
                if global_step == 1 or global_step % config.validate_every_steps == 0:
                    if is_main:
                        if val_vae is None:
                            from diffusers import AutoencoderKL
                            from zimage.scheduler import FlowMatchEulerDiscreteScheduler
                            logger.info("[Validate] Loading VAE...")
                            val_vae = AutoencoderKL.from_pretrained(
                                str(model_dir / "vae"), torch_dtype=torch.bfloat16
                            ).to(device)
                            val_vae.eval()
                            val_scheduler = FlowMatchEulerDiscreteScheduler()
                        val_mse = validate_full(
                            transformer=transformer,
                            semantic_processor=semantic_processor.module,
                            val_dataset=val_dataset,
                            vae=val_vae, scheduler=val_scheduler, config=config,
                            global_step=global_step, output_dir=output_dir,
                            device=device, num_val_steps=10, max_val_samples=config.val_samples_per_run,
                        )
                        writer.add_scalar("validation/latent_mse", val_mse, global_step)
                    dist.barrier()

                # Save checkpoint
                if global_step % config.save_every_steps == 0:
                    save_dir = output_dir / f"checkpoint-{global_step}"
                    if is_main:
                        save_dir.mkdir(parents=True, exist_ok=True)
                    # Save full transformer state dict
                    with FSDP.state_dict_type(
                        transformer, StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    ):
                        if is_main:
                            torch.save(transformer.state_dict(), save_dir / "transformer.pt")
                            torch.save(semantic_processor.module.state_dict(), save_dir / "semantic_processor.pt")
                            torch.save({
                                "epoch": epoch, "global_step": global_step,
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                            }, save_dir / "training_state.pt")
                            # Symlink latest
                            latest = output_dir / "latest_checkpoint"
                            latest.unlink(missing_ok=True) if latest.is_symlink() else None
                            if latest.exists():
                                import shutil
                                shutil.rmtree(latest, ignore_errors=True)
                            latest.symlink_to(save_dir.resolve())
                            logger.info(f"Saved checkpoint to {save_dir}")
                    dist.barrier()

    # ========== 8. Final save ==========
    final_dir = output_dir / "final"
    if is_main:
        final_dir.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(
        transformer, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        if is_main:
            torch.save(transformer.state_dict(), final_dir / "transformer.pt")
            torch.save(semantic_processor.module.state_dict(), final_dir / "semantic_processor.pt")
            if writer:
                writer.close()
            logger.info(f"Training complete! Saved to {final_dir}")

    dist.barrier()
    dist.destroy_process_group()


# ============================================================================
# 推理
# ============================================================================


def vae_encode(vae, images, scaling_factor, shift_factor):
    with torch.no_grad():
        h = vae.encoder(images)
        if vae.quant_conv is not None:
            h = vae.quant_conv(h)
        mean, _ = h.chunk(2, dim=1)
        return (mean - shift_factor) * scaling_factor


def extract_siglip_features(siglip_model, siglip_processor, images, device):
    with torch.no_grad():
        inputs = siglip_processor(images=images, return_tensors="pt").to(device)
        outputs = siglip_model.vision_model(pixel_values=inputs["pixel_values"])
        return outputs.last_hidden_state.cpu()


def encode_text(tokenizer, text_encoder, prompts, max_length, device):
    formatted = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        formatted.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
    text_inputs = tokenizer(formatted, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()
    with torch.no_grad():
        hidden = text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-2]
    return [hidden[i][attention_mask[i]].cpu() for i in range(len(hidden))]


def inference(config: FullParamConfig, weights_path: str, source_path: str, prompt: str):
    """使用全参数微调权重进行推理。"""
    from torchvision import transforms
    from transformers import AutoModel, AutoProcessor
    from config import BASE_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_IMAGE_SEQ_LEN, MAX_SHIFT
    from utils.loader import load_sharded_safetensors
    from zimage.transformer import ZImageTransformer2DModel
    from zimage.scheduler import FlowMatchEulerDiscreteScheduler

    device = torch.device("cuda")
    weights_path = Path(weights_path)
    model_dir = Path(config.model_path)

    # Load transformer with finetuned weights
    logger.info("Loading finetuned transformer...")
    transformer_dir = model_dir / "transformer"
    with open(transformer_dir / "config.json") as f:
        tc = json.load(f)
    with torch.device("meta"):
        transformer = ZImageTransformer2DModel(
            in_channels=tc.get("in_channels", 16), dim=tc.get("dim", 3840),
            n_layers=tc.get("n_layers", 30), n_refiner_layers=tc.get("n_refiner_layers", 2),
            n_heads=tc.get("n_heads", 30), n_kv_heads=tc.get("n_kv_heads", 30),
            cap_feat_dim=tc.get("cap_feat_dim", 2560),
            all_patch_size=tuple(tc.get("all_patch_size", [2])),
            all_f_patch_size=tuple(tc.get("all_f_patch_size", [1])),
            norm_eps=tc.get("norm_eps", 1e-5), qk_norm=tc.get("qk_norm", True),
            rope_theta=tc.get("rope_theta", 256.0), t_scale=tc.get("t_scale", 1000.0),
            axes_dims=tc.get("axes_dims", [32, 48, 48]), axes_lens=tc.get("axes_lens", [1536, 512, 512]),
        )
    state_dict = torch.load(weights_path / "transformer.pt", map_location="cpu", weights_only=True)
    transformer.load_state_dict(state_dict, assign=True)
    del state_dict
    transformer = transformer.to(device=device, dtype=torch.bfloat16).eval()

    # Load semantic processor
    semantic_processor = SemanticProcessor(siglip_dim=config.siglip_dim, output_dim=2560).to(device=device, dtype=torch.bfloat16)
    sp_state = torch.load(weights_path / "semantic_processor.pt", map_location=device, weights_only=True)
    semantic_processor.load_state_dict(sp_state)
    semantic_processor.eval()

    # Load other components
    from utils import load_from_local_dir
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    # Don't need the base transformer from components
    del components["transformer"]
    torch.cuda.empty_cache()

    # SigLip-2
    siglip_processor = AutoProcessor.from_pretrained(config.siglip_model_name)
    siglip_model = AutoModel.from_pretrained(config.siglip_model_name).to(device).eval()

    # Process source image
    source_img = Image.open(source_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    with torch.no_grad():
        semantic_feat = extract_siglip_features(siglip_model, siglip_processor, [source_img], device)
        semantic_feat = semantic_feat.to(device=device, dtype=torch.bfloat16)
        semantic_embeds = semantic_processor(semantic_feat).squeeze(0)

        source_tensor = transform(source_img).unsqueeze(0).to(device, dtype=torch.float32)
        source_latents = vae_encode(vae, source_tensor, config.vae_scaling_factor, config.vae_shift_factor)
        source_latents = source_latents.unsqueeze(2).to(torch.bfloat16)

        text_emb = encode_text(tokenizer, text_encoder, [prompt], config.max_sequence_length, device)[0].to(device=device, dtype=torch.bfloat16)
        cap_feats = torch.cat([text_emb, semantic_embeds], dim=0)

        neg_text_emb = encode_text(tokenizer, text_encoder, [""], config.max_sequence_length, device)[0].to(device=device, dtype=torch.bfloat16)
        neg_cap_feats = torch.cat([neg_text_emb, torch.zeros_like(semantic_embeds)], dim=0)

    # Denoise
    height_latent = config.resolution // 8
    width_latent = config.resolution // 8
    image_seq_len = (height_latent // 2) * (width_latent // 2)
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN)
    b_val = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN
    mu = image_seq_len * m + b_val

    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(config.inference_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps

    latents = torch.randn(1, 16, 1, height_latent, width_latent, device=device, dtype=torch.float32)

    logger.info(f"Denoising: {config.inference_steps} steps, cfg={config.guidance_scale}")
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps, desc="Edit denoising")):
            if t == 0 and i == len(timesteps) - 1:
                continue
            combined = torch.cat([latents, source_latents.to(latents.dtype)], dim=2)
            timestep_model = (1000 - t.expand(1)) / 1000
            x_list = [combined[0].to(torch.bfloat16)]

            if config.guidance_scale > 1.0:
                pred_cond = transformer(x_list, timestep_model, [cap_feats])[0][0]
                pred_uncond = transformer(x_list, timestep_model, [neg_cap_feats])[0][0]
                noise_pred = pred_uncond[:, :1] + config.guidance_scale * (pred_cond[:, :1] - pred_uncond[:, :1])
            else:
                noise_pred = transformer(x_list, timestep_model, [cap_feats])[0][0][:, :1]

            noise_pred = -noise_pred.float().unsqueeze(0).squeeze(2)
            latents_2d = scheduler.step(noise_pred, t, latents.squeeze(2), return_dict=False)[0]
            latents = latents_2d.unsqueeze(2)

    # Decode
    shift_factor = getattr(vae.config, "shift_factor", 0.0) if hasattr(vae, "config") else 0.0
    decode_latents = (latents.squeeze(2).to(vae.dtype) / config.vae_scaling_factor) + (shift_factor or 0.0)
    from diffusers import AutoencoderKL
    diffusers_vae = AutoencoderKL.from_pretrained(str(model_dir / "vae"), torch_dtype=torch.bfloat16).to(device).eval()
    decoded = (diffusers_vae.decode(decode_latents, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)

    img = decoded[0].cpu().permute(1, 2, 0).float().numpy()
    img = Image.fromarray((img * 255).round().clip(0, 255).astype("uint8"))
    out_path = Path(config.output_dir) / "inference_result.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    logger.info(f"Saved result to {out_path}")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-Image-Edit Full Parameter Fine-tuning (FSDP)")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = FullParamConfig()
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.grad_accum:
        config.gradient_accumulation_steps = args.grad_accum
    if args.val_every:
        config.validate_every_steps = args.val_every
    if args.save_every:
        config.save_every_steps = args.save_every
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    if args.train:
        train(config)
    elif args.inference:
        weights = args.weights or str(Path(config.output_dir) / "final")
        if not args.source or not args.prompt:
            parser.error("--inference requires --source and --prompt")
        inference(config, weights, args.source, args.prompt)
    else:
        parser.print_help()
