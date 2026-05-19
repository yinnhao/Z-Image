"""
Z-Image LoRA Fine-tuning Script
基于 Flow Matching 的 DiT 模型 LoRA 微调，使用 linoyts/3d_icon 数据集。

Usage:
    # Step 1: 预计算 latents 和 text embeddings（只需运行一次）
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute

    # Step 2: 训练
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --train

    # Step 3: 推理验证
    CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference --lora_path output/lora_weights
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
# Configuration
# ============================================================================

class TrainConfig:
    model_path = "ckpts/Z-Image-Turbo"
    dataset_name = "linoyts/3d_icon"
    output_dir = "output/lora_3d_icon"
    cache_dir = "output/precomputed"

    resolution = 512
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    epochs = 100
    warmup_steps = 50
    save_every_steps = 500
    log_every_steps = 10
    validate_every_steps = 100
    validate_prompt = "a 3dicon, a cute cat on purple background"
    seed = 42

    # LoRA
    lora_rank = 64
    lora_alpha = 64
    lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]

    # Flow matching
    cfg_dropout_prob = 0.1

    # VAE constants (from Z-Image-Turbo config)
    vae_scaling_factor = 0.3611
    vae_shift_factor = 0.1159

    # Text encoding
    max_sequence_length = 512


# ============================================================================
# Utility Functions
# ============================================================================

def vae_encode(vae, images, scaling_factor, shift_factor):
    """Encode images to latent space using VAE encoder."""
    with torch.no_grad():
        h = vae.encoder(images)
        if vae.quant_conv is not None:
            h = vae.quant_conv(h)
        mean, _ = h.chunk(2, dim=1)
        latents = (mean - shift_factor) * scaling_factor
    return latents


def encode_text(tokenizer, text_encoder, prompts, max_length, device):
    """Encode text prompts to embeddings using the text encoder."""
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


# ============================================================================
# Precomputation
# ============================================================================

def precompute(config: TrainConfig):
    """Precompute VAE latents and text embeddings for the dataset."""
    from datasets import load_dataset
    from utils import load_from_local_dir

    device = torch.device("cuda")
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models for precomputation...")
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    vae = components["vae"]  # Already in float32
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    logger.info("Loading dataset...")
    ds = load_dataset(config.dataset_name, split="train")

    transform = transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(config.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
    ])

    logger.info(f"Precomputing {len(ds)} samples...")
    all_latents = []
    all_embeddings = []
    all_prompts = []

    for i, sample in enumerate(tqdm(ds, desc="Encoding")):
        img = sample["image"].convert("RGB")
        prompt = sample["prompt"]
        all_prompts.append(prompt)

        # VAE encode
        img_tensor = transform(img).unsqueeze(0).to(device, dtype=torch.float32)
        latent = vae_encode(vae, img_tensor, config.vae_scaling_factor, config.vae_shift_factor)
        all_latents.append(latent.squeeze(0).cpu())

        # Text encode
        emb = encode_text(tokenizer, text_encoder, [prompt], config.max_sequence_length, device)
        all_embeddings.append(emb[0])

    # Save
    torch.save(all_latents, cache_dir / "latents.pt")
    torch.save(all_embeddings, cache_dir / "embeddings.pt")
    with open(cache_dir / "prompts.json", "w") as f:
        json.dump(all_prompts, f, ensure_ascii=False)

    logger.info(f"Saved to {cache_dir}/")
    logger.info(f"  latents.pt: {len(all_latents)} items, shape={all_latents[0].shape}")
    logger.info(f"  embeddings.pt: {len(all_embeddings)} items, shape[0]={all_embeddings[0].shape}")

    # Cleanup GPU memory
    del vae, text_encoder, tokenizer, components
    torch.cuda.empty_cache()


# ============================================================================
# Dataset
# ============================================================================

class PrecomputedDataset(Dataset):
    """Dataset loading precomputed latents and text embeddings."""

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
    """Custom collate for variable-length embeddings."""
    latents = torch.stack([item["latent"] for item in batch])
    embeddings = [item["embedding"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    return {"latents": latents, "embeddings": embeddings, "prompts": prompts}


# ============================================================================
# Training
# ============================================================================

def train(config: TrainConfig):
    """Main training loop with LoRA."""
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

    # --- Load Transformer ---
    logger.info("Loading transformer...")
    model_dir = Path(config.model_path)
    transformer_dir = model_dir / "transformer"

    # Load config
    with open(transformer_dir / "config.json") as f:
        transformer_config = json.load(f)

    # Instantiate on meta device then load weights
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

    # Load weights to CPU then move to GPU
    state_dict = load_sharded_safetensors(transformer_dir)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    transformer.eval()

    # --- Inject LoRA ---
    logger.info("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    transformer.get_input_embeddings = lambda: None  # PEFT compatibility
    # Enable grads on inputs for LoRA backward pass
    for param in transformer.parameters():
        if not param.requires_grad:
            param.requires_grad_(False)

    transformer.train()

    # --- Dataset ---
    dataset = PrecomputedDataset(config.cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # --- Load validation components (VAE, text_encoder, scheduler) ---
    logger.info("Loading validation components (VAE, text_encoder, scheduler)...")
    val_components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    val_vae = val_components["vae"]
    val_text_encoder = val_components["text_encoder"]
    val_tokenizer = val_components["tokenizer"]
    val_scheduler = val_components["scheduler"]
    # We don't need the transformer from val_components, we use our LoRA one
    del val_components["transformer"]
    torch.cuda.empty_cache()

    # --- Optimizer ---
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config.learning_rate, weight_decay=0.01)

    # --- LR Scheduler ---
    total_steps = config.epochs * math.ceil(len(dataset) / config.batch_size)
    effective_steps = total_steps // config.gradient_accumulation_steps

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, effective_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training Loop ---
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    running_loss = 0.0

    # --- TensorBoard ---
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
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            embeddings = [e.to(device, dtype=torch.bfloat16) for e in batch["embeddings"]]
            B = latents.shape[0]

            # CFG dropout: randomly drop text conditioning
            if config.cfg_dropout_prob > 0:
                for i in range(B):
                    if torch.rand(1).item() < config.cfg_dropout_prob:
                        embeddings[i] = torch.zeros(1, embeddings[i].shape[-1], device=device, dtype=torch.bfloat16)

            # Sample timestep (logit-normal distribution)
            # sigma here represents noise level: 0=clean, 1=pure noise
            sigma = torch.sigmoid(torch.randn(B, device=device, dtype=torch.bfloat16))

            # Create noisy latents (flow matching interpolation)
            noise = torch.randn_like(latents)
            sigma_expand = sigma[:, None, None, None]
            noisy_latents = (1 - sigma_expand) * latents + sigma_expand * noise

            # Target: model predicts (latents - noise), pipeline negates it for scheduler
            target = latents - noise

            # Convert sigma to model timestep convention:
            # Pipeline uses (1000-t)/1000 where t=1000 is noisy, so model sees 0=noisy, 1=clean
            # sigma is noise level (1=noisy, 0=clean), so we pass 1-sigma to match pipeline
            model_timestep = 1 - sigma

            # Prepare model inputs (add frame dimension)
            x_list = [noisy_latents[i].unsqueeze(1) for i in range(B)]  # each [16, 1, H, W]
            cap_feats_list = embeddings

            # Forward pass (call through PEFT wrapper)
            pred_list, _ = transformer(x_list, model_timestep, cap_feats_list)
            pred = torch.stack([p.squeeze(1) for p in pred_list])

            # Loss
            loss = F.mse_loss(pred.float(), target.float())
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()

            # Gradient accumulation step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every_steps == 0:
                    avg_loss = running_loss / config.log_every_steps
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(f"Step {global_step} | Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running_loss = 0.0

                if global_step % config.validate_every_steps == 0:
                    # Generate a validation sample
                    logger.info(f"[Validate] Generating sample at step {global_step}...")
                    transformer.eval()
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
                            num_inference_steps=8,
                            guidance_scale=0.0,
                            generator=val_generator,
                        )
                    # Log image to TensorBoard
                    val_img = val_images[0]
                    val_img_np = np.array(val_img).transpose(2, 0, 1)  # HWC -> CHW
                    writer.add_image("validation/sample", val_img_np, global_step)
                    # Also save to disk
                    val_dir = output_dir / "validation"
                    val_dir.mkdir(parents=True, exist_ok=True)
                    val_img.save(val_dir / f"step_{global_step:06d}.png")
                    logger.info(f"[Validate] Saved: {val_dir}/step_{global_step:06d}.png")
                    transformer.train()

                if global_step % config.save_every_steps == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    transformer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

    # Save final weights
    final_path = output_dir / "lora_weights"
    transformer.save_pretrained(final_path)
    writer.close()
    logger.info(f"Training complete! LoRA weights saved to {final_path}")
    logger.info(f"TensorBoard logs: tensorboard --logdir {tb_log_dir}")


# ============================================================================
# Inference with LoRA
# ============================================================================

def inference(config: TrainConfig, lora_path: str, prompt: str = None):
    """Generate images using the base model + LoRA weights."""
    from peft import PeftModel, LoraConfig
    from utils import load_from_local_dir
    from zimage.pipeline import generate

    device = torch.device("cuda")

    logger.info("Loading base model...")
    components = load_from_local_dir(config.model_path, device=device, dtype=torch.bfloat16)
    transformer = components["transformer"]
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    scheduler = components["scheduler"]

    # Load LoRA weights
    logger.info(f"Loading LoRA weights from {lora_path}...")
    transformer = PeftModel.from_pretrained(transformer, lora_path)
    transformer = transformer.merge_and_unload()
    transformer.eval()

    # Generate
    if prompt is None:
        prompt = "a 3dicon, a cute cat icon on a purple background"

    logger.info(f"Generating with prompt: {prompt}")
    generator = torch.Generator(device).manual_seed(config.seed)

    images = generate(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prompt=prompt,
        height=config.resolution,
        width=config.resolution,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=generator,
    )

    # Save
    output_dir = Path(config.output_dir) / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_path = output_dir / f"lora_sample_{i}.png"
        img.save(save_path)
        logger.info(f"Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z-Image LoRA Fine-tuning")
    parser.add_argument("--precompute", action="store_true", help="Precompute latents and text embeddings")
    parser.add_argument("--train", action="store_true", help="Run LoRA training")
    parser.add_argument("--inference", action="store_true", help="Run inference with LoRA weights")
    parser.add_argument("--lora_path", type=str, default="output/lora_3d_icon/lora_weights", help="Path to LoRA weights")
    parser.add_argument("--prompt", type=str, default=None, help="Inference prompt")

    # Override config
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)

    args = parser.parse_args()
    config = TrainConfig()

    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.rank:
        config.lora_rank = args.rank
        config.lora_alpha = args.rank
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.resolution:
        config.resolution = args.resolution

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
