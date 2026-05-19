"""
Z-Image LoRA Ablation Study
消融实验：验证各改动的贡献

Baseline (当前最佳):
  - 模型: Z-Image (非 Turbo)
  - 时间步: 1 - sigma (修复后)
  - LoRA: rank=64, alpha=64, target=attn+FFN
  - 训练: 500 epochs, lr=1e-4
  - 推理: 30 steps, CFG=3.5

消融组 (每次只改一个变量):
  A. w/o timestep fix: 直接传 sigma (不做 1-sigma 转换)
  B. Turbo model: 使用 Z-Image-Turbo 替代 Z-Image
  C. Attention only: 去掉 w1/w2/w3, 只保留 attention 层
  D. Rank 16: 降低 rank 从 64 到 16
  E. 8 steps, no CFG: 推理时用 8 步 + guidance_scale=0 (不需重新训练)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AblationConfig:
    """Single ablation experiment configuration."""
    name: str
    description: str
    # Training overrides (None = use baseline)
    model_path: Optional[str] = None
    timestep_fix: Optional[bool] = None  # True = use 1-sigma, False = use sigma
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_targets: Optional[list] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    # Inference overrides
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    # If True, skip training (reuse baseline weights with different inference)
    inference_only: bool = False
    lora_path_override: Optional[str] = None


# ============================================================================
# Ablation Experiments Definition
# ============================================================================

BASELINE = AblationConfig(
    name="baseline",
    description="当前最佳配置 (Z-Image, 1-sigma, rank64, attn+FFN, 30steps+CFG3.5)",
)

ABLATIONS = [
    AblationConfig(
        name="A_no_timestep_fix",
        description="消融: 不做时间步修复, 直接传 sigma 给模型",
        timestep_fix=False,
    ),
    AblationConfig(
        name="B_turbo_model",
        description="消融: 使用 Z-Image-Turbo 模型",
        model_path="ckpts/Z-Image-Turbo",
    ),
    AblationConfig(
        name="C_attention_only",
        description="消融: LoRA 只作用于 attention 层 (去掉 FFN w1/w2/w3)",
        lora_targets=["to_q", "to_k", "to_v", "to_out.0"],
    ),
    AblationConfig(
        name="D_rank16",
        description="消融: LoRA rank 从 64 降到 16",
        lora_rank=16,
        lora_alpha=16,
    ),
    AblationConfig(
        name="E_8steps_no_cfg",
        description="消融: 推理时用 8 步 + 无 CFG (复用 baseline 权重, 不重新训练)",
        inference_only=True,
        lora_path_override="output/lora_3d_icon/lora_weights",
        num_inference_steps=8,
        guidance_scale=0.0,
    ),
]


# ============================================================================
# Training and Inference Runner
# ============================================================================

# Baseline defaults
DEFAULTS = {
    "model_path": "ckpts/Z-Image",
    "timestep_fix": True,
    "lora_rank": 64,
    "lora_alpha": 64,
    "lora_targets": ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"],
    "epochs": 500,
    "learning_rate": 1e-4,
    "num_inference_steps": 30,
    "guidance_scale": 3.5,
    "resolution": 512,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "cfg_dropout_prob": 0.1,
    "seed": 42,
    "validate_every_steps": 100,
    "save_every_steps": 500,
}

EVAL_PROMPTS = None  # Will be loaded from training set (all 23 prompts)


def get_effective_config(ablation: AblationConfig) -> dict:
    """Merge ablation overrides into baseline defaults."""
    cfg = DEFAULTS.copy()
    for key in ["model_path", "timestep_fix", "lora_rank", "lora_alpha",
                "lora_targets", "epochs", "learning_rate",
                "num_inference_steps", "guidance_scale"]:
        val = getattr(ablation, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


def run_experiment(ablation: AblationConfig, output_base: Path):
    """Run a single ablation experiment."""
    import torch
    import numpy as np
    global EVAL_PROMPTS

    # Load all training prompts as evaluation set
    if EVAL_PROMPTS is None:
        prompts_file = Path("output/precomputed/prompts.json")
        with open(prompts_file) as f:
            EVAL_PROMPTS = json.load(f)
        print(f"[Eval] Loaded {len(EVAL_PROMPTS)} prompts from training set")

    exp_dir = output_base / ablation.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_effective_config(ablation)

    # Save experiment config
    with open(exp_dir / "config.json", "w") as f:
        json.dump({"name": ablation.name, "description": ablation.description, **cfg}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Experiment: {ablation.name}")
    print(f"Description: {ablation.description}")
    print(f"{'='*70}\n")

    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])

    if not ablation.inference_only:
        # --- Training ---
        _run_training(cfg, ablation, exp_dir, device)

    # --- Inference ---
    lora_path = ablation.lora_path_override or str(exp_dir / "lora_weights")
    _run_inference(cfg, lora_path, exp_dir, device)

    print(f"\n[Done] {ablation.name} -> {exp_dir}")


def _run_training(cfg: dict, ablation: AblationConfig, exp_dir: Path, device):
    """Training phase of an ablation experiment."""
    import math
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from peft import LoraConfig, get_peft_model
    from utils.loader import load_sharded_safetensors
    from zimage.transformer import ZImageTransformer2DModel
    import bitsandbytes as bnb

    # Check if precomputed cache exists for this model
    # VAE params are identical between Z-Image and Z-Image-Turbo, share cache
    cache_dir = Path("output/precomputed")
    if not (cache_dir / "latents.pt").exists():
        print(f"[Precompute] Generating cache for {cfg['model_path']}...")
        _precompute_for_model(cfg, cache_dir, device)

    # Load transformer
    model_dir = Path(cfg["model_path"])
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

    state_dict = load_sharded_safetensors(transformer_dir)
    transformer.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    transformer = transformer.to(device=device, dtype=torch.bfloat16)
    transformer.eval()

    # Inject LoRA
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_targets"],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    for param in transformer.parameters():
        if not param.requires_grad:
            param.requires_grad_(False)
    transformer.train()

    # Dataset
    sys.path.insert(0, str(Path(__file__).parent))
    from train_lora import PrecomputedDataset, collate_fn
    dataset = PrecomputedDataset(str(cache_dir))
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )

    # Optimizer
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg["learning_rate"], weight_decay=0.01)

    total_steps = cfg["epochs"] * math.ceil(len(dataset) / cfg["batch_size"])
    effective_steps = total_steps // cfg["gradient_accumulation_steps"]

    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        progress = (step - cfg["warmup_steps"]) / max(1, effective_steps - cfg["warmup_steps"])
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(exp_dir / "tensorboard"))

    # Training loop
    global_step = 0
    running_loss = 0.0
    print(f"[Train] {cfg['epochs']} epochs, ~{effective_steps} effective steps")

    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(dataloader):
            latents = batch["latents"].to(device, dtype=torch.bfloat16)
            embeddings = [e.to(device, dtype=torch.bfloat16) for e in batch["embeddings"]]
            B = latents.shape[0]

            # CFG dropout
            if cfg["cfg_dropout_prob"] > 0:
                for i in range(B):
                    if torch.rand(1).item() < cfg["cfg_dropout_prob"]:
                        embeddings[i] = torch.zeros(1, embeddings[i].shape[-1], device=device, dtype=torch.bfloat16)

            # Timestep sampling
            sigma = torch.sigmoid(torch.randn(B, device=device, dtype=torch.bfloat16))

            # Noisy latents
            noise = torch.randn_like(latents)
            sigma_expand = sigma[:, None, None, None]
            noisy_latents = (1 - sigma_expand) * latents + sigma_expand * noise

            # Target
            target = latents - noise

            # Timestep to pass to model
            if cfg["timestep_fix"]:
                model_timestep = 1 - sigma
            else:
                model_timestep = sigma  # Ablation: no fix

            # Forward
            x_list = [noisy_latents[i].unsqueeze(1) for i in range(B)]
            pred_list, _ = transformer(x_list, model_timestep, embeddings)
            pred = torch.stack([p.squeeze(1) for p in pred_list])

            # Loss
            loss = F.mse_loss(pred.float(), target.float())
            loss = loss / cfg["gradient_accumulation_steps"]
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = running_loss / 10
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    if global_step % 50 == 0:
                        print(f"  Step {global_step} | Loss: {avg_loss:.6f}")
                    running_loss = 0.0

    # Save LoRA weights
    save_path = exp_dir / "lora_weights"
    transformer.save_pretrained(save_path)
    writer.close()
    print(f"[Train] Saved LoRA weights to {save_path}")

    # Cleanup
    del transformer, optimizer, trainable_params
    torch.cuda.empty_cache()


def _precompute_for_model(cfg: dict, cache_dir: Path, device):
    """Precompute latents and embeddings for a given model."""
    import torch
    from torchvision import transforms
    from datasets import load_dataset
    from utils import load_from_local_dir
    from train_lora import vae_encode, encode_text

    cache_dir.mkdir(parents=True, exist_ok=True)

    components = load_from_local_dir(cfg["model_path"], device=device, dtype=torch.bfloat16)
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    ds = load_dataset("linoyts/3d_icon", split="train")

    transform = transforms.Compose([
        transforms.Resize(cfg["resolution"], interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(cfg["resolution"]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    all_latents, all_embeddings, all_prompts = [], [], []
    for sample in ds:
        img = sample["image"].convert("RGB")
        prompt = sample["prompt"]
        all_prompts.append(prompt)

        img_tensor = transform(img).unsqueeze(0).to(device, dtype=torch.float32)
        latent = vae_encode(vae, img_tensor, 0.3611, 0.1159)
        all_latents.append(latent.squeeze(0).cpu())

        emb = encode_text(tokenizer, text_encoder, [prompt], 512, device)
        all_embeddings.append(emb[0])

    torch.save(all_latents, cache_dir / "latents.pt")
    torch.save(all_embeddings, cache_dir / "embeddings.pt")
    with open(cache_dir / "prompts.json", "w") as f:
        json.dump(all_prompts, f, ensure_ascii=False)

    del vae, text_encoder, tokenizer, components
    torch.cuda.empty_cache()
    print(f"[Precompute] Cached {len(all_latents)} samples to {cache_dir}")


def _run_inference(cfg: dict, lora_path: str, exp_dir: Path, device):
    """Inference phase: generate images with LoRA weights."""
    import torch
    import numpy as np
    from peft import PeftModel
    from utils import load_from_local_dir
    from zimage.pipeline import generate

    # Determine which model to load for inference
    # For ablation E (inference_only), use baseline model
    model_path = cfg["model_path"]

    print(f"[Inference] Loading model from {model_path}...")
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    transformer = components["transformer"]
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    scheduler = components["scheduler"]

    # Load and merge LoRA
    print(f"[Inference] Loading LoRA from {lora_path}...")
    transformer = PeftModel.from_pretrained(transformer, lora_path)
    transformer = transformer.merge_and_unload()
    transformer.eval()

    # Generate for each eval prompt (all training set prompts)
    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        generator = torch.Generator(device).manual_seed(cfg["seed"])
        images = generate(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompt,
            height=cfg["resolution"],
            width=cfg["resolution"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"],
            generator=generator,
        )
        # Use short prompt as filename
        short_name = prompt.replace("a 3dicon, ", "").replace(" ", "_")[:40]
        save_path = samples_dir / f"{i:02d}_{short_name}.png"
        images[0].save(save_path)
        results.append({"prompt": prompt, "file": str(save_path)})
        print(f"  [{i+1:2d}/{len(EVAL_PROMPTS)}] {prompt[:50]}")

    # Save prompt-to-file mapping
    with open(samples_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    del transformer, vae, text_encoder, tokenizer, components
    torch.cuda.empty_cache()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Z-Image LoRA Ablation Study")
    parser.add_argument("--run", type=str, nargs="*", default=None,
                        help="Run specific experiments by name (default: all). "
                             "Options: baseline, A_no_timestep_fix, B_turbo_model, "
                             "C_attention_only, D_rank16, E_8steps_no_cfg")
    parser.add_argument("--output_dir", type=str, default="output/ablation",
                        help="Base output directory for all experiments")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable ablation experiments:")
        print(f"  {'baseline':<25} {BASELINE.description}")
        for ab in ABLATIONS:
            print(f"  {ab.name:<25} {ab.description}")
        return

    output_base = Path(args.output_dir)

    # Determine which experiments to run
    all_experiments = [BASELINE] + ABLATIONS
    if args.run is None:
        experiments = all_experiments
    else:
        name_map = {e.name: e for e in all_experiments}
        experiments = []
        for name in args.run:
            if name not in name_map:
                print(f"Error: Unknown experiment '{name}'")
                print(f"Available: {list(name_map.keys())}")
                return
            experiments.append(name_map[name])

    print(f"\n{'#'*70}")
    print(f"# Z-Image LoRA Ablation Study")
    print(f"# Running {len(experiments)} experiment(s)")
    print(f"# Output: {output_base}")
    print(f"{'#'*70}")

    for exp in experiments:
        run_experiment(exp, output_base)

    # Summary
    print(f"\n\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_base}/")
    print("\nTo compare results:")
    print(f"  1. TensorBoard: tensorboard --logdir {output_base}")
    print(f"  2. Images: ls {output_base}/*/samples/")
    print("\nExperiment layout:")
    for exp in experiments:
        print(f"  {output_base}/{exp.name}/")
        print(f"    ├── config.json       (实验配置)")
        print(f"    ├── tensorboard/      (loss 曲线)")
        print(f"    ├── lora_weights/     (LoRA 权重)")
        print(f"    └── samples/          (生成图片)")


if __name__ == "__main__":
    main()
