# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Z-Image is a 6B-parameter text-to-image generation model built on a Scalable Single-Stream DiT (S3-DiT) architecture with Flow Matching. This repository provides native PyTorch inference, LoRA fine-tuning, and ablation study tooling.

Two model variants:
- **Z-Image** (base): 50-step inference with classifier-free guidance, used as base for fine-tuning
- **Z-Image-Turbo**: 8-step distilled version (no CFG), default for fast inference

## Commands

### Installation
```bash
pip install -e .          # Core dependencies
pip install -e ".[dev]"   # + black, isort, ruff
```

### Inference
```bash
python inference.py                  # Native PyTorch (Turbo, 8 steps)
python infer_with_diffusers.py       # HuggingFace diffusers pipeline
python batch_inference.py            # Batch from test_prompts.txt
```

### LoRA Training Pipeline
```bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute   # Precompute VAE latents + text embeddings
CUDA_VISIBLE_DEVICES=0 python train_lora.py --train        # Train LoRA (TensorBoard: output/lora_3d_icon/runs/)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference    # Generate with trained LoRA
```

### Ablation Studies
```bash
python ablation_study.py --run <experiment_name> --output_dir output/ablation
./run_ablation_parallel.sh   # 4-GPU parallel ablation
```

### Formatting
```bash
black .
isort .
ruff check .
```

## Architecture

### Generation Pipeline Flow
1. **Text Encoding**: Qwen3 tokenizer → text embeddings `[N, 2560]`
2. **Noise Sampling**: Random latent `[B, 16, H/16, W/16]`
3. **Denoising**: DiT transformer iterates via FlowMatchEulerDiscreteScheduler (Euler ODE)
4. **VAE Decode**: Latent → pixels via AutoencoderKL

### Source Layout (`src/`)
- `src/zimage/` — Core model: `transformer.py` (DiT), `autoencoder.py` (VAE), `pipeline.py` (denoising loop), `scheduler.py` (Flow Matching Euler)
- `src/utils/` — Attention dispatch, model loading, weight verification
- `src/config/` — Model hyperparameters and inference defaults
- `src/config/manifests/` — MD5 checksums for weight integrity

### Critical: Sign Convention (non-standard)

Z-Image uses an inverted "denoising-centric" convention unlike standard Flow Matching:

| Aspect | Standard | Z-Image |
|--------|----------|---------|
| Prediction target | `ε - x_0` (noise direction) | `x_0 - ε` (denoise direction) |
| Timestep 0 | Clean image | Pure noise |
| Timestep 1 | Pure noise | Clean image |

This means the inference pipeline must adapt when interfacing with diffusers-style schedulers:
- Timestep conversion: `(1000 - t) / 1000`
- Output negation: `-noise_pred`

See `docs/pipeline_sign_convention.md` for full derivation.

### Key Model Parameters
- Transformer: dim=3840, 30 layers + 2 refiner, 30 heads, patch_size=2
- VAE: 16 latent channels, scale_factor=8
- Text encoder: Qwen3, cap_feat_dim=2560
- Turbo defaults: 1024×1024, 8 steps, guidance_scale=0.0

## Training Notes

- LoRA targets the DiT transformer (not VAE or text encoder)
- Training uses precomputed latents/embeddings to save VRAM
- The base model (not Turbo) is used for LoRA training — Turbo's distillation makes it unsuitable for fine-tuning
- Training loss target: `latents - noise` (i.e., `x_0 - ε`)
- Model timestep during training: `1 - sigma`

## File Organization

- `ckpts/` — Model weights (~30GB, gitignored)
- `output/` — Training outputs, precomputed caches (gitignored)
- `3d_icon_data/` — Example training dataset for 3D icon LoRA
- `docs/` — Detailed architecture and training documentation
