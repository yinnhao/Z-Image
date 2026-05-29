# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Z-Image is a 6B-parameter text-to-image generation model built on a Scalable Single-Stream DiT (S3-DiT) architecture with Flow Matching. This repository provides native PyTorch inference, LoRA fine-tuning, image editing training, and ablation study tooling.

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
python img2img_inference.py          # Image-to-image editing (no LoRA, concat-based)
```

### LoRA Training Pipeline (text-to-image)
```bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py --precompute   # Precompute VAE latents + text embeddings
CUDA_VISIBLE_DEVICES=0 python train_lora.py --train        # Train LoRA (TensorBoard: output/lora_3d_icon/runs/)
CUDA_VISIBLE_DEVICES=0 python train_lora.py --inference    # Generate with trained LoRA
```

### Edit LoRA Training Pipeline (image editing)
```bash
# Precompute (single GPU)
CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --precompute \
    --data_dir /path/to/edit_data --prompt_level medium

# Precompute (8-GPU parallel)
bash run_precompute_parallel.sh

# Merge shards after parallel precompute
python train_edit_lora.py --merge_shards 8

# Train
CUDA_VISIBLE_DEVICES=0 python train_edit_lora.py --train

# Inference
python train_edit_lora.py --inference --source input.png --prompt "把 logo 改成绿色"
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

### Edit Pipeline Flow (Z-Image-Edit)
Source image enters through two paths:
1. **Semantic path**: Source → SigLip-2 (frozen) → SemanticProcessor (1024→2560) → concat with text tokens as `cap_feats`
2. **Pixel path**: Source → VAE encode → concat with noised target along T dimension (frame 0=target, frame 1=source)

Unified sequence: `[noised_target_tokens, source_vae_tokens | text_tokens, semantic_tokens]`
- Image tokens → noise_refiner (with timestep modulation)
- Context tokens → context_refiner (no timestep)

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
- SigLip-2: google/siglip2-large-patch16-384 (1024 dim, 576 tokens for 384px)
- Turbo defaults: 1024×1024, 8 steps, guidance_scale=0.0

## Training Notes

### Common
- LoRA targets the DiT transformer (not VAE or text encoder)
- Training uses precomputed latents/embeddings to save VRAM
- The base model (not Turbo) is used for LoRA training — Turbo's distillation makes it unsuitable for fine-tuning
- Training loss target: `latents - noise` (i.e., `x_0 - ε`)
- Model timestep during training: `1 - sigma`

### Edit LoRA Specific
- SemanticProcessor outputs 2560-dim (not 3840) → goes through transformer's existing `cap_embedder` naturally, zero code modification needed
- Source image as frame 1 in T-dimension concatenation, target as frame 0
- Only frame 0 of transformer output is used for loss computation
- Trainable params: LoRA (~159M) + SemanticProcessor (~2.6M)
- CFG dropout: 10% probability to zero out cap_feats during training

### Edit Dataset Format (TSV)
Internal dataset at `/root/paddlejob/workspace/env/vfs_benchmark_cnn/xuziyuan01/zhushou_image_edit_train`:
- 3 subdirs: `apple_full/` (3375), `log_full/` (1899), `text_data/` (1290) = 6564 total
- TSV format: `md5 \t label \t json \t source_b64 \t target_b64 \t short_prompt \t medium_prompt \t long_prompt`
- `--prompt_level`: short/medium/long selects which prompt field to use
- `--shard i/N`: parallel precompute, skips non-matching lines without base64 decode

## File Organization

- `ckpts/` — Model weights (~30GB, gitignored)
- `output/` — Training outputs, precomputed caches (gitignored)
- `3d_icon_data/` — Example training dataset for 3D icon LoRA
- `edit_data/` — Example edit dataset (metadata.jsonl + source/target dirs)
- `docs/` — Detailed architecture and training documentation

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_lora.py` | Text-to-image LoRA fine-tuning (3D icons etc.) |
| `train_edit_lora.py` | Image editing LoRA training (source+prompt→edited) |
| `img2img_inference.py` | Standalone img2img inference (no LoRA, T-dim concat) |
| `run_precompute_parallel.sh` | 8-GPU parallel precompute launcher |
| `inference.py` | Standard text-to-image inference |

## Environment Notes

- Proxy required for HuggingFace: `export https_proxy=http://agent.baidu.com:8891`
- SigLip-2 model auto-downloads on first use (needs proxy)
- 8× A100-40GB available for parallel precompute
- Single A100-40GB sufficient for edit LoRA training (batch_size=1, grad_accum=4)
