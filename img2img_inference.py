"""Z-Image Img2Img (Edit) Inference.

Standalone script for image-to-image generation using Z-Image.
Based on the edit_plus pipeline approach: the input image is VAE-encoded
and concatenated with noise latents along the temporal (T) dimension,
allowing the transformer to attend to the reference image during denoising.

Usage:
    python img2img_inference.py

Key parameters:
    strength: Controls noise level. 0.0 = no noise (reconstruct), 1.0 = full noise (ignore input).
"""

import inspect
import os
import time
import warnings
from typing import List, Optional, Union

import torch
import torchvision.transforms.functional as TF
from PIL import Image

warnings.filterwarnings("ignore")
from loguru import logger

from config import (
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
)
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend


# ─── Helper functions ───────────────────────────────────────────────────────────


def preprocess_image(image, height, width):
    """PIL Image -> tensor [-1, 1], resize to target (height, width)."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    image_tensor = TF.to_tensor(image)  # [0, 1]
    image_tensor = image_tensor * 2.0 - 1.0  # [-1, 1]
    return image_tensor.unsqueeze(0)  # [1, 3, H, W]


def calculate_shift(
    image_seq_len,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError("The scheduler does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError("The scheduler does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def encode_image_to_latents(vae, image_tensor, device):
    """Encode preprocessed image tensor to VAE latents."""
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
    h = vae.encoder(image_tensor)
    if hasattr(vae, "quant_conv") and vae.quant_conv is not None:
        h = vae.quant_conv(h)
    mean, _ = h.chunk(2, dim=1)
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    init_latents = (mean - shift_factor) * vae.config.scaling_factor
    return init_latents  # [1, 16, H/8, W/8]


# ─── Img2Img Generation (Edit-style) ───────────────────────────────────────────


@torch.no_grad()
def generate_img2img(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt: Union[str, List[str]],
    image: "PIL.Image.Image",
    strength: float = 0.6,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    cfg_normalization: bool = False,
    cfg_truncation: float = DEFAULT_CFG_TRUNCATION,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    output_type: str = "pil",
):
    """Generate images using img2img pipeline (edit-style concatenation).

    The input image is VAE-encoded and concatenated with noise latents along
    the temporal dimension. The transformer attends to both during denoising.
    Only the noise portion of the output is used for the scheduler step.

    Args:
        image: Input PIL Image as reference.
        strength: Noise strength (0.0 = reconstruct input, 1.0 = ignore input structure).
        Other args: Same as generate().
    """
    device = next(transformer.parameters()).device

    if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    else:
        vae_scale_factor = 8
    vae_scale = vae_scale_factor * 2

    # Auto-infer height/width from input image, aligned to vae_scale
    if height is None:
        height = (image.height // vae_scale) * vae_scale
    if width is None:
        width = (image.width // vae_scale) * vae_scale

    if height % vae_scale != 0:
        raise ValueError(f"Height must be divisible by {vae_scale} (got {height}).")
    if width % vae_scale != 0:
        raise ValueError(f"Width must be divisible by {vae_scale} (got {width}).")

    if isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    else:
        batch_size = len(prompt)

    do_classifier_free_guidance = guidance_scale > 1.0
    logger.info(
        f"Img2img (edit): {height}x{width}, steps={num_inference_steps}, "
        f"strength={strength}, cfg={guidance_scale}"
    )

    # --- VAE encode input image ---
    image_tensor = preprocess_image(image, height, width)
    image_latents = encode_image_to_latents(vae, image_tensor, device)
    # image_latents shape: [1, 16, H/8, W/8] -> add T dim -> [1, 16, 1, H/8, W/8]
    image_latents = image_latents.unsqueeze(2).float()

    # --- Text encoding ---
    formatted_prompts = []
    for p in prompt:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(formatted_prompt)

    text_inputs = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_list = []
    for i in range(len(prompt_embeds)):
        prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

    negative_prompt_embeds_list = []
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = ["" for _ in prompt]
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        neg_formatted = []
        for p in negative_prompt:
            messages = [{"role": "user", "content": p}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            neg_formatted.append(formatted_prompt)

        neg_inputs = tokenizer(
            neg_formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        neg_input_ids = neg_inputs.input_ids.to(device)
        neg_masks = neg_inputs.attention_mask.to(device).bool()

        neg_embeds = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        for i in range(len(neg_embeds)):
            negative_prompt_embeds_list.append(neg_embeds[i][neg_masks[i]])

    if num_images_per_prompt > 1:
        prompt_embeds_list = [pe for pe in prompt_embeds_list for _ in range(num_images_per_prompt)]
        if do_classifier_free_guidance:
            negative_prompt_embeds_list = [
                npe for npe in negative_prompt_embeds_list for _ in range(num_images_per_prompt)
            ]

    # --- Prepare latents ---
    height_latent = 2 * (int(height) // vae_scale)
    width_latent = 2 * (int(width) // vae_scale)
    actual_batch_size = batch_size * num_images_per_prompt

    # Expand image_latents to match batch
    image_latents = image_latents.repeat(actual_batch_size, 1, 1, 1, 1)

    # Generate noise latents [B, 16, 1, H_lat, W_lat]
    noise_shape = (actual_batch_size, transformer.in_channels, 1, height_latent, width_latent)
    latents = torch.randn(noise_shape, generator=generator, device=device, dtype=torch.float32)

    # Apply strength: mix noise with image latents for the noise portion
    # strength=1.0 -> pure noise, strength=0.0 -> pure image latents (no change)
    if strength < 1.0:
        latents = strength * latents + (1.0 - strength) * image_latents

    # For image_seq_len calculation, we use only the noise latent portion
    # (since that's what determines the output resolution)
    image_seq_len = (height_latent // 2) * (width_latent // 2)

    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        **scheduler_kwargs,
    )

    # --- Denoising loop ---
    logger.info(f"Sampling loop start: {num_inference_steps} steps")

    from tqdm import tqdm

    for i, t in enumerate(tqdm(timesteps, desc="Img2img denoising", total=len(timesteps))):
        if t == 0 and i == len(timesteps) - 1:
            continue

        # Concatenate noise latents with image latents along T dimension
        # noise_latents: [B, C, 1, H, W], image_latents: [B, C, 1, H, W]
        # combined: [B, C, 2, H, W]
        combined_latents = torch.cat([latents, image_latents.to(latents.dtype)], dim=2)

        timestep = t.expand(latents.shape[0])
        timestep = (1000 - timestep) / 1000
        t_norm = timestep[0].item()

        current_guidance_scale = guidance_scale
        if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
            if t_norm > cfg_truncation:
                current_guidance_scale = 0.0

        apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

        if apply_cfg:
            combined_typed = combined_latents.to(
                transformer.dtype if hasattr(transformer, "dtype") else next(transformer.parameters()).dtype
            )
            latent_model_input = combined_typed.repeat(2, 1, 1, 1, 1)
            prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
            timestep_model_input = timestep.repeat(2)
        else:
            latent_model_input = combined_latents.to(next(transformer.parameters()).dtype)
            prompt_embeds_model_input = prompt_embeds_list
            timestep_model_input = timestep

        # Transformer expects list of [C, T, H, W] tensors
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        model_out_list = transformer(
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
        )[0]

        # Extract only the noise portion (first frame T=0) from output
        # model_out_list elements are [C, 2, H, W], take [:, :1, :, :]
        model_out_list = [out[:, :1, :, :] for out in model_out_list]

        if apply_cfg:
            pos_out = model_out_list[:actual_batch_size]
            neg_out = model_out_list[actual_batch_size:]
            noise_pred = []
            for j in range(actual_batch_size):
                pos = pos_out[j].float()
                neg = neg_out[j].float()
                pred = pos + current_guidance_scale * (pos - neg)

                if cfg_normalization and float(cfg_normalization) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(pos)
                    new_pos_norm = torch.linalg.vector_norm(pred)
                    max_new_norm = ori_pos_norm * float(cfg_normalization)
                    if new_pos_norm > max_new_norm:
                        pred = pred * (max_new_norm / new_pos_norm)
                noise_pred.append(pred)
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            noise_pred = torch.stack([x.float() for x in model_out_list], dim=0)

        noise_pred = -noise_pred.squeeze(2)  # [B, C, H, W]
        latents_2d = latents.squeeze(2)  # [B, C, H, W]
        latents_2d = scheduler.step(noise_pred.to(torch.float32), t, latents_2d, return_dict=False)[0]
        latents = latents_2d.unsqueeze(2)  # back to [B, C, 1, H, W]
        assert latents.dtype == torch.float32

    # --- VAE decode ---
    if output_type == "latent":
        return latents.squeeze(2)

    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    decode_latents = (latents.squeeze(2).to(vae.dtype) / vae.config.scaling_factor) + shift_factor
    decoded = vae.decode(decode_latents, return_dict=False)[0]

    if output_type == "pil":
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
        decoded = (decoded * 255).round().astype("uint8")
        decoded = [Image.fromarray(img) for img in decoded]

    return decoded


# ─── Main ───────────────────────────────────────────────────────────────────────


def main():
    model_path = ensure_model_weights("ckpts/Z-Image", verify=False)
    dtype = torch.bfloat16
    compile = False
    output_path = "output_img2img_edit.png"
    input_image_path = "input.png"  # Replace with your input image path
    prompt = "turn the netflix logo to green"
    strength = 0.6  # 0.0 = no change, 1.0 = full txt2img
    height = None  # None = auto-infer from input image
    width = None
    num_inference_steps = 50
    guidance_scale = 5.0
    seed = 42
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Chosen device: mps")
            else:
                device = "cpu"
                print("Chosen device: cpu")

    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    # Load input image
    image = Image.open(input_image_path).convert("RGB")
    print(f"Input image: {input_image_path} ({image.width}x{image.height})")

    # Generate img2img
    start_time = time.time()
    images = generate_img2img(
        prompt=prompt,
        image=image,
        strength=strength,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    images[0].save(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
