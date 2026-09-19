"""Krea2 CcC Paint node."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY
from ..paint_patch import patch_krea2_paint_model
from ..patch import _conditioning_set_values


def _encode_qwen_with_reference(clip: Any, prompt: str, image: torch.Tensor):
    image_prefix = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    images = [image]
    try:
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE

        tokens = clip.tokenize(
            image_prefix + (prompt or ""),
            images=images,
            llama_template=KREA2_TEMPLATE,
        )
    except Exception:
        try:
            tokens = clip.tokenize(image_prefix + (prompt or ""), images=images)
        except TypeError:
            tokens = clip.tokenize(prompt or "")
    return clip.encode_from_tokens_scheduled(tokens)


def _encode_negative(clip: Any):
    try:
        tokens = clip.tokenize("")
        return clip.encode_from_tokens_scheduled(tokens)
    except Exception:
        return []


def _ensure_image_latent_5d(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 4:
        return latent.unsqueeze(2)
    if latent.ndim == 5:
        return latent
    raise ValueError(
        f"[Krea2 CcC Paint] Expected image latent B,C,H,W or B,C,T,H,W, got {tuple(latent.shape)}."
    )


def _token_aligned_soft_noise_mask(
    generated_mask: torch.Tensor,
    latent_h: int,
    latent_w: int,
    *,
    patch_size: int = 2,
) -> torch.Tensor:
    """Resize a soft generation mask and keep every Krea token spatially coherent."""
    mask = generated_mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(
            f"[Krea2 CcC Paint] generated_mask must be [B,H,W], got {tuple(mask.shape)}."
        )
    mask = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)

    if latent_h % patch_size or latent_w % patch_size:
        return mask.unsqueeze(1)

    blocks = mask.view(
        mask.shape[0],
        latent_h // patch_size,
        patch_size,
        latent_w // patch_size,
        patch_size,
    )
    block_values = blocks.mean(dim=(2, 4))
    aligned = (
        block_values.repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
    )
    return aligned.unsqueeze(1)


class CcCKrea2Paint:
    """Encode Krea2 AnyPaint-style conditioning and install the Paint runtime."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "paint_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 CcC Paint consumes Krea2 CcC Paint Prepare output. It image-grounds Qwen, "
        "VAE-encodes the semantic reference, preserves known pixels through a soft noise_mask, "
        "and registers the reference over the complete target grid at t=0 with isolated K/V."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "paint_context": ("KREA2_PAINT_CONTEXT",),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "empty background continuing naturally, no people, coherent perspective and lighting",
                    },
                ),
                "apply_krea2_paint_patch": ("BOOLEAN", {"default": True}),
                "kv_cache": ("BOOLEAN", {"default": True}),
            }
        }

    def process(
        self,
        model,
        clip,
        vae,
        paint_context: Dict[str, Any],
        positive_prompt="",
        apply_krea2_paint_patch=True,
        kv_cache=True,
    ):
        if not isinstance(paint_context, dict):
            raise ValueError("[Krea2 CcC Paint] paint_context is invalid.")

        required = ("known_image", "semantic_reference", "generated_mask", "keep_mask")
        missing = [key for key in required if key not in paint_context]
        if missing:
            raise ValueError(
                f"[Krea2 CcC Paint] paint_context is missing: {', '.join(missing)}."
            )

        known_image = paint_context["known_image"][..., :3].clamp(0.0, 1.0)
        semantic_reference = paint_context["semantic_reference"][..., :3].clamp(0.0, 1.0)
        generated_mask = paint_context["generated_mask"].clamp(0.0, 1.0)

        positive = _encode_qwen_with_reference(clip, positive_prompt, semantic_reference)
        negative = _encode_negative(clip)

        reference_latent = vae.encode(semantic_reference)
        positive = _conditioning_set_values(
            positive,
            {"reference_latents": [reference_latent]},
        )

        if negative:
            negative = _conditioning_set_values(
                negative,
                {"reference_latents": [reference_latent]},
            )

        known_latent = vae.encode(known_image)
        latent_tensor = _ensure_image_latent_5d(known_latent)
        latent_h, latent_w = latent_tensor.shape[-2:]
        noise_mask = _token_aligned_soft_noise_mask(
            generated_mask,
            latent_h,
            latent_w,
            patch_size=2,
        ).to(device=known_latent.device, dtype=known_latent.dtype)

        latent = {
            "samples": known_latent,
            "noise_mask": noise_mask,
        }

        if apply_krea2_paint_patch:
            patched_model = patch_krea2_paint_model(model, kv_cache=bool(kv_cache))
            runtime = "registered t=0 reference K/V"
        else:
            patched_model = model
            runtime = "patch disabled"

        info = "\n".join(
            [
                "=== Krea2 CcC Paint Report ===",
                (
                    f"Canvas: {paint_context.get('canvas_width')}x"
                    f"{paint_context.get('canvas_height')}"
                ),
                (
                    f"Semantic Reference: {semantic_reference.shape[2]}x"
                    f"{semantic_reference.shape[1]}"
                ),
                "Qwen: Picture 1 grounded reference",
                "Appearance Reference: semantic reference VAE latent",
                "Reference Placement: registered over complete target grid",
                f"Runtime: {runtime}",
                f"K/V Cache: {'enabled' if kv_cache else 'disabled'}",
                "Known-region preservation: soft token-aligned ComfyUI noise_mask",
                (
                    f"Mask: grow={paint_context.get('mask_grow')}, "
                    f"blur={paint_context.get('mask_blur_mode')} "
                    f"{paint_context.get('mask_blur_amount')}, "
                    f"direction={paint_context.get('mask_blur_direction')}"
                ),
                "Recommended sampling: Krea2 Turbo / AnyPaint LoRA, 8 steps, CFG 1, Euler, simple.",
            ]
        )
        return patched_model, positive, negative, latent, info
