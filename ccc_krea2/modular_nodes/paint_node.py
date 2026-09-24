"""Krea2 CcC Paint runtime node."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..constants import NODE_CATEGORY
from ..paint_patch import patch_krea2_paint_model
from ..patch import _conditioning_set_values


def _encode_qwen_with_reference(clip: Any, prompt: str, image: torch.Tensor):
    image_prefix = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
    images = [image]
    try:
        from comfy.text_encoders.krea2 import KREA2_TEMPLATE
        tokens = clip.tokenize(image_prefix + (prompt or ""), images=images, llama_template=KREA2_TEMPLATE)
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


class CcCKrea2Paint:
    """Prepare AnyPaint conditioning and install the Paint runtime."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "paint_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Consumes Krea2 CcC Paint Prepare context. Qwen is grounded with the neutralized semantic reference, "
        "the pre-encoded appearance reference is attached to conditioning, and the registered t=0 K/V runtime is installed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
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

    def process(self, model, clip, paint_context: Dict[str, Any], positive_prompt="", apply_krea2_paint_patch=True, kv_cache=True):
        if not isinstance(paint_context, dict):
            raise ValueError("[Krea2 CcC Paint] paint_context is invalid.")
        required = ("semantic_reference", "reference_latent", "generated_mask", "keep_mask")
        missing = [key for key in required if key not in paint_context]
        if missing:
            raise ValueError(f"[Krea2 CcC Paint] paint_context is missing: {', '.join(missing)}.")

        semantic_reference = paint_context["semantic_reference"][..., :3].clamp(0.0, 1.0)
        reference_latent = paint_context["reference_latent"]

        positive = _encode_qwen_with_reference(clip, positive_prompt, semantic_reference)
        negative = _encode_negative(clip)
        positive = _conditioning_set_values(positive, {"reference_latents": [reference_latent]})
        if negative:
            negative = _conditioning_set_values(negative, {"reference_latents": [reference_latent]})

        if apply_krea2_paint_patch:
            patched_model = patch_krea2_paint_model(model, kv_cache=bool(kv_cache))
            runtime = "registered t=0 reference K/V"
        else:
            patched_model = model
            runtime = "patch disabled"

        info = "\n".join([
            "=== Krea2 CcC Paint Report ===",
            f"Canvas: {paint_context.get('canvas_width')}x{paint_context.get('canvas_height')}",
            f"Geometry Mode: {paint_context.get('geometry_mode')}",
            f"Semantic Reference: {semantic_reference.shape[2]}x{semantic_reference.shape[1]}",
            "Qwen: Picture 1 grounded reference",
            "Appearance Reference: pre-encoded Paint Prepare latent",
            "Reference Placement: registered over complete target grid",
            f"Runtime: {runtime}",
            f"K/V Cache: {'enabled' if kv_cache else 'disabled'}",
            "Sampling latent/noise_mask: supplied by Krea2 CcC Paint Prepare",
            "Recommended sampling: Krea2 Turbo / AnyPaint LoRA, 8 steps, CFG 1, Euler, simple.",
        ])
        return patched_model, positive, negative, info
