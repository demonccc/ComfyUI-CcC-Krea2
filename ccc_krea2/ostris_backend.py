"""Ostris Edit backend implementation for CcC Krea2 suite.

Based on Ostris's ai-toolkit edit implementation.
Original work Copyright (c) Ostris / ai-toolkit contributors.
Licensed under the MIT License.
"""

import math
from typing import List, Dict, Any, Optional
import torch

from .style_processing import get_style_processing_image_count
from .geometry import resize_tensor


# Ostris vision pixel budget (~384x384 = 147456 pixels)
OSTRIS_VISION_PIXEL_BUDGET = 384 * 384
# Ostris VAE latent max pixel budget (1024x1024 = 1048576 pixels)
OSTRIS_VAE_MAX_PIXELS = 1024 * 1024


def preprocess_ostris_vision_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess vision image for Ostris backend: area-constrained to ~384x384, never upscaled.

    Uses AREA downscaling. NO /16 snapping is performed on vision input images.
    """
    if image_tensor is None:
        return image_tensor

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    bs, h, w, c = image_tensor.shape
    curr_area = h * w

    if curr_area > OSTRIS_VISION_PIXEL_BUDGET:
        scale = math.sqrt(OSTRIS_VISION_PIXEL_BUDGET / float(curr_area))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = resize_tensor(image_tensor, target_h=new_h, target_w=new_w, method="area")
        return torch.clamp(resized, 0.0, 1.0)

    return image_tensor


def preprocess_ostris_ref_pixel_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess pixel image for Ostris VAE reference encoding matching upstream _fit_area: max 1MP (1024x1024), /16 snapped using AREA interpolation whenever dimensions change."""
    if image_tensor is None:
        return image_tensor

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    bs, h, w, c = image_tensor.shape
    curr_area = float(h * w)

    scale = min(1.0, math.sqrt(OSTRIS_VAE_MAX_PIXELS / curr_area))
    snapped_w = max(16, int(round((w * scale) / 16.0) * 16))
    snapped_h = max(16, int(round((h * scale) / 16.0) * 16))

    if (snapped_h, snapped_w) != (h, w):
        resized = resize_tensor(image_tensor, target_h=snapped_h, target_w=snapped_w, method="area")
        return torch.clamp(resized, 0.0, 1.0)

    return image_tensor


def build_ostris_qwen_prompt(
    resolved_references: List[Dict[str, Any]],
    user_prompt: str = "",
) -> str:
    """Format Qwen text prompt for Ostris backend using 'Picture N: <vision block>' layout.

    Picture N numbering corresponds ONLY to actual Ostris appearance edit references.
    Non-appearance and Style references render as standard vision blocks without 'Picture N:'.
    """
    lines = []
    picture_counter = 1
    for item in resolved_references:
        spec = item.get("spec")
        ref_path = getattr(spec, "reference_path", getattr(spec, "role", "").lower()) if spec else "edit"
        is_appearance = getattr(spec, "appearance_reference", True) if spec else True

        if ref_path == "style":
            style_proc = getattr(spec, "style_processing", "2x2") if spec else "2x2"
            num_crops = get_style_processing_image_count(style_proc)
            for _ in range(num_crops):
                lines.append("<|vision_start|><|image_pad|><|vision_end|>")
        elif not is_appearance:
            expanded_aliases = item.get("expanded_aliases", ())
            alias_str = ", ".join(expanded_aliases) if expanded_aliases else (getattr(spec, "alias", "") if spec else "")
            instruction = getattr(spec, "vision_instruction", "") if spec else ""
            annotation = ""
            if alias_str and instruction:
                annotation = f" ({alias_str}): {instruction}"
            elif alias_str:
                annotation = f" ({alias_str})"
            elif instruction:
                annotation = f": {instruction}"
            lines.append(f"<|vision_start|><|image_pad|><|vision_end|>{annotation}")
        else:
            expanded_aliases = item.get("expanded_aliases", ())
            alias_str = ", ".join(expanded_aliases) if expanded_aliases else (getattr(spec, "alias", "") if spec else "")
            instruction = getattr(spec, "vision_instruction", "") if spec else ""
            annotation = ""
            if alias_str and instruction:
                annotation = f" ({alias_str}): {instruction}"
            elif alias_str:
                annotation = f" ({alias_str})"
            elif instruction:
                annotation = f": {instruction}"

            lines.append(f"Picture {picture_counter}: <|vision_start|><|image_pad|><|vision_end|>{annotation}")
            picture_counter += 1

    body = "\n".join(lines)
    if body and user_prompt:
        return f"{body}\n{user_prompt}"
    return body or (user_prompt or "")


def patch_ostris_model(
    model: Any,
    prepared_refs: Optional[Any] = None,
    ostris_kv_cache: bool = False
) -> Any:
    """Deprecated compatibility shim.

    Canonical regular Ostris in CcC Krea2 uses conditioning metadata ('index_timestep_zero')
    and leaves the MODEL object unchanged.
    """
    if ostris_kv_cache:
        raise NotImplementedError(
            "[CcC Krea2] ostris_kv_cache=True is not supported in the CcC Ostris backend. "
            "Intended only for LoRAs trained with ai-toolkit kv_cache."
        )
    return model
