"""Mask, semantic-reference and latent preparation for Krea2 CcC Paint."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY


_ALIGNMENT = 16
_SEMANTIC_REFERENCE_MAX_EDGE = 384


def _fill_mask_holes(mask: torch.Tensor) -> torch.Tensor:
    """Fill enclosed background regions while preserving the existing soft mask boundary."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"[Krea2 CcC Paint Prepare] mask must be [B,H,W], got {tuple(mask.shape)}.")

    result = mask.clone()
    binary = (mask.detach().float().cpu() >= 0.5).to(torch.uint8).numpy()

    for batch_index in range(binary.shape[0]):
        # Pad with one guaranteed exterior-background border, then flood that exterior.
        image = Image.fromarray(binary[batch_index] * 255, mode="L")
        padded = ImageOps.expand(image, border=1, fill=0)
        ImageDraw.floodfill(padded, (0, 0), 128, border=255)

        flooded = np.asarray(padded, dtype=np.uint8)[1:-1, 1:-1]
        holes = flooded == 0
        if holes.any():
            hole_mask = torch.from_numpy(holes.copy()).to(device=result.device)
            result[batch_index][hole_mask] = 1.0

    return result.clamp(0.0, 1.0)


def _grow_or_shrink(mask: torch.Tensor, amount: int) -> torch.Tensor:
    radius = abs(int(amount))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    if amount > 0:
        return F.max_pool2d(mask.unsqueeze(1), kernel, stride=1, padding=radius).squeeze(1)
    inverse = 1.0 - mask
    grown_inverse = F.max_pool2d(inverse.unsqueeze(1), kernel, stride=1, padding=radius).squeeze(1)
    return (1.0 - grown_inverse).clamp(0.0, 1.0)


def _box_blur(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    padded = F.pad(mask.unsqueeze(1), (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1).squeeze(1)


def _gaussian_blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return mask
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    value = F.pad(mask.unsqueeze(1), (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, kernel.view(1, 1, 1, -1))
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(value, kernel.view(1, 1, -1, 1)).squeeze(1)


def _apply_feather(hard_mask: torch.Tensor, *, mode: str, amount: float, direction: str) -> torch.Tensor:
    amount = float(amount)
    if amount <= 0.0:
        return hard_mask
    if mode == "standard":
        blurred = _box_blur(hard_mask, max(1, int(round(amount))))
    elif mode == "gaussian_sigma":
        blurred = _gaussian_blur(hard_mask, amount)
    else:
        raise ValueError(f"[Krea2 CcC Paint Prepare] Unknown mask_blur_mode '{mode}'.")
    if direction == "outside":
        result = torch.maximum(hard_mask, blurred)
    elif direction == "inside":
        result = torch.minimum(hard_mask, blurred)
    elif direction == "both":
        result = blurred
    else:
        raise ValueError(f"[Krea2 CcC Paint Prepare] Unknown mask_blur_direction '{direction}'.")
    return result.clamp(0.0, 1.0)


def _neutral_fill(image: torch.Tensor, preserve_mask: torch.Tensor) -> torch.Tensor:
    weights = preserve_mask.unsqueeze(-1).to(image.dtype)
    count = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    mean = (image * weights).sum(dim=(1, 2), keepdim=True) / count
    fallback = image.mean(dim=(1, 2), keepdim=True)
    valid = (weights.sum(dim=(1, 2), keepdim=True) > 0).to(image.dtype)
    return mean * valid + fallback * (1.0 - valid)


def _downscale_semantic_reference(image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[1:3]
    longest = max(height, width)
    if longest <= _SEMANTIC_REFERENCE_MAX_EDGE:
        return image
    scale = _SEMANTIC_REFERENCE_MAX_EDGE / float(longest)
    target_h = max(_ALIGNMENT, int(round(height * scale / _ALIGNMENT)) * _ALIGNMENT)
    target_w = max(_ALIGNMENT, int(round(width * scale / _ALIGNMENT)) * _ALIGNMENT)
    value = F.interpolate(
        image.movedim(-1, 1).float(),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return value.movedim(1, -1).to(image.dtype).clamp(0.0, 1.0)


def _extract_latent(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value.get("samples")
    elif hasattr(value, "sample") and callable(value.sample):
        value = value.sample()
    if not torch.is_tensor(value):
        raise ValueError("[Krea2 CcC Paint Prepare] VAE encode did not return a latent tensor.")
    return value


def _ensure_image_latent_5d(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 4:
        return latent.unsqueeze(2)
    if latent.ndim == 5:
        return latent
    raise ValueError(f"[Krea2 CcC Paint Prepare] Expected B,C,H,W or B,C,T,H,W latent, got {tuple(latent.shape)}.")


def _token_aligned_soft_noise_mask(generated_mask: torch.Tensor, latent_h: int, latent_w: int, *, patch_size: int = 2) -> torch.Tensor:
    mask = generated_mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"[Krea2 CcC Paint Prepare] generated_mask must be [B,H,W], got {tuple(mask.shape)}.")
    mask = F.interpolate(mask.unsqueeze(1).float(), size=(latent_h, latent_w), mode="bilinear", align_corners=False).squeeze(1).clamp(0.0, 1.0)
    if latent_h % patch_size or latent_w % patch_size:
        return mask.unsqueeze(1)
    blocks = mask.view(mask.shape[0], latent_h // patch_size, patch_size, latent_w // patch_size, patch_size)
    block_values = blocks.mean(dim=(2, 4))
    aligned = block_values.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return aligned.unsqueeze(1)


def prepare_paint_context(
    vae: Any,
    paint_geometry: Dict[str, Any],
    *,
    fill_holes: bool = False,
    mask_grow: int,
    mask_blur_mode: str,
    mask_blur_amount: float,
    mask_blur_direction: str,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(paint_geometry, dict):
        raise ValueError("[Krea2 CcC Paint Prepare] paint_geometry is invalid.")
    known_image = paint_geometry.get("working_image")
    hard_mask = paint_geometry.get("working_mask")
    if not torch.is_tensor(known_image) or not torch.is_tensor(hard_mask):
        raise ValueError("[Krea2 CcC Paint Prepare] paint_geometry is missing working image/mask tensors.")

    known_image = known_image[..., :3].float().clamp(0.0, 1.0)
    hard_mask = hard_mask.float().clamp(0.0, 1.0)

    expansion_mask = paint_geometry.get("working_expansion_mask")
    if expansion_mask is None:
        expansion_mask = torch.zeros_like(hard_mask)
    elif not torch.is_tensor(expansion_mask):
        raise ValueError("[Krea2 CcC Paint Prepare] working_expansion_mask must be a MASK tensor.")
    else:
        expansion_mask = expansion_mask.to(device=hard_mask.device, dtype=hard_mask.dtype).clamp(0.0, 1.0)
        if expansion_mask.shape != hard_mask.shape:
            raise ValueError(
                "[Krea2 CcC Paint Prepare] working_expansion_mask must match working_mask geometry."
            )
    if fill_holes:
        hard_mask = _fill_mask_holes(hard_mask)
    hard_generated = _grow_or_shrink(hard_mask, int(mask_grow)).clamp(0.0, 1.0)
    generated_mask = _apply_feather(
        hard_generated,
        mode=mask_blur_mode,
        amount=float(mask_blur_amount),
        direction=mask_blur_direction,
    )
    keep_mask = (1.0 - generated_mask).clamp(0.0, 1.0)

    # Explicit outpaint expansion is still fully generable, but its Geometry padding
    # (edge/reflect/neutral/white) remains visible to the semantic/appearance reference.
    # This carries border color and lighting continuity into AnyPaint instead of replacing
    # the whole new margin with one neutral tone. Manual masks, temporary Krea padding,
    # and any grow/feather that reaches into known pixels remain neutralized.
    semantic_neutralize_mask = (generated_mask - expansion_mask).clamp(0.0, 1.0)
    semantic_keep_mask = (1.0 - semantic_neutralize_mask).clamp(0.0, 1.0)
    semantic_fill = _neutral_fill(known_image, keep_mask)
    semantic_full = (
        known_image * semantic_keep_mask.unsqueeze(-1).to(known_image.dtype)
        + semantic_fill * semantic_neutralize_mask.unsqueeze(-1).to(known_image.dtype)
    ).clamp(0.0, 1.0)
    semantic_reference = _downscale_semantic_reference(semantic_full)

    reference_latent = _extract_latent(vae.encode(semantic_reference))
    known_latent = _extract_latent(vae.encode(known_image))
    latent_tensor = _ensure_image_latent_5d(known_latent)
    latent_h, latent_w = latent_tensor.shape[-2:]
    noise_mask = _token_aligned_soft_noise_mask(generated_mask, latent_h, latent_w, patch_size=2).to(device=known_latent.device, dtype=known_latent.dtype)

    latent = {"samples": known_latent, "noise_mask": noise_mask}
    context: Dict[str, Any] = {
        "semantic_reference": semantic_reference,
        "reference_latent": reference_latent,
        "generated_mask": generated_mask,
        "keep_mask": keep_mask,
        "expansion_context_mask": expansion_mask,
        "semantic_neutralize_mask": semantic_neutralize_mask,
        "canvas_width": int(known_image.shape[2]),
        "canvas_height": int(known_image.shape[1]),
        "geometry_mode": paint_geometry.get("mode"),
        "fill_holes": bool(fill_holes),
        "mask_grow": int(mask_grow),
        "mask_blur_mode": str(mask_blur_mode),
        "mask_blur_amount": float(mask_blur_amount),
        "mask_blur_direction": str(mask_blur_direction),
    }
    return context, latent, known_image, semantic_reference, generated_mask, keep_mask


class CcCKrea2PaintPrepare:
    """Build Paint masks, semantic reference and the sampling latent from prepared geometry."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_PAINT_CONTEXT", "LATENT", "IMAGE", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("paint_context", "latent", "prepared_image", "semantic_reference", "generated_mask", "keep_mask", "paint_prepare_info")
    FUNCTION = "prepare"
    DESCRIPTION = (
        "Consumes Krea2 CcC Paint Geometry, optionally fills enclosed mask holes, applies mask grow/feather, creates the semantic reference, "
        "VAE-encodes the known Krea canvas, and returns the sampling latent with token-aligned noise_mask."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "paint_geometry": ("KREA2_PAINT_GEOMETRY",),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "mask_blur_mode": (["standard", "gaussian_sigma"], {"default": "gaussian_sigma"}),
                "mask_blur_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5}),
                "mask_blur_direction": (["outside", "inside", "both"], {"default": "outside"}),
            }
        }

    def prepare(
        self,
        vae,
        paint_geometry,
        fill_holes=False,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    ):
        result = prepare_paint_context(
            vae=vae,
            paint_geometry=paint_geometry,
            fill_holes=fill_holes,
            mask_grow=mask_grow,
            mask_blur_mode=mask_blur_mode,
            mask_blur_amount=mask_blur_amount,
            mask_blur_direction=mask_blur_direction,
        )
        context, _, _, semantic_reference, generated_mask, _ = result
        info = "\n".join([
            "=== Krea2 CcC Paint Prepare ===",
            f"Canvas: {context['canvas_width']}x{context['canvas_height']}",
            f"Geometry Mode: {context['geometry_mode']}",
            f"Fill Holes: {'yes' if fill_holes else 'no'}",
            f"Mask Grow: {mask_grow}px",
            f"Mask Feather: {mask_blur_mode} amount={float(mask_blur_amount):.2f} direction={mask_blur_direction}",
            f"Expansion Context Pixels: {int((context['expansion_context_mask'] > 0.5).sum())}",
            f"Semantic Reference: {semantic_reference.shape[2]}x{semantic_reference.shape[1]} max-edge {_SEMANTIC_REFERENCE_MAX_EDGE}px",
            f"Generated Mask Range: {float(generated_mask.min()):.3f}..{float(generated_mask.max()):.3f}",
            "Latent: known-image samples + token-aligned noise_mask",
        ])
        return (*result, info)
