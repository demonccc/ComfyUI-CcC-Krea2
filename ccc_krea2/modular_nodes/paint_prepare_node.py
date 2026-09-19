"""Canvas and mask preparation for Krea2 CcC Paint."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY


_ALIGNMENT = 16
_SEMANTIC_REFERENCE_MAX_EDGE = 384


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return max(alignment, int(math.ceil(int(value) / float(alignment)) * alignment))


def _normalize_mask(
    mask: Optional[torch.Tensor],
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.zeros((batch, height, width), device=device, dtype=torch.float32)
    value = mask
    while value.ndim > 3:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(
            f"[Krea2 CcC Paint Prepare] MASK must be [H,W] or [B,H,W], got {tuple(value.shape)}."
        )
    value = value.to(device=device, dtype=torch.float32)
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1, -1)
    if value.shape[0] != batch:
        raise ValueError(
            "[Krea2 CcC Paint Prepare] Mask batch must be 1 or match the image batch."
        )
    if value.shape[1:] != (height, width):
        value = F.interpolate(
            value.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return value.clamp(0.0, 1.0)


def _grow_or_shrink(mask: torch.Tensor, amount: int) -> torch.Tensor:
    radius = abs(int(amount))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    if amount > 0:
        return F.max_pool2d(mask.unsqueeze(1), kernel, stride=1, padding=radius).squeeze(1)
    inverse = 1.0 - mask
    grown_inverse = F.max_pool2d(
        inverse.unsqueeze(1), kernel, stride=1, padding=radius
    ).squeeze(1)
    return (1.0 - grown_inverse).clamp(0.0, 1.0)


def _box_blur(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius == 0:
        return mask
    kernel = radius * 2 + 1
    padded = F.pad(
        mask.unsqueeze(1),
        (radius, radius, radius, radius),
        mode="replicate",
    )
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1).squeeze(1)


def _gaussian_blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        return mask
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(
        -radius,
        radius + 1,
        device=mask.device,
        dtype=mask.dtype,
    )
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-8)

    value = F.pad(mask.unsqueeze(1), (radius, radius, 0, 0), mode="replicate")
    value = F.conv2d(value, kernel.view(1, 1, 1, -1))
    value = F.pad(value, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(value, kernel.view(1, 1, -1, 1)).squeeze(1)


def _apply_feather(
    hard_mask: torch.Tensor,
    *,
    mode: str,
    amount: float,
    direction: str,
) -> torch.Tensor:
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
        raise ValueError(
            f"[Krea2 CcC Paint Prepare] Unknown mask_blur_direction '{direction}'."
        )
    return result.clamp(0.0, 1.0)


def _neutral_fill(image: torch.Tensor, preserve_mask: torch.Tensor) -> torch.Tensor:
    """Return one neutral RGB value per batch from pixels that are still known."""
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


def prepare_paint_context(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    expand_left: int,
    expand_top: int,
    expand_right: int,
    expand_bottom: int,
    mask_grow: int,
    mask_blur_mode: str,
    mask_blur_amount: float,
    mask_blur_direction: str,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare an unscaled native canvas plus soft generation/preservation masks."""
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            "[Krea2 CcC Paint Prepare] IMAGE must be [B,H,W,C] with at least 3 channels."
        )
    if min(expand_left, expand_top, expand_right, expand_bottom) < 0:
        raise ValueError("[Krea2 CcC Paint Prepare] Canvas expansion cannot be negative.")

    source = image[..., :3].float().clamp(0.0, 1.0)
    batch, source_h, source_w, _ = source.shape
    source_mask = _normalize_mask(
        mask,
        batch=batch,
        height=source_h,
        width=source_w,
        device=source.device,
    )

    requested_w = source_w + int(expand_left) + int(expand_right)
    requested_h = source_h + int(expand_top) + int(expand_bottom)
    canvas_w = _align_up(requested_w)
    canvas_h = _align_up(requested_h)
    alignment_right = canvas_w - requested_w
    alignment_bottom = canvas_h - requested_h

    source_keep = 1.0 - source_mask
    fill = _neutral_fill(source, source_keep)
    known_image = fill.expand(batch, canvas_h, canvas_w, 3).clone()
    known_image[
        :,
        expand_top : expand_top + source_h,
        expand_left : expand_left + source_w,
        :,
    ] = source

    hard_generated = torch.ones(
        (batch, canvas_h, canvas_w),
        device=source.device,
        dtype=torch.float32,
    )
    hard_generated[
        :,
        expand_top : expand_top + source_h,
        expand_left : expand_left + source_w,
    ] = source_mask

    hard_generated = _grow_or_shrink(hard_generated, int(mask_grow)).clamp(0.0, 1.0)
    generated_mask = _apply_feather(
        hard_generated,
        mode=mask_blur_mode,
        amount=float(mask_blur_amount),
        direction=mask_blur_direction,
    )
    keep_mask = (1.0 - generated_mask).clamp(0.0, 1.0)

    semantic_fill = _neutral_fill(known_image, keep_mask)
    semantic_full = (
        known_image * keep_mask.unsqueeze(-1).to(known_image.dtype)
        + semantic_fill * generated_mask.unsqueeze(-1).to(known_image.dtype)
    )
    semantic_reference = _downscale_semantic_reference(semantic_full)

    context: Dict[str, Any] = {
        "known_image": known_image,
        "semantic_reference": semantic_reference,
        "generated_mask": generated_mask,
        "keep_mask": keep_mask,
        "canvas_width": int(canvas_w),
        "canvas_height": int(canvas_h),
        "source_width": int(source_w),
        "source_height": int(source_h),
        "source_offset_x": int(expand_left),
        "source_offset_y": int(expand_top),
        "alignment_right": int(alignment_right),
        "alignment_bottom": int(alignment_bottom),
        "mask_grow": int(mask_grow),
        "mask_blur_mode": str(mask_blur_mode),
        "mask_blur_amount": float(mask_blur_amount),
        "mask_blur_direction": str(mask_blur_direction),
    }
    return context, known_image, semantic_reference, generated_mask, keep_mask


class CcCKrea2PaintPrepare:
    """Prepare a native-resolution canvas and mask contract for Krea2 CcC Paint."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = (
        "KREA2_PAINT_CONTEXT",
        "IMAGE",
        "IMAGE",
        "MASK",
        "MASK",
        "STRING",
    )
    RETURN_NAMES = (
        "paint_context",
        "prepared_image",
        "semantic_reference",
        "generated_mask",
        "keep_mask",
        "paint_prepare_info",
    )
    FUNCTION = "prepare"
    DESCRIPTION = (
        "Builds the Krea2 CcC Paint canvas without resizing the source. "
        "The source mask and outpaint padding are combined, then signed grow/shrink "
        "and directional feathering are applied before semantic neutralization."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "mask_grow": ("INT", {"default": 0, "min": -256, "max": 256, "step": 1}),
                "mask_blur_mode": (["standard", "gaussian_sigma"], {"default": "gaussian_sigma"}),
                "mask_blur_amount": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5},
                ),
                "mask_blur_direction": (
                    ["outside", "inside", "both"],
                    {"default": "outside"},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def prepare(
        self,
        image,
        expand_left=0,
        expand_top=0,
        expand_right=0,
        expand_bottom=0,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
        mask=None,
    ):
        result = prepare_paint_context(
            image=image,
            mask=mask,
            expand_left=expand_left,
            expand_top=expand_top,
            expand_right=expand_right,
            expand_bottom=expand_bottom,
            mask_grow=mask_grow,
            mask_blur_mode=mask_blur_mode,
            mask_blur_amount=mask_blur_amount,
            mask_blur_direction=mask_blur_direction,
        )
        context, _, semantic_reference, generated_mask, _ = result
        info = "\n".join(
            [
                "=== Krea2 CcC Paint Prepare ===",
                (
                    f"Canvas: {context['canvas_width']}x{context['canvas_height']} "
                    f"(source {context['source_width']}x{context['source_height']})"
                ),
                (
                    "Expansion: "
                    f"L{expand_left} T{expand_top} R{expand_right} B{expand_bottom} "
                    f"+ align R{context['alignment_right']} B{context['alignment_bottom']}"
                ),
                f"Mask Grow: {mask_grow}px",
                (
                    f"Mask Feather: {mask_blur_mode} amount={float(mask_blur_amount):.2f} "
                    f"direction={mask_blur_direction}"
                ),
                (
                    f"Semantic Reference: {semantic_reference.shape[2]}x"
                    f"{semantic_reference.shape[1]} max-edge {_SEMANTIC_REFERENCE_MAX_EDGE}px"
                ),
                (
                    f"Generated Mask Range: {float(generated_mask.min()):.3f}.."
                    f"{float(generated_mask.max()):.3f}"
                ),
            ]
        )
        return (*result, info)
