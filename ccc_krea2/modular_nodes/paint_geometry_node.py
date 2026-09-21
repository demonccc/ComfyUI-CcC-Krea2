"""Geometry preparation and exact restore for Krea2 CcC Paint."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from .latent_node import KREA_PRESET_GEOMETRIES


GEOMETRY_MODES = ("pad", "crop")
PADDING_FILLS = ("edge", "reflect", "neutral", "white")
HORIZONTAL_POSITIONS = ("center", "left", "right")
VERTICAL_POSITIONS = ("center", "top", "bottom")


def _normalize_mask(mask: Optional[torch.Tensor], *, batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.zeros((batch, height, width), device=device, dtype=torch.float32)
    value = mask
    while value.ndim > 3:
        value = value[:, 0]
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"[Krea2 CcC Paint Geometry] MASK must be [H,W] or [B,H,W], got {tuple(value.shape)}.")
    value = value.to(device=device, dtype=torch.float32)
    if value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, -1, -1)
    if value.shape[0] != batch:
        raise ValueError("[Krea2 CcC Paint Geometry] Mask batch must be 1 or match the image batch.")
    if value.shape[1:] != (height, width):
        import torch.nn.functional as F
        value = F.interpolate(value.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
    return value.clamp(0.0, 1.0)


def _axis_split(delta: int, position: str) -> Tuple[int, int]:
    if delta < 0:
        raise ValueError("Paint geometry delta must be non-negative.")
    if position in ("left", "top"):
        return 0, delta
    if position in ("right", "bottom"):
        return delta, 0
    before = delta // 2
    return before, delta - before


def _reflect_indices(length: int, before: int, after: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(-before, length + after, device=device)
    if length <= 1:
        return torch.zeros_like(coords)
    period = 2 * length - 2
    folded = torch.remainder(coords, period)
    return torch.where(folded < length, folded, period - folded).long()


def _neutral_rgb(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = (1.0 - mask).unsqueeze(-1).to(image.dtype)
    count = keep.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    known_mean = (image * keep).sum(dim=(1, 2), keepdim=True) / count
    fallback = image.mean(dim=(1, 2), keepdim=True)
    valid = (keep.sum(dim=(1, 2), keepdim=True) > 0).to(image.dtype)
    return known_mean * valid + fallback * (1.0 - valid)


def _pad_image(image: torch.Tensor, mask: torch.Tensor, *, left: int, top: int, right: int, bottom: int, fill: str) -> torch.Tensor:
    if not any((left, top, right, bottom)):
        return image
    batch, height, width, channels = image.shape
    if fill == "edge":
        ys = torch.arange(-top, height + bottom, device=image.device).clamp(0, height - 1).long()
        xs = torch.arange(-left, width + right, device=image.device).clamp(0, width - 1).long()
        return image[:, ys][:, :, xs, :]
    if fill == "reflect":
        ys = _reflect_indices(height, top, bottom, image.device)
        xs = _reflect_indices(width, left, right, image.device)
        return image[:, ys][:, :, xs, :]
    target_h = height + top + bottom
    target_w = width + left + right
    if fill == "neutral":
        rgb = _neutral_rgb(image, mask)
        canvas = rgb.expand(batch, target_h, target_w, channels).clone()
    elif fill == "white":
        canvas = torch.ones((batch, target_h, target_w, channels), device=image.device, dtype=image.dtype)
    else:
        raise ValueError(f"[Krea2 CcC Paint Geometry] Invalid padding_fill '{fill}'. Expected one of: {', '.join(PADDING_FILLS)}.")
    canvas[:, top:top + height, left:left + width, :] = image
    return canvas


def _pad_mask(
    mask: torch.Tensor,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    fill_value: float = 1.0,
) -> torch.Tensor:
    if not any((left, top, right, bottom)):
        return mask
    batch, height, width = mask.shape
    canvas = torch.full(
        (batch, height + top + bottom, width + left + right),
        float(fill_value),
        device=mask.device,
        dtype=mask.dtype,
    )
    canvas[:, top:top + height, left:left + width] = mask
    return canvas


def _select_krea_geometry(width: int, height: int, mode: str) -> Tuple[int, int]:
    source_ratio = width / float(height)
    candidates = []
    for candidate_w, candidate_h in sorted(set(KREA_PRESET_GEOMETRIES.values())):
        if mode == "pad" and candidate_w >= width and candidate_h >= height:
            candidates.append((candidate_w, candidate_h))
        elif mode == "crop" and candidate_w <= width and candidate_h <= height:
            candidates.append((candidate_w, candidate_h))
    if not candidates:
        if mode == "crop":
            raise ValueError("[Krea2 CcC Paint Geometry] crop is not possible: the working image is smaller than every compatible Krea target geometry. Use padding.")
        if mode == "pad":
            raise ValueError("[Krea2 CcC Paint Geometry] padding is not possible: the working image exceeds the available Krea target size on at least one axis. Use crop.")
        raise ValueError(f"[Krea2 CcC Paint Geometry] Invalid geometry mode '{mode}'.")

    def score(geometry: Tuple[int, int]) -> Tuple[float, float]:
        candidate_w, candidate_h = geometry
        ratio = candidate_w / float(candidate_h)
        ratio_error = abs((ratio / source_ratio) - 1.0)
        size_error = abs(candidate_w - width) / float(max(1, width)) + abs(candidate_h - height) / float(max(1, height))
        return ratio_error, size_error

    return min(candidates, key=score)


def prepare_paint_geometry(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    geometry_mode: str,
    padding_fill: str,
    horizontal_position: str,
    vertical_position: str,
    expand_left: int,
    expand_top: int,
    expand_right: int,
    expand_bottom: int,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("[Krea2 CcC Paint Geometry] IMAGE must be [B,H,W,C] with at least 3 channels.")
    if min(expand_left, expand_top, expand_right, expand_bottom) < 0:
        raise ValueError("[Krea2 CcC Paint Geometry] Canvas expansion cannot be negative.")
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f"[Krea2 CcC Paint Geometry] Invalid geometry_mode '{geometry_mode}'. Expected one of: {', '.join(GEOMETRY_MODES)}.")
    if padding_fill not in PADDING_FILLS:
        raise ValueError(f"[Krea2 CcC Paint Geometry] Invalid padding_fill '{padding_fill}'.")
    if horizontal_position not in HORIZONTAL_POSITIONS:
        raise ValueError(f"[Krea2 CcC Paint Geometry] Invalid horizontal_position '{horizontal_position}'.")
    if vertical_position not in VERTICAL_POSITIONS:
        raise ValueError(f"[Krea2 CcC Paint Geometry] Invalid vertical_position '{vertical_position}'.")

    source = image[..., :3].float().clamp(0.0, 1.0)
    batch, source_h, source_w, _ = source.shape
    source_mask = _normalize_mask(mask, batch=batch, height=source_h, width=source_w, device=source.device)

    # Explicit expansion belongs to the requested/final canvas, not to the temporary Krea normalization.
    # Keep its mask separate from the user-provided mask so Paint Prepare can preserve
    # the padding-fill color/lighting as semantic context while still generating the area.
    expansion = {
        "left": int(expand_left),
        "top": int(expand_top),
        "right": int(expand_right),
        "bottom": int(expand_bottom),
    }
    base_image = _pad_image(
        source,
        source_mask,
        left=expansion["left"],
        top=expansion["top"],
        right=expansion["right"],
        bottom=expansion["bottom"],
        fill=padding_fill,
    )
    base_user_mask = _pad_mask(
        source_mask,
        left=expansion["left"],
        top=expansion["top"],
        right=expansion["right"],
        bottom=expansion["bottom"],
        fill_value=0.0,
    )
    base_expansion_mask = _pad_mask(
        torch.zeros_like(source_mask),
        left=expansion["left"],
        top=expansion["top"],
        right=expansion["right"],
        bottom=expansion["bottom"],
        fill_value=1.0,
    )
    base_mask = torch.maximum(base_user_mask, base_expansion_mask)
    base_h, base_w = base_image.shape[1:3]

    target_w, target_h = _select_krea_geometry(base_w, base_h, geometry_mode)

    if geometry_mode == "pad":
        pad_left, pad_right = _axis_split(target_w - base_w, horizontal_position)
        pad_top, pad_bottom = _axis_split(target_h - base_h, vertical_position)
        working_image = _pad_image(
            base_image,
            base_mask,
            left=pad_left,
            top=pad_top,
            right=pad_right,
            bottom=pad_bottom,
            fill=padding_fill,
        )
        working_user_mask = _pad_mask(
            base_user_mask,
            left=pad_left,
            top=pad_top,
            right=pad_right,
            bottom=pad_bottom,
            fill_value=0.0,
        )
        working_expansion_mask = _pad_mask(
            base_expansion_mask,
            left=pad_left,
            top=pad_top,
            right=pad_right,
            bottom=pad_bottom,
            fill_value=0.0,
        )
        working_krea_padding_mask = _pad_mask(
            torch.zeros_like(base_mask),
            left=pad_left,
            top=pad_top,
            right=pad_right,
            bottom=pad_bottom,
            fill_value=1.0,
        )
        working_mask = torch.maximum(
            torch.maximum(working_user_mask, working_expansion_mask),
            working_krea_padding_mask,
        )
        transform = {
            "pad_left": int(pad_left),
            "pad_top": int(pad_top),
            "pad_right": int(pad_right),
            "pad_bottom": int(pad_bottom),
        }
    else:
        crop_left, _ = _axis_split(base_w - target_w, horizontal_position)
        crop_top, _ = _axis_split(base_h - target_h, vertical_position)
        y_slice = slice(crop_top, crop_top + target_h)
        x_slice = slice(crop_left, crop_left + target_w)
        working_image = base_image[:, y_slice, x_slice, :]
        working_user_mask = base_user_mask[:, y_slice, x_slice]
        working_expansion_mask = base_expansion_mask[:, y_slice, x_slice]
        working_krea_padding_mask = torch.zeros_like(working_user_mask)
        working_mask = torch.maximum(working_user_mask, working_expansion_mask)
        transform = {
            "crop_x": int(crop_left),
            "crop_y": int(crop_top),
            "crop_width": int(target_w),
            "crop_height": int(target_h),
        }

    context: Dict[str, Any] = {
        "mode": geometry_mode,
        "padding_fill": padding_fill,
        "horizontal_position": horizontal_position,
        "vertical_position": vertical_position,
        "source_width": int(source_w),
        "source_height": int(source_h),
        "base_width": int(base_w),
        "base_height": int(base_h),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "source_offset_x": int(expand_left),
        "source_offset_y": int(expand_top),
        "expand_left": int(expand_left),
        "expand_top": int(expand_top),
        "expand_right": int(expand_right),
        "expand_bottom": int(expand_bottom),
        "transform": transform,
        "base_image": base_image,
        "base_mask": base_mask,
        "base_user_mask": base_user_mask,
        "base_expansion_mask": base_expansion_mask,
        "working_image": working_image,
        "working_mask": working_mask,
        "working_user_mask": working_user_mask,
        "working_expansion_mask": working_expansion_mask,
        "working_krea_padding_mask": working_krea_padding_mask,
    }
    return context, working_image, working_mask


def restore_paint_geometry(image: torch.Tensor, paint_geometry: Dict[str, Any], generated_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(paint_geometry, dict):
        raise ValueError("[Krea2 CcC Paint Restore] paint_geometry is invalid.")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError("[Krea2 CcC Paint Restore] image must be [B,H,W,C].")

    mode = paint_geometry.get("mode")
    transform = dict(paint_geometry.get("transform") or {})
    base_image = paint_geometry.get("base_image")
    if not torch.is_tensor(base_image):
        raise ValueError("[Krea2 CcC Paint Restore] geometry context is missing base_image.")

    if mode == "pad":
        left = int(transform.get("pad_left", 0))
        top = int(transform.get("pad_top", 0))
        base_w = int(paint_geometry["base_width"])
        base_h = int(paint_geometry["base_height"])
        return image[:, top:top + base_h, left:left + base_w, :].clamp(0.0, 1.0)

    if mode == "crop":
        x = int(transform["crop_x"])
        y = int(transform["crop_y"])
        width = int(transform["crop_width"])
        height = int(transform["crop_height"])
        restored = base_image.to(device=image.device, dtype=image.dtype).clone()
        generated = image[:, :height, :width, :3].clamp(0.0, 1.0)
        if generated_mask is None:
            alpha = torch.ones((generated.shape[0], height, width, 1), device=generated.device, dtype=generated.dtype)
        else:
            alpha = generated_mask
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            if alpha.ndim != 3:
                raise ValueError("[Krea2 CcC Paint Restore] generated_mask must be [B,H,W].")
            if alpha.shape[1:] != (height, width):
                import torch.nn.functional as F
                alpha = F.interpolate(alpha.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
            alpha = alpha.to(device=generated.device, dtype=generated.dtype).clamp(0.0, 1.0).unsqueeze(-1)
        original_region = restored[:, y:y + height, x:x + width, :3]
        restored[:, y:y + height, x:x + width, :3] = original_region * (1.0 - alpha) + generated * alpha
        return restored.clamp(0.0, 1.0)

    raise ValueError(f"[Krea2 CcC Paint Restore] Unknown geometry mode '{mode}'.")


class CcCKrea2PaintGeometry:
    """Resolve Paint to a curated Krea geometry using only padding or cropping."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_PAINT_GEOMETRY", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("paint_geometry", "prepared_image", "prepared_mask", "geometry_info")
    FUNCTION = "prepare"
    DESCRIPTION = (
        "Maps the native paint canvas to a curated Krea target using pad or crop only. "
        "The source is never resized. Padding fill can use edge, reflect, neutral or white. "
        "The returned geometry context allows exact depadding or crop compositing later."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "geometry_mode": (GEOMETRY_MODES, {"default": "pad"}),
                "padding_fill": (PADDING_FILLS, {"default": "edge"}),
                "horizontal_position": (HORIZONTAL_POSITIONS, {"default": "center"}),
                "vertical_position": (VERTICAL_POSITIONS, {"default": "center"}),
                "expand_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "expand_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "expand_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "expand_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
            },
            "optional": {"mask": ("MASK",)},
        }

    def prepare(self, image, geometry_mode="pad", padding_fill="edge", horizontal_position="center", vertical_position="center", expand_left=0, expand_top=0, expand_right=0, expand_bottom=0, mask=None):
        context, prepared_image, prepared_mask = prepare_paint_geometry(
            image=image, mask=mask, geometry_mode=geometry_mode, padding_fill=padding_fill,
            horizontal_position=horizontal_position, vertical_position=vertical_position,
            expand_left=expand_left, expand_top=expand_top, expand_right=expand_right, expand_bottom=expand_bottom,
        )
        info = "\n".join([
            "=== Krea2 CcC Paint Geometry ===",
            f"Mode: {geometry_mode}",
            f"Source: {context['source_width']} x {context['source_height']}",
            f"Requested Canvas: {context['base_width']} x {context['base_height']}",
            f"Krea Working Geometry: {context['target_width']} x {context['target_height']}",
            f"Position: {horizontal_position} / {vertical_position}",
            f"Padding Fill: {padding_fill if geometry_mode == 'pad' or any((expand_left, expand_top, expand_right, expand_bottom)) else '<unused>'}",
            f"Transform: {context['transform']}",
        ])
        return context, prepared_image, prepared_mask, info


class CcCKrea2PaintRestore:
    """Restore a Paint result from temporary Krea pad/crop geometry."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    DESCRIPTION = (
        "Restores Paint output to the requested native canvas. Pad mode removes only temporary "
        "Krea padding. Crop mode composites the generated crop back into the preserved native canvas."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "paint_geometry": ("KREA2_PAINT_GEOMETRY",)},
            "optional": {"generated_mask": ("MASK",)},
        }

    def restore(self, image, paint_geometry, generated_mask=None):
        return (restore_paint_geometry(image, paint_geometry, generated_mask),)
