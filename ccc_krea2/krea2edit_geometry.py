"""Pixel-space geometry for Krea2 CcC Visual Reference.

Public modes:
- crop: use the target grid as an inside crop window over the source.
- resize: map the reference long edge to the corresponding target edge, preserve
  aspect ratio, then center-crop only the minimum pixels required for /16.
- contain: scale the reference to fit inside the target, preserve aspect ratio,
  then center-crop only the minimum pixels required for /16.
- native: preserve 1:1 source scale and center-crop each edge down to /16.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .geometry import resize_tensor


@dataclass
class ResolvedGeometry:
    mode_requested: str
    mode_resolved: str
    source_size: Tuple[int, int]
    crop_rectangle: Tuple[int, int, int, int]
    vae_input_pixel_size: Tuple[int, int]
    vae_latent_grid_size: Tuple[int, int]
    target_grid_size: Tuple[int, int]
    centered_fractional_offset: Tuple[float, float]
    interpolation_method: str
    whether_interpolation_occurred: bool


def _floor16(value: float) -> int:
    return max(16, int(value) // 16 * 16)


def _positioned_crop_offset(available: int, position: str) -> int:
    if available <= 0:
        return 0
    if position in ("left", "up"):
        return 0
    if position in ("right", "down"):
        return available
    return available // 2


def _centered_crop_for_scaled_grid(
    src_h: int,
    src_w: int,
    scale: float,
    out_h: int,
    out_w: int,
) -> Tuple[int, int, int, int]:
    """Crop source pixels so one uniform scale lands exactly on a /16 output grid."""
    if scale <= 0.0:
        raise ValueError("Krea2 reference scale must be positive.")

    crop_h = min(src_h, max(1, int(round(out_h / scale))))
    crop_w = min(src_w, max(1, int(round(out_w / scale))))
    top = (src_h - crop_h) // 2
    left = (src_w - crop_w) // 2
    return left, top, crop_w, crop_h


def resolve_krea2edit_geometry(
    src_h: int,
    src_w: int,
    tgt_h: int,
    tgt_w: int,
    fit_mode: str = "native",
    grid_horizontal_position: Optional[str] = None,
    grid_vertical_position: Optional[str] = None,
    resize_method: str = "bicubic",
) -> ResolvedGeometry:
    """Resolve visual-reference geometry for the Krea2 CcC Edit pipeline."""
    requested = fit_mode

    # Backward-compatible aliases are internal only. Public nodes expose:
    # crop / resize / contain / native.
    if fit_mode == "fit":
        fit_mode = "contain"
    elif fit_mode == "exact":
        fit_mode = "native"
    elif fit_mode == "stretch":
        fit_mode = "crop"
    elif fit_mode == "auto":
        fit_mode = "contain"

    target_cap_h = _floor16(tgt_h)
    target_cap_w = _floor16(tgt_w)
    target_lat_h = tgt_h // 8
    target_lat_w = tgt_w // 8

    left = 0
    top = 0
    crop_w = src_w
    crop_h = src_h
    interpolation = resize_method

    if fit_mode == "crop":
        # The target grid is a window over the source. Crop never resizes.
        crop_w = _floor16(min(src_w, target_cap_w))
        crop_h = _floor16(min(src_h, target_cap_h))
        if crop_w > src_w or crop_h > src_h:
            raise ValueError("[Krea2 CcC Edit] crop reference is smaller than the minimum /16 VAE grid.")
        left = _positioned_crop_offset(src_w - crop_w, grid_horizontal_position or "center")
        top = _positioned_crop_offset(src_h - crop_h, grid_vertical_position or "center")
        vae_w = crop_w
        vae_h = crop_h
        interpolation = "none"
        resolved = "crop"

    elif fit_mode == "resize":
        # CcC resize: the long edge of the REFERENCE is mapped to the corresponding
        # axis of the TARGET. The other edge follows proportionally. A minimal
        # centered source crop makes that uniform scale land exactly on /16.
        if src_h >= src_w:
            scale = target_cap_h / float(src_h)
            vae_h = target_cap_h
            vae_w = _floor16(src_w * scale)
        else:
            scale = target_cap_w / float(src_w)
            vae_w = target_cap_w
            vae_h = _floor16(src_h * scale)

        left, top, crop_w, crop_h = _centered_crop_for_scaled_grid(
            src_h=src_h,
            src_w=src_w,
            scale=scale,
            out_h=vae_h,
            out_w=vae_w,
        )
        interpolation = "none" if (crop_w, crop_h) == (vae_w, vae_h) else resize_method
        resolved = "resize"

    elif fit_mode == "contain":
        # Fit inside the target. Unlike crop, contain never chooses a target-AR crop.
        # It only trims the few source pixels needed so one uniform scale lands on /16.
        scale = min(target_cap_h / float(src_h), target_cap_w / float(src_w))
        vae_h = min(_floor16(src_h * scale), target_cap_h)
        vae_w = min(_floor16(src_w * scale), target_cap_w)
        left, top, crop_w, crop_h = _centered_crop_for_scaled_grid(
            src_h=src_h,
            src_w=src_w,
            scale=scale,
            out_h=vae_h,
            out_w=vae_w,
        )
        interpolation = "none" if (crop_w, crop_h) == (vae_w, vae_h) else resize_method
        resolved = "contain"

    elif fit_mode in ("contain_no_upscale", "fit_no_upscale"):
        # Internal compatibility mode: contain, but do not enlarge a smaller source.
        scale = min(1.0, target_cap_h / float(src_h), target_cap_w / float(src_w))
        vae_h = min(_floor16(src_h * scale), target_cap_h)
        vae_w = min(_floor16(src_w * scale), target_cap_w)
        left, top, crop_w, crop_h = _centered_crop_for_scaled_grid(
            src_h=src_h,
            src_w=src_w,
            scale=scale,
            out_h=vae_h,
            out_w=vae_w,
        )
        interpolation = "none" if (crop_w, crop_h) == (vae_w, vae_h) else resize_method
        resolved = "contain_no_upscale"

    elif fit_mode == "native":
        # Native keeps a 1:1 pixel scale. Alignment is crop-down, never pad-up.
        if src_h < 16 or src_w < 16:
            raise ValueError("[Krea2 CcC Edit] native reference edges must be at least 16 pixels.")
        crop_w = _floor16(src_w)
        crop_h = _floor16(src_h)
        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        vae_w = crop_w
        vae_h = crop_h
        interpolation = "none"
        resolved = "native"

    else:
        raise ValueError(f"Unsupported Krea2 reference fit mode: {fit_mode}")

    vae_lat_w = vae_w // 8
    vae_lat_h = vae_h // 8
    offset_y = (target_lat_h - vae_lat_h) / 2.0
    offset_x = (target_lat_w - vae_lat_w) / 2.0

    return ResolvedGeometry(
        mode_requested=requested,
        mode_resolved=resolved,
        source_size=(src_w, src_h),
        crop_rectangle=(left, top, crop_w, crop_h),
        vae_input_pixel_size=(vae_w, vae_h),
        vae_latent_grid_size=(vae_lat_w, vae_lat_h),
        target_grid_size=(target_lat_w, target_lat_h),
        centered_fractional_offset=(offset_y, offset_x),
        interpolation_method=interpolation,
        whether_interpolation_occurred=(
            interpolation != "none" and (crop_w, crop_h) != (vae_w, vae_h)
        ),
    )


def process_image_and_mask_geometry(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    geom: ResolvedGeometry,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply resolved pixel geometry to image and optional mask in lockstep."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    left, top, crop_w, crop_h = geom.crop_rectangle
    vae_w, vae_h = geom.vae_input_pixel_size
    cropped = image[:, top:top + crop_h, left:left + crop_w, :]

    if (cropped.shape[1], cropped.shape[2]) != (vae_h, vae_w):
        processed = resize_tensor(
            cropped,
            target_h=vae_h,
            target_w=vae_w,
            method=geom.interpolation_method,
        )
    else:
        processed = cropped

    processed_mask = None
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        cropped_mask = mask[:, top:top + crop_h, left:left + crop_w]
        if (cropped_mask.shape[1], cropped_mask.shape[2]) != (vae_h, vae_w):
            processed_mask = F.interpolate(
                cropped_mask.unsqueeze(1),
                size=(vae_h, vae_w),
                mode="nearest-exact",
            ).squeeze(1)
        else:
            processed_mask = cropped_mask

    return processed, processed_mask


def resolve_visual_reference_fit(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "native",
    mask: Optional[torch.Tensor] = None,
    grid_horizontal_position: Optional[str] = None,
    grid_vertical_position: Optional[str] = None,
    resize_method: str = "bicubic",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    geom = resolve_krea2edit_geometry(
        src_h=image.shape[1],
        src_w=image.shape[2],
        tgt_h=target_h,
        tgt_w=target_w,
        fit_mode=mode,
        grid_horizontal_position=grid_horizontal_position,
        grid_vertical_position=grid_vertical_position,
        resize_method=resize_method,
    )
    fit_img, fit_mask = process_image_and_mask_geometry(image, mask, geom)
    return fit_img, fit_mask, {
        "mode_requested": geom.mode_requested,
        "mode_resolved": geom.mode_resolved,
        "source_size": geom.source_size,
        "crop_rectangle": geom.crop_rectangle,
        "spatial_hw": (geom.vae_input_pixel_size[1], geom.vae_input_pixel_size[0]),
        "vae_latent_grid_size": geom.vae_latent_grid_size,
        "target_grid_size": geom.target_grid_size,
        "centered_fractional_offset": geom.centered_fractional_offset,
        "interpolation_method": geom.interpolation_method,
        "whether_interpolation_occurred": geom.whether_interpolation_occurred,
        "geom": geom,
    }
