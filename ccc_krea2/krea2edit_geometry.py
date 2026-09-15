"""Krea2 Edit pixel-space geometry.

The `fit` path intentionally matches RedNode / Identity Edit v1.2: references are resampled in
pixel space to the resolved target-grid density before VAE encoding, and genuine aspect-ratio
mismatches keep the complete source image (no hidden crop).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .geometry import resize_tensor


CROP_TOL = 0.08


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


def _crop_to_target_ar(src_h: int, src_w: int, tgt_h: int, tgt_w: int) -> Tuple[int, int, int, int]:
    src_ar = src_w / float(src_h)
    tgt_ar = tgt_w / float(tgt_h)
    if src_ar > tgt_ar:
        crop_h = src_h
        crop_w = int(round(src_h * tgt_ar))
    else:
        crop_w = src_w
        crop_h = int(round(src_w / tgt_ar))
    return (src_w - crop_w) // 2, (src_h - crop_h) // 2, crop_w, crop_h


def resolve_krea2edit_geometry(
    src_h: int,
    src_w: int,
    tgt_h: int,
    tgt_w: int,
    fit_mode: str = "auto",
) -> ResolvedGeometry:
    """Resolve reference geometry while preserving the Identity Edit v1.2 fit contract."""
    requested = fit_mode
    if fit_mode == "exact":
        fit_mode = "auto"
    elif fit_mode == "stretch":
        fit_mode = "crop"

    target_cap_h = _floor16(tgt_h)
    target_cap_w = _floor16(tgt_w)
    target_lat_h = tgt_h // 8
    target_lat_w = tgt_w // 8

    scale = min(tgt_h / float(src_h), tgt_w / float(src_w))
    near_match = (
        src_h * scale >= tgt_h * (1.0 - CROP_TOL)
        and src_w * scale >= tgt_w * (1.0 - CROP_TOL)
    )

    if fit_mode == "auto":
        if (src_w, src_h) == (tgt_w, tgt_h):
            resolved = "exact"
        elif near_match:
            resolved = "crop_and_resize"
        else:
            resolved = "fit"
    elif fit_mode == "fit":
        resolved = "exact" if (src_w, src_h) == (tgt_w, tgt_h) else ("crop_and_resize" if near_match else "fit")
    elif fit_mode in ("crop", "contain", "native", "contain_no_upscale", "fit_no_upscale"):
        resolved = "contain_no_upscale" if fit_mode in ("contain_no_upscale", "fit_no_upscale") else fit_mode
    else:
        resolved = fit_mode

    left = 0
    top = 0
    crop_w = src_w
    crop_h = src_h
    interpolation = "bicubic"

    if resolved == "exact":
        vae_w = tgt_w
        vae_h = tgt_h
        interpolation = "none" if (src_w, src_h) == (vae_w, vae_h) else "bicubic"
    elif resolved in ("crop", "crop_and_resize"):
        left, top, crop_w, crop_h = _crop_to_target_ar(src_h, src_w, tgt_h, tgt_w)
        vae_w = tgt_w
        vae_h = tgt_h
    elif resolved == "fit":
        # Exact RedNode v1.2 mismatch behavior: preserve the COMPLETE source image and only
        # resample it to a /16-snapped grid inside the target. Do not back-compute a crop from
        # the snapped dimensions; that tiny crop was a CcC divergence from upstream.
        vae_h = min(_floor16(src_h * scale), target_cap_h)
        vae_w = min(_floor16(src_w * scale), target_cap_w)
    elif resolved == "contain":
        vae_h = min(_floor16(round(src_h * scale)), target_cap_h)
        vae_w = min(_floor16(round(src_w * scale)), target_cap_w)
    elif resolved == "contain_no_upscale":
        downscale = min(1.0, scale)
        if downscale >= 1.0:
            vae_h = min(max(16, math.ceil(src_h / 16) * 16), target_cap_h)
            vae_w = min(max(16, math.ceil(src_w / 16) * 16), target_cap_w)
            interpolation = "pad" if (vae_w, vae_h) != (src_w, src_h) else "none"
        else:
            vae_h = min(_floor16(round(src_h * downscale)), target_cap_h)
            vae_w = min(_floor16(round(src_w * downscale)), target_cap_w)
    elif resolved == "native":
        vae_h = max(16, math.ceil(src_h / 16) * 16)
        vae_w = max(16, math.ceil(src_w / 16) * 16)
        interpolation = "pad" if (vae_w, vae_h) != (src_w, src_h) else "none"
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
        whether_interpolation_occurred=(crop_w, crop_h) != (vae_w, vae_h),
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

    if geom.interpolation_method == "pad":
        pad_h = vae_h - cropped.shape[1]
        pad_w = vae_w - cropped.shape[2]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        nchw = cropped.permute(0, 3, 1, 2)
        processed = F.pad(nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate").permute(0, 2, 3, 1)
    elif (cropped.shape[1], cropped.shape[2]) != (vae_h, vae_w):
        processed = resize_tensor(cropped, target_h=vae_h, target_w=vae_w, method=geom.interpolation_method)
    else:
        processed = cropped

    processed_mask = None
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        cropped_mask = mask[:, top:top + crop_h, left:left + crop_w]
        if geom.interpolation_method == "pad":
            pad_h = vae_h - cropped_mask.shape[1]
            pad_w = vae_w - cropped_mask.shape[2]
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            processed_mask = F.pad(
                cropped_mask.unsqueeze(1),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            ).squeeze(1)
        elif (cropped_mask.shape[1], cropped_mask.shape[2]) != (vae_h, vae_w):
            processed_mask = F.interpolate(
                cropped_mask.unsqueeze(1), size=(vae_h, vae_w), mode="nearest-exact"
            ).squeeze(1)
        else:
            processed_mask = cropped_mask

    return processed, processed_mask


def resolve_visual_reference_fit(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "auto",
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    geom = resolve_krea2edit_geometry(
        src_h=image.shape[1],
        src_w=image.shape[2],
        tgt_h=target_h,
        tgt_w=target_w,
        fit_mode=mode,
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
