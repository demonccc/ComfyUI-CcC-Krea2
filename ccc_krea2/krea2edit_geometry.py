"""Krea2Edit pixel-space geometry processing, cropped/fitted VAE reference calculations, and mask alignment.

Ported and attributed from ComfyUI-Krea2Edit by lbouaraba (GPL-3.0 / MIT).
https://github.com/lbouaraba/comfyui-krea2edit
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from .geometry import resize_tensor


@dataclass
class ResolvedGeometry:
    mode_requested: str
    mode_resolved: str
    source_size: Tuple[int, int]  # (W, H)
    crop_rectangle: Tuple[int, int, int, int]  # (left, top, crop_w, crop_h)
    vae_input_pixel_size: Tuple[int, int]  # (W, H)
    vae_latent_grid_size: Tuple[int, int]  # (lat_w, lat_h)
    target_grid_size: Tuple[int, int]  # (tgt_lat_w, tgt_lat_h)
    centered_fractional_offset: Tuple[float, float]  # (offset_y, offset_x)
    interpolation_method: str
    whether_interpolation_occurred: bool


CROP_TOL = 0.08  # Krea2Edit upstream near-matched aspect ratio tolerance


def resolve_krea2edit_geometry(
    src_h: int, src_w: int, tgt_h: int, tgt_w: int, fit_mode: str = "auto"
) -> ResolvedGeometry:
    """Resolve Krea2Edit pixel-space geometry and RoPE offsets matching upstream _fit_encode_image logic."""
    tgt_lat_h = tgt_h // 8
    tgt_lat_w = tgt_w // 8

    src_ar = src_w / float(src_h)
    tgt_ar = tgt_w / float(tgt_h)

    # Handle legacy fit mode aliases
    if fit_mode == "exact":
        fit_mode = "auto"
    elif fit_mode == "stretch":
        fit_mode = "crop"

    # Step 1: Resolve mode
    if fit_mode == "auto":
        if (src_w, src_h) == (tgt_w, tgt_h):
            resolved_mode = "exact"
        elif src_w >= tgt_w and src_h >= tgt_h:
            # Custom crop_only optimization applies strictly when Visual Reference Fit = auto
            crop_w = tgt_w
            crop_h = tgt_h
            dw_pct = (src_w - crop_w) / float(src_w)
            dh_pct = (src_h - crop_h) / float(src_h)
            area_discarded = (src_w * src_h - crop_w * crop_h) / float(src_w * src_h)

            if dw_pct <= 0.05 and dh_pct <= 0.05 and area_discarded <= 0.10:
                resolved_mode = "crop_only"
            else:
                sc = min(tgt_h / float(src_h), tgt_w / float(src_w))
                coverage_h = (src_h * sc) / float(tgt_h)
                coverage_w = (src_w * sc) / float(tgt_w)
                if coverage_h >= 0.92 and coverage_w >= 0.92:
                    resolved_mode = "crop_and_resize"
                else:
                    resolved_mode = "fit"
        else:
            sc = min(tgt_h / float(src_h), tgt_w / float(src_w))
            coverage_h = (src_h * sc) / float(tgt_h)
            coverage_w = (src_w * sc) / float(tgt_w)
            if coverage_h >= 0.92 and coverage_w >= 0.92:
                resolved_mode = "crop_and_resize"
            else:
                resolved_mode = "fit"
    elif fit_mode == "fit":
        # Section 6.2: Manual "fit" mode uses upstream resolver directly (does NOT run custom crop_only optimization)
        if (src_w, src_h) == (tgt_w, tgt_h):
            resolved_mode = "exact"
        else:
            sc = min(tgt_h / float(src_h), tgt_w / float(src_w))
            coverage_h = (src_h * sc) / float(tgt_h)
            coverage_w = (src_w * sc) / float(tgt_w)
            if coverage_h >= 0.92 and coverage_w >= 0.92:
                resolved_mode = "crop_and_resize"
            else:
                resolved_mode = "fit"
    elif fit_mode == "crop":
        resolved_mode = "crop"
    elif fit_mode == "contain":
        resolved_mode = "contain"
    elif fit_mode in ("contain_no_upscale", "fit_no_upscale"):
        resolved_mode = "contain_no_upscale"
    else:
        resolved_mode = fit_mode

    # Step 2: Calculate crop rectangle and VAE input dimensions according to resolved mode
    if resolved_mode in ("exact", "crop_only"):
        crop_w = tgt_w
        crop_h = tgt_h
        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        vae_input_w = tgt_w
        vae_input_h = tgt_h
        interp_occurred = False
        interp_method = "none"

    elif resolved_mode == "contain":
        # Contain with upscale allowed: preserve source AR and full image, scale to fit inside target bounds without crop
        sc = min(tgt_h / float(src_h), tgt_w / float(src_w))
        target_cap_h = max(16, (tgt_h // 16) * 16)
        target_cap_w = max(16, (tgt_w // 16) * 16)
        fitted_h = min(max(16, (int(round(src_h * sc)) // 16) * 16), target_cap_h)
        fitted_w = min(max(16, (int(round(src_w * sc)) // 16) * 16), target_cap_w)

        crop_h = src_h
        crop_w = src_w
        left = 0
        top = 0
        vae_input_w = fitted_w
        vae_input_h = fitted_h
        interp_occurred = (crop_w, crop_h) != (fitted_w, fitted_h)
        interp_method = "bicubic"

    elif resolved_mode in ("contain_no_upscale", "fit_no_upscale"):
        # Contain without upscale: preserve source AR and full image, scale down only if exceeding target bounds
        sc = min(1.0, tgt_h / float(src_h), tgt_w / float(src_w))
        target_cap_h = max(16, (tgt_h // 16) * 16)
        target_cap_w = max(16, (tgt_w // 16) * 16)
        fitted_h = min(max(16, (int(round(src_h * sc)) // 16) * 16), target_cap_h)
        fitted_w = min(max(16, (int(round(src_w * sc)) // 16) * 16), target_cap_w)

        crop_h = src_h
        crop_w = src_w
        left = 0
        top = 0
        vae_input_w = fitted_w
        vae_input_h = fitted_h
        interp_occurred = (crop_w, crop_h) != (fitted_w, fitted_h)
        interp_method = "bicubic"

    elif resolved_mode == "crop":
        # Manual Visual Reference Fit = crop: center crop to exact target aspect ratio & resize to exact target pixel dimensions
        if src_ar > tgt_ar:
            crop_h = src_h
            crop_w = int(round(src_h * tgt_ar))
        else:
            crop_w = src_w
            crop_h = int(round(src_w / tgt_ar))

        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        vae_input_w = tgt_w
        vae_input_h = tgt_h
        interp_occurred = (crop_w, crop_h) != (tgt_w, tgt_h)
        interp_method = "bicubic"

    elif resolved_mode == "crop_and_resize":
        # Minimal center crop to match target aspect ratio, then resize to exact target dimensions
        if src_ar > tgt_ar:
            crop_h = src_h
            crop_w = int(round(src_h * tgt_ar))
        else:
            crop_w = src_w
            crop_h = int(round(src_w / tgt_ar))

        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        vae_input_w = tgt_w
        vae_input_h = tgt_h
        interp_occurred = (crop_w, crop_h) != (vae_input_w, vae_input_h)
        interp_method = "bicubic"

    else:  # "fit" - Genuine aspect-ratio mismatch
        sc = min(tgt_h / float(src_h), tgt_w / float(src_w))
        # Cap against target dimensions floored to /16
        target_cap_h = max(16, (tgt_h // 16) * 16)
        target_cap_w = max(16, (tgt_w // 16) * 16)
        fitted_h = min(max(16, (int(src_h * sc) // 16) * 16), target_cap_h)
        fitted_w = min(max(16, (int(src_w * sc) // 16) * 16), target_cap_w)

        crop_h = min(src_h, max(1, int(round(fitted_h / sc))))
        crop_w = min(src_w, max(1, int(round(fitted_w / sc))))

        left = (src_w - crop_w) // 2
        top = (src_h - crop_h) // 2
        vae_input_w = fitted_w
        vae_input_h = fitted_h
        interp_occurred = (crop_w, crop_h) != (fitted_w, fitted_h)
        interp_method = "bicubic"

    vae_lat_w = vae_input_w // 8
    vae_lat_h = vae_input_h // 8

    # Calculate centered fractional RoPE offset
    offset_y = (tgt_lat_h - vae_lat_h) / 2.0
    offset_x = (tgt_lat_w - vae_lat_w) / 2.0

    return ResolvedGeometry(
        mode_requested=fit_mode,
        mode_resolved=resolved_mode,
        source_size=(src_w, src_h),
        crop_rectangle=(left, top, crop_w, crop_h),
        vae_input_pixel_size=(vae_input_w, vae_input_h),
        vae_latent_grid_size=(vae_lat_w, vae_lat_h),
        target_grid_size=(tgt_lat_w, tgt_lat_h),
        centered_fractional_offset=(offset_y, offset_x),
        interpolation_method=interp_method,
        whether_interpolation_occurred=interp_occurred,
    )


def process_image_and_mask_geometry(
    image: torch.Tensor, mask: Optional[torch.Tensor], geom: ResolvedGeometry
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply exact crop and resize geometry to both image tensor and attention mask tensor in lockstep."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    left, top, crop_w, crop_h = geom.crop_rectangle
    vae_w, vae_h = geom.vae_input_pixel_size

    # 1. Crop image [B, H, W, C]
    cropped_img = image[:, top : top + crop_h, left : left + crop_w, :]

    # 2. Resize image if required
    if (cropped_img.shape[1], cropped_img.shape[2]) != (vae_h, vae_w):
        processed_img = resize_tensor(cropped_img, target_h=vae_h, target_w=vae_w, method=geom.interpolation_method)
    else:
        processed_img = cropped_img

    processed_mask = None
    if mask is not None:
        # Handle mask format [B, H, W] or [H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Crop mask [B, H, W]
        cropped_mask = mask[:, top : top + crop_h, left : left + crop_w]

        # Resize mask using nearest-exact for binary integrity or bilinear if smooth
        if (cropped_mask.shape[1], cropped_mask.shape[2]) != (vae_h, vae_w):
            # Convert to [B, 1, H, W] for interpolation
            m_4d = cropped_mask.unsqueeze(1)
            resized_m = F.interpolate(m_4d, size=(vae_h, vae_w), mode="nearest-exact")
            processed_mask = resized_m.squeeze(1)
        else:
            processed_mask = cropped_mask

    return processed_img, processed_mask


def resolve_visual_reference_fit(
    image: torch.Tensor, target_h: int, target_w: int, mode: str = "auto", mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Backward compatibility wrapper around resolve_krea2edit_geometry and process_image_and_mask_geometry."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    src_h, src_w = image.shape[1], image.shape[2]
    geom = resolve_krea2edit_geometry(src_h=src_h, src_w=src_w, tgt_h=target_h, tgt_w=target_w, fit_mode=mode)
    fit_img, fit_mask = process_image_and_mask_geometry(image=image, mask=mask, geom=geom)

    fit_meta = {
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

    return fit_img, fit_mask, fit_meta
