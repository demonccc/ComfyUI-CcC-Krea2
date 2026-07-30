"""Pixel-space geometric transformations matching native Krea 2 training-aligned geometry."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def apply_sampling_transform(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "fit",
    mask: Optional[torch.Tensor] = None,
    mask_interpolation: str = "bicubic"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Transform base image/mask for sampling LATENT output.

    Modes allowed:
    - fit: Preserve aspect ratio, scale inside, center-pad to target_h x target_w.
    - crop: Preserve aspect ratio, scale to cover, center-crop to target_h x target_w.
    - stretch: Directly resize to target_h x target_w without AR preservation.
    """
    if mode not in ("fit", "crop", "stretch"):
        raise ValueError(f"Invalid sampling_resize_mode '{mode}'. Expected 'fit', 'crop', or 'stretch'.")
    return _apply_sampling_transform_internal(image, target_h, target_w, mode, mask, mask_interpolation)


def apply_reference_fit_transform(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "fit",
    mask: Optional[torch.Tensor] = None,
    crop_tolerance: float = 0.1,
    mask_interpolation: str = "bicubic",
    alignment: int = 16
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Transform reference image/mask for VAE reference tokens matching Krea 2 pixel-path geometry.

    Key principles:
    - NO black target-sized canvas padding.
    - Preserves native fitted reference grid aligned to multiples of 16 (/16 training geometry).
    - Near-matched-AR crop tolerance: if AR difference <= crop_tolerance, center-crops directly to target AR.
    - Masks receive the exact same spatial transformation.
    """
    if mode not in ("fit", "crop"):
        raise ValueError(f"Invalid reference_fit_mode '{mode}'. Expected 'fit' or 'crop'.")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.shape[-1] in (1, 3, 4):
        img_bchw = image.movedim(-1, 1).float()
    else:
        img_bchw = image.float()

    bs, c, ih, iw = img_bchw.shape

    mask_bchw = None
    if mask is not None:
        mask_float = mask.float()
        if mask_float.ndim == 2:
            mask_bchw = mask_float.unsqueeze(0).unsqueeze(0)
        elif mask_float.ndim == 3:
            mask_bchw = mask_float.unsqueeze(1)
        elif mask_float.ndim == 4:
            mask_bchw = mask_float
        if mask_bchw.shape[0] != bs:
            mask_bchw = mask_bchw[:1].repeat(bs, 1, 1, 1)

    target_ar = target_h / float(target_w)
    image_ar = ih / float(iw)

    # If aspect ratios match within crop_tolerance or mode is 'crop', center-crop then resize to target grid
    if mode == "crop" or abs(image_ar - target_ar) <= crop_tolerance:
        scale = max(target_h / float(ih), target_w / float(iw))
        new_h = int(round(ih * scale))
        new_w = int(round(iw * scale))

        scaled_img = F.interpolate(img_bchw, size=(new_h, new_w), mode="bicubic", antialias=True)
        y0 = (new_h - target_h) // 2
        x0 = (new_w - target_w) // 2
        cropped_img = scaled_img[..., y0:y0 + target_h, x0:x0 + target_w]

        cropped_mask = None
        if mask_bchw is not None:
            mask_m = "nearest" if mask_interpolation == "nearest" else "bicubic"
            kwargs = {"antialias": True} if mask_m == "bicubic" else {}
            scaled_mask = F.interpolate(mask_bchw, size=(new_h, new_w), mode=mask_m, **kwargs)
            cropped_mask = scaled_mask[..., y0:y0 + target_h, x0:x0 + target_w].squeeze(1).clamp(0.0, 1.0)

        return cropped_img.movedim(1, -1).clamp(0.0, 1.0), cropped_mask

    # 'fit' mode without black canvas padding: preserve native fitted grid aligned to multiples of 16
    scale = min(target_h / float(ih), target_w / float(iw))
    raw_h = ih * scale
    raw_w = iw * scale

    fh = max(alignment, int(round(raw_h / float(alignment))) * alignment)
    fw = max(alignment, int(round(raw_w / float(alignment))) * alignment)

    fitted_img = F.interpolate(img_bchw, size=(fh, fw), mode="bicubic", antialias=True)

    fitted_mask = None
    if mask_bchw is not None:
        mask_m = "nearest" if mask_interpolation == "nearest" else "bicubic"
        kwargs = {"antialias": True} if mask_m == "bicubic" else {}
        fitted_mask = F.interpolate(mask_bchw, size=(fh, fw), mode=mask_m, **kwargs).squeeze(1).clamp(0.0, 1.0)

    return fitted_img.movedim(1, -1).clamp(0.0, 1.0), fitted_mask


def _apply_sampling_transform_internal(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str,
    mask: Optional[torch.Tensor],
    mask_interpolation: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.shape[-1] in (1, 3, 4):
        img_bchw = image.movedim(-1, 1).float()
    else:
        img_bchw = image.float()

    bs, c, ih, iw = img_bchw.shape

    mask_bchw = None
    if mask is not None:
        mask_float = mask.float()
        if mask_float.ndim == 2:
            mask_bchw = mask_float.unsqueeze(0).unsqueeze(0)
        elif mask_float.ndim == 3:
            mask_bchw = mask_float.unsqueeze(1)
        elif mask_float.ndim == 4:
            mask_bchw = mask_float
        if mask_bchw.shape[0] != bs:
            mask_bchw = mask_bchw[:1].repeat(bs, 1, 1, 1)

    if mode == "stretch":
        res_img = F.interpolate(img_bchw, size=(target_h, target_w), mode="bicubic", antialias=True)
        res_mask = None
        if mask_bchw is not None:
            mask_m = "nearest" if mask_interpolation == "nearest" else "bicubic"
            kwargs = {"antialias": True} if mask_m == "bicubic" else {}
            res_mask = F.interpolate(mask_bchw, size=(target_h, target_w), mode=mask_m, **kwargs).squeeze(1).clamp(0.0, 1.0)
        return res_img.movedim(1, -1).clamp(0.0, 1.0), res_mask

    elif mode == "crop":
        scale = max(target_h / ih, target_w / iw)
        new_h = int(round(ih * scale))
        new_w = int(round(iw * scale))

        scaled_img = F.interpolate(img_bchw, size=(new_h, new_w), mode="bicubic", antialias=True)
        y0 = (new_h - target_h) // 2
        x0 = (new_w - target_w) // 2
        cropped_img = scaled_img[..., y0:y0 + target_h, x0:x0 + target_w]

        cropped_mask = None
        if mask_bchw is not None:
            mask_m = "nearest" if mask_interpolation == "nearest" else "bicubic"
            kwargs = {"antialias": True} if mask_m == "bicubic" else {}
            scaled_mask = F.interpolate(mask_bchw, size=(new_h, new_w), mode=mask_m, **kwargs)
            cropped_mask = scaled_mask[..., y0:y0 + target_h, x0:x0 + target_w].squeeze(1).clamp(0.0, 1.0)

        return cropped_img.movedim(1, -1).clamp(0.0, 1.0), cropped_mask

    elif mode == "fit":
        scale = min(target_h / ih, target_w / iw)
        new_h = max(1, int(round(ih * scale)))
        new_w = max(1, int(round(iw * scale)))

        scaled_img = F.interpolate(img_bchw, size=(new_h, new_w), mode="bicubic", antialias=True)

        padded_img = torch.zeros((bs, c, target_h, target_w), dtype=image.dtype, device=image.device)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        padded_img[..., y0:y0 + new_h, x0:x0 + new_w] = scaled_img

        padded_mask = None
        if mask_bchw is not None:
            mask_m = "nearest" if mask_interpolation == "nearest" else "bicubic"
            kwargs = {"antialias": True} if mask_m == "bicubic" else {}
            scaled_mask = F.interpolate(mask_bchw, size=(new_h, new_w), mode=mask_m, **kwargs)

            padded_mask_tensor = torch.zeros((bs, 1, target_h, target_w), dtype=mask_bchw.dtype, device=mask_bchw.device)
            padded_mask_tensor[..., y0:y0 + new_h, x0:x0 + new_w] = scaled_mask
            padded_mask = padded_mask_tensor.squeeze(1).clamp(0.0, 1.0)

        return padded_img.movedim(1, -1).clamp(0.0, 1.0), padded_mask

    raise RuntimeError("Unreachable mode state in _apply_sampling_transform_internal")
