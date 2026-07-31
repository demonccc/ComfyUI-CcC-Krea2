"""Pixel-space geometric transformations matching Krea 2 reference fit geometry."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


def resize_tensor(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    method: str = "auto"
) -> torch.Tensor:
    """Centralized tensor resizing helper for images and features.

    Supported methods: 'auto', 'nearest-exact', 'bilinear', 'bicubic', 'area', 'lanczos'.
    Preserves input tensor device and floating-point dtype.
    """
    orig_ndim = tensor.ndim
    is_channels_last = False

    if orig_ndim == 3:
        if tensor.shape[-1] in (1, 3, 4):
            t_bchw = tensor.unsqueeze(0).movedim(-1, 1)
            is_channels_last = True
        else:
            t_bchw = tensor.unsqueeze(0)
    elif orig_ndim == 4:
        if tensor.shape[-1] in (1, 3, 4):
            t_bchw = tensor.movedim(-1, 1)
            is_channels_last = True
        else:
            t_bchw = tensor
    else:
        t_bchw = tensor

    curr_h, curr_w = t_bchw.shape[-2], t_bchw.shape[-1]

    if (curr_h, curr_w) == (target_h, target_w):
        return tensor

    method_key = method.lower()
    if method_key == "auto":
        if target_h * target_w < curr_h * curr_w:
            method_key = "area"
        else:
            method_key = "bicubic"

    if method_key == "nearest-exact":
        out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="nearest-exact")
    elif method_key == "nearest":
        out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="nearest")
    elif method_key == "bilinear":
        out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="bilinear", antialias=True)
    elif method_key == "bicubic":
        out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="bicubic", antialias=True)
    elif method_key == "area":
        out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="area")
    elif method_key == "lanczos":
        try:
            import comfy.utils
            out_bchw = comfy.utils.common_upscale(
                t_bchw.float(), target_w, target_h, upscale_method="lanczos", crop="disabled"
            )
        except Exception:
            out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="bicubic", antialias=True)
    else:
        out_bchw = F.interpolate(t_bchw.float(), size=(target_h, target_w), mode="bicubic", antialias=True)

    out_bchw = out_bchw.to(device=tensor.device, dtype=tensor.dtype if tensor.is_floating_point() else torch.float32)

    if orig_ndim == 3:
        if is_channels_last:
            return out_bchw.squeeze(0).movedim(0, -1)
        return out_bchw.squeeze(0)
    elif orig_ndim == 4:
        if is_channels_last:
            return out_bchw.movedim(1, -1)
        return out_bchw

    return out_bchw


def apply_sampling_transform(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "fit",
    mask: Optional[torch.Tensor] = None,
    mask_interpolation: str = "nearest-exact",
    resize_method: str = "auto"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Transform base image/mask for sampling LATENT output.

    Modes allowed:
    - fit: Preserve aspect ratio, scale inside, center-pad to target_h x target_w.
    - crop: Preserve aspect ratio, scale to cover, center-crop to target_h x target_w.
    - stretch: Directly resize to target_h x target_w without AR preservation.
    """
    if mode not in ("fit", "crop", "stretch"):
        raise ValueError(f"Invalid sampling_resize_mode '{mode}'. Expected 'fit', 'crop', or 'stretch'.")
    return _apply_sampling_transform_internal(
        image=image,
        target_h=target_h,
        target_w=target_w,
        mode=mode,
        mask=mask,
        mask_interpolation=mask_interpolation,
        resize_method=resize_method
    )


def apply_reference_fit_transform(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "fit",
    mask: Optional[torch.Tensor] = None,
    crop_tolerance: float = 0.08,
    mask_interpolation: str = "bilinear",
    alignment: int = 16,
    resize_method: str = "auto"
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Transform reference image/mask for VAE reference tokens matching Krea 2 pixel-path geometry.

    Key principles:
    - NO black target-sized canvas padding.
    - Preserves native fitted reference grid aligned to /16 floor dimensions.
    - Near-matched scale check: (image_h * scale >= target_h * (1 - 0.08)) and (image_w * scale >= target_w * (1 - 0.08)).
    - For near-match or crop modes, center-crop source image first then resize to target_h x target_w.
    - Masks receive the exact same spatial crop and resize transformation (hard -> nearest-exact, soft -> bilinear).
    """
    if mode not in ("fit", "crop"):
        raise ValueError(f"Invalid reference_fit_mode '{mode}'. Expected 'fit' or 'crop'.")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.shape[-1] in (1, 3, 4):
        img_bchw = image.movedim(-1, 1).float()
        is_channels_last = True
    else:
        img_bchw = image.float()
        is_channels_last = False

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

    scale_fit = min(target_h / float(ih), target_w / float(iw))
    is_near_match = (
        (ih * scale_fit >= target_h * (1.0 - crop_tolerance)) and
        (iw * scale_fit >= target_w * (1.0 - crop_tolerance))
    )

    mask_m = "nearest-exact" if mask_interpolation in ("nearest", "nearest-exact", "hard") else "bilinear"
    mask_kwargs = {"antialias": True} if mask_m == "bilinear" else {}

    # 1. Mode 'crop' or near-matched scale: center-crop original source image first, then resize to target_h x target_w
    if mode == "crop" or is_near_match:
        scale_crop = max(target_h / float(ih), target_w / float(iw))
        crop_h = min(ih, int(round(target_h / scale_crop)))
        crop_w = min(iw, int(round(target_w / scale_crop)))

        y0 = (ih - crop_h) // 2
        x0 = (iw - crop_w) // 2

        cropped_src = img_bchw[..., y0:y0 + crop_h, x0:x0 + crop_w]
        fitted_img_bchw = _resize_bchw(cropped_src, target_h=target_h, target_w=target_w, method=resize_method)

        fitted_mask = None
        if mask_bchw is not None:
            cropped_mask_src = mask_bchw[..., y0:y0 + crop_h, x0:x0 + crop_w]
            fitted_mask = F.interpolate(cropped_mask_src, size=(target_h, target_w), mode=mask_m, **mask_kwargs).squeeze(1).clamp(0.0, 1.0)

        ref_fit_meta = {"spatial_hw": (target_h, target_w)}
        out_img = fitted_img_bchw.movedim(1, -1) if is_channels_last else fitted_img_bchw
        return out_img.clamp(0.0, 1.0), fitted_mask, ref_fit_meta

    # 2. Genuine AR mismatch in 'fit' mode: align to /16 floor dimensions and crop source
    fit_scale = min(target_h / float(ih), target_w / float(iw))
    raw_h = ih * fit_scale
    raw_w = iw * fit_scale

    fh = max(alignment, (int(raw_h) // alignment) * alignment)
    fw = max(alignment, (int(raw_w) // alignment) * alignment)

    # Calculate source crop dimensions as round(fitted_dimension / fit_scale)
    crop_h = min(ih, int(round(fh / fit_scale)))
    crop_w = min(iw, int(round(fw / fit_scale)))

    sy0 = (ih - crop_h) // 2
    sx0 = (iw - crop_w) // 2

    cropped_src = img_bchw[..., sy0:sy0 + crop_h, sx0:sx0 + crop_w]
    fitted_img_bchw = _resize_bchw(cropped_src, target_h=fh, target_w=fw, method=resize_method)

    fitted_mask = None
    if mask_bchw is not None:
        cropped_mask_src = mask_bchw[..., sy0:sy0 + crop_h, sx0:sx0 + crop_w]
        fitted_mask = F.interpolate(cropped_mask_src, size=(fh, fw), mode=mask_m, **mask_kwargs).squeeze(1).clamp(0.0, 1.0)

    ref_fit_meta = {"spatial_hw": (fh, fw)}
    out_img = fitted_img_bchw.movedim(1, -1) if is_channels_last else fitted_img_bchw
    return out_img.clamp(0.0, 1.0), fitted_mask, ref_fit_meta


def _resize_bchw(tensor_bchw: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    """Helper to resize BCHW tensor using resize_tensor while maintaining BCHW shape."""
    res = resize_tensor(tensor_bchw, target_h=target_h, target_w=target_w, method=method)
    if res.ndim == 4 and res.shape[1] not in (1, 3, 4) and res.shape[-1] in (1, 3, 4):
        # res was returned as BHWC
        return res.movedim(-1, 1)
    return res


def _apply_sampling_transform_internal(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str,
    mask: Optional[torch.Tensor],
    mask_interpolation: str,
    resize_method: str = "auto"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.shape[-1] in (1, 3, 4):
        img_bchw = image.movedim(-1, 1).float()
        is_channels_last = True
    else:
        img_bchw = image.float()
        is_channels_last = False

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

    # Inpaint mask ALWAYS uses nearest-exact
    mask_m = "nearest-exact"

    if mode == "stretch":
        res_img = _resize_bchw(img_bchw, target_h=target_h, target_w=target_w, method=resize_method)
        res_mask = None
        if mask_bchw is not None:
            res_mask = F.interpolate(mask_bchw, size=(target_h, target_w), mode=mask_m).squeeze(1).clamp(0.0, 1.0)
        out_img = res_img.movedim(1, -1) if is_channels_last else res_img
        return out_img.clamp(0.0, 1.0), res_mask

    elif mode == "crop":
        scale = max(target_h / ih, target_w / iw)
        new_h = int(round(ih * scale))
        new_w = int(round(iw * scale))

        scaled_img = _resize_bchw(img_bchw, target_h=new_h, target_w=new_w, method=resize_method)
        y0 = (new_h - target_h) // 2
        x0 = (new_w - target_w) // 2
        cropped_img = scaled_img[..., y0:y0 + target_h, x0:x0 + target_w]

        cropped_mask = None
        if mask_bchw is not None:
            scaled_mask = F.interpolate(mask_bchw, size=(new_h, new_w), mode=mask_m)
            cropped_mask = scaled_mask[..., y0:y0 + target_h, x0:x0 + target_w].squeeze(1).clamp(0.0, 1.0)

        out_img = cropped_img.movedim(1, -1) if is_channels_last else cropped_img
        return out_img.clamp(0.0, 1.0), cropped_mask

    elif mode == "fit":
        scale = min(target_h / ih, target_w / iw)
        new_h = max(1, int(round(ih * scale)))
        new_w = max(1, int(round(iw * scale)))

        scaled_img = _resize_bchw(img_bchw, target_h=new_h, target_w=new_w, method=resize_method)

        padded_img = torch.zeros((bs, c, target_h, target_w), dtype=image.dtype, device=image.device)
        y0 = (target_h - new_h) // 2
        x0 = (target_w - new_w) // 2
        padded_img[..., y0:y0 + new_h, x0:x0 + new_w] = scaled_img

        padded_mask = None
        if mask_bchw is not None:
            scaled_mask = F.interpolate(mask_bchw, size=(new_h, new_w), mode=mask_m)

            padded_mask_tensor = torch.zeros((bs, 1, target_h, target_w), dtype=mask_bchw.dtype, device=mask_bchw.device)
            padded_mask_tensor[..., y0:y0 + new_h, x0:x0 + new_w] = scaled_mask
            padded_mask = padded_mask_tensor.squeeze(1).clamp(0.0, 1.0)

        out_img = padded_img.movedim(1, -1) if is_channels_last else padded_img
        return out_img.clamp(0.0, 1.0), padded_mask

    raise RuntimeError("Unreachable mode state in _apply_sampling_transform_internal")
