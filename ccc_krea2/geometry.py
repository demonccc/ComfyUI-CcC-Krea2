"""Shared image tensor resampling helper; target/reference layout lives in dedicated geometry modules."""

import torch
import torch.nn.functional as F


def resize_tensor(tensor: torch.Tensor, target_h: int, target_w: int, method: str = "auto") -> torch.Tensor:
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
