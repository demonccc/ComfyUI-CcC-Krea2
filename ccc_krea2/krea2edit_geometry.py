"""Pixel-space visual reference fit resolver matching Krea2Edit geometry principles."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from ccc_krea2.geometry import resize_tensor


def resolve_visual_reference_fit(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "auto",
    mask: Optional[torch.Tensor] = None,
    alignment: int = 16
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Transform a reference image and optional mask according to target latent geometry.

    Modes:
    - 'auto': Resolves to 'exact', 'crop_only', 'crop_and_resize', or 'fit'.
    - 'fit': Training-matched /16 floor dimension alignment without black canvas padding.
    - 'crop': Center crop to target aspect ratio and resize to target_h x target_w.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)

    if image.shape[-1] in (1, 3, 4):
        img_bchw = image.movedim(-1, 1).float()
        is_channels_last = True
    else:
        img_bchw = image.float()
        is_channels_last = False

    bs, c, ih, iw = img_bchw.shape
    src_ar = iw / float(ih)
    target_ar = target_w / float(target_h)

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

    resolved_mode = mode

    if mode == "auto":
        # 1. Exact match check
        if (ih, iw) == (target_h, target_w):
            resolved_mode = "exact"
        else:
            # 2. Crop-only check (dim diff <= 5%, area diff <= 10%)
            h_diff = (ih - target_h) / float(target_h)
            w_diff = (iw - target_w) / float(target_w)
            area_diff = (ih * iw - target_h * target_w) / float(target_h * target_w)

            if 0.0 <= h_diff <= 0.05 and 0.0 <= w_diff <= 0.05 and 0.0 <= area_diff <= 0.10:
                resolved_mode = "crop_only"
            else:
                # 3. Crop-and-resize check (AR diff <= 8%)
                ar_diff = abs(src_ar - target_ar) / target_ar
                if ar_diff <= 0.08:
                    resolved_mode = "crop_and_resize"
                else:
                    resolved_mode = "fit"

    # Execution of resolved mode
    if resolved_mode == "exact":
        out_img = img_bchw.movedim(1, -1) if is_channels_last else img_bchw
        out_mask = mask_bchw.squeeze(1).clamp(0.0, 1.0) if mask_bchw is not None else None
        return out_img.clamp(0.0, 1.0), out_mask, {"mode_requested": mode, "mode_resolved": "exact", "spatial_hw": (target_h, target_w)}

    elif resolved_mode in ("crop_only", "crop", "crop_and_resize"):
        if resolved_mode == "crop_only":
            y0 = (ih - target_h) // 2
            x0 = (iw - target_w) // 2
            cropped_img = img_bchw[..., y0:y0 + target_h, x0:x0 + target_w]
            out_img = cropped_img.movedim(1, -1) if is_channels_last else cropped_img
            out_mask = mask_bchw[..., y0:y0 + target_h, x0:x0 + target_w].squeeze(1).clamp(0.0, 1.0) if mask_bchw is not None else None
            return out_img.clamp(0.0, 1.0), out_mask, {"mode_requested": mode, "mode_resolved": "crop_only", "spatial_hw": (target_h, target_w)}

        # crop / crop_and_resize
        scale = max(target_h / float(ih), target_w / float(iw))
        crop_h = min(ih, int(round(target_h / scale)))
        crop_w = min(iw, int(round(target_w / scale)))

        y0 = (ih - crop_h) // 2
        x0 = (iw - crop_w) // 2
        cropped = img_bchw[..., y0:y0 + crop_h, x0:x0 + crop_w]
        resized = resize_tensor(cropped.movedim(1, -1) if is_channels_last else cropped, target_h=target_h, target_w=target_w)
        resized_bchw = resized.movedim(-1, 1) if is_channels_last else resized

        out_mask = None
        if mask_bchw is not None:
            cropped_mask = mask_bchw[..., y0:y0 + crop_h, x0:x0 + crop_w]
            out_mask = F.interpolate(cropped_mask, size=(target_h, target_w), mode="bilinear", antialias=True).squeeze(1).clamp(0.0, 1.0)

        out_img = resized_bchw.movedim(1, -1) if is_channels_last else resized_bchw
        return out_img.clamp(0.0, 1.0), out_mask, {"mode_requested": mode, "mode_resolved": resolved_mode, "spatial_hw": (target_h, target_w)}

    elif resolved_mode == "fit":
        scale = min(target_h / float(ih), target_w / float(iw))
        raw_h = ih * scale
        raw_w = iw * scale

        fh = max(alignment, (int(raw_h) // alignment) * alignment)
        fw = max(alignment, (int(raw_w) // alignment) * alignment)

        crop_h = min(ih, int(round(fh / scale)))
        crop_w = min(iw, int(round(fw / scale)))

        y0 = (ih - crop_h) // 2
        x0 = (iw - crop_w) // 2
        cropped = img_bchw[..., y0:y0 + crop_h, x0:x0 + crop_w]
        resized = resize_tensor(cropped.movedim(1, -1) if is_channels_last else cropped, target_h=fh, target_w=fw)
        resized_bchw = resized.movedim(-1, 1) if is_channels_last else resized

        out_mask = None
        if mask_bchw is not None:
            cropped_mask = mask_bchw[..., y0:y0 + crop_h, x0:x0 + crop_w]
            out_mask = F.interpolate(cropped_mask, size=(fh, fw), mode="bilinear", antialias=True).squeeze(1).clamp(0.0, 1.0)

        out_img = resized_bchw.movedim(1, -1) if is_channels_last else resized_bchw
        return out_img.clamp(0.0, 1.0), out_mask, {"mode_requested": mode, "mode_resolved": "fit", "spatial_hw": (fh, fw)}

    raise RuntimeError(f"Unreachable mode state '{resolved_mode}' in resolve_visual_reference_fit")
