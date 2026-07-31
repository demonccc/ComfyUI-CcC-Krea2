"""Output resolution and megapixel geometry calculation module for CcC Krea2."""

import math
import logging
from typing import Dict, Optional, Tuple
import torch

logger = logging.getLogger("CcCKrea2")


def _get_image_spatial_wh(image: torch.Tensor) -> Tuple[int, int]:
    """Extract spatial width and height from PyTorch image tensor [B, H, W, C] or [B, C, H, W] or [H, W, C]."""
    if image.ndim == 4:
        if image.shape[-1] in (1, 3, 4):
            return image.shape[2], image.shape[1]  # W, H from [B, H, W, C]
        else:
            return image.shape[3], image.shape[2]  # W, H from [B, C, H, W]
    elif image.ndim == 3:
        if image.shape[-1] in (1, 3, 4):
            return image.shape[1], image.shape[0]
        else:
            return image.shape[2], image.shape[1]
    raise ValueError(f"Invalid image tensor dimensions: {image.shape}")


def resolve_output_resolution(
    output_resolution: str,
    megapixels: float,
    node_images: Dict[str, Optional[torch.Tensor]],
    default_auto_role: str,
    custom_aspect_source: str = "auto",
    node_name: str = ""
) -> Tuple[int, int]:
    """Calculate output width and height based on role-based resolution or custom megapixel aspect ratio.

    Rules:
    - Role-based: Read original image spatial dimensions, preserve aspect ratio, align to /16 (min 128x128).
      Log VRAM warning if total pixels > 2,000,000.
    - Custom: Calculate area = megapixels * 1,000,000 with aspect ratio from selected role (or auto fallback).
      Align width & height to /16 (min 128x128).
    """
    if output_resolution != "custom":
        role_image = node_images.get(output_resolution)
        if role_image is None:
            # Fallback to default auto role if specified role image is missing
            role_image = node_images.get(default_auto_role)

        if role_image is None:
            # Default fallback 1024x1024
            return 1024, 1024

        orig_w, orig_h = _get_image_spatial_wh(role_image)

        out_w = max(128, int(round(orig_w / 16.0) * 16))
        out_h = max(128, int(round(orig_h / 16.0) * 16))

        if out_w * out_h > 2_000_000:
            logger.warning(
                f"[CcC Krea2] [{node_name}] High output resolution calculated ({out_w}x{out_h} = {out_w * out_h / 1e6:.2f} MP). "
                f"This may require high VRAM during generation."
            )

        return out_w, out_h

    # Custom megapixel resolution
    selected_role = custom_aspect_source
    if selected_role == "auto" or selected_role not in node_images or node_images.get(selected_role) is None:
        if selected_role != "auto":
            logger.warning(
                f"[CcC Krea2] [{node_name}] Selected custom_aspect_source '{selected_role}' "
                f"is not available in this node. Falling back to default role '{default_auto_role}'."
            )
        selected_role = default_auto_role

    aspect_image = node_images.get(selected_role)
    if aspect_image is None:
        # Fallback square aspect ratio if no image is available
        source_w, source_h = 1024, 1024
    else:
        source_w, source_h = _get_image_spatial_wh(aspect_image)

    target_pixels = megapixels * 1_000_000.0
    aspect = float(source_w) / float(source_h)

    raw_w = math.sqrt(target_pixels * aspect)
    raw_h = math.sqrt(target_pixels / aspect)

    out_w = max(128, int(round(raw_w / 16.0) * 16))
    out_h = max(128, int(round(raw_h / 16.0) * 16))

    if out_w * out_h > 2_000_000:
        logger.warning(
            f"[CcC Krea2] [{node_name}] Custom output resolution calculated ({out_w}x{out_h} = {out_w * out_h / 1e6:.2f} MP). "
            f"This may require high VRAM during generation."
        )

    return out_w, out_h
