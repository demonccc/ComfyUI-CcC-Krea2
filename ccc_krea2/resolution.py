"""Output resolution and megapixel geometry calculation module for CcC Krea2."""

import math
import logging
from typing import Dict, Optional, Tuple
import torch

logger = logging.getLogger("CcCKrea2")

VALID_ROLE_RESOLUTIONS = ("subject", "scene", "source")


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
    role_resolution_limit_mode: str = "max_megapixels",
    role_resolution_max_megapixels: float = 2.0,
    node_name: str = ""
) -> Tuple[int, int]:
    """Calculate output width and height based on role-based resolution or custom megapixel aspect ratio.

    Rules:
    - Role-based: Read original image spatial dimensions (subject, scene, source only - outfit is never a resolution source),
      preserve aspect ratio, align to /16 (min 128x128).
      If role_resolution_limit_mode == "max_megapixels" and aligned area > role_resolution_max_megapixels:
      scale width and height down proportionally, preserving aspect ratio, until output area <= limit.
      Align result again to multiples of 16 (min 128x128). Log concise info message indicating role resolution was limited.
    - Custom: Calculate area = megapixels * 1,000,000 with aspect ratio from selected role (or auto fallback).
      Align width & height to /16 (min 128x128). Custom mode ignores role_resolution_limit_mode.
    """
    if output_resolution != "custom":
        role_image = None
        if output_resolution in VALID_ROLE_RESOLUTIONS:
            role_image = node_images.get(output_resolution)

        if role_image is None and default_auto_role in VALID_ROLE_RESOLUTIONS:
            role_image = node_images.get(default_auto_role)

        if role_image is None:
            # Default fallback 1024x1024
            return 1024, 1024

        orig_w, orig_h = _get_image_spatial_wh(role_image)

        out_w = max(128, int(round(orig_w / 16.0) * 16))
        out_h = max(128, int(round(orig_h / 16.0) * 16))

        if role_resolution_limit_mode == "max_megapixels":
            max_pixels = role_resolution_max_megapixels * 1_000_000.0
            aligned_area = float(out_w * out_h)
            if aligned_area > max_pixels:
                aspect = float(orig_w) / float(orig_h)
                raw_w = math.sqrt(max_pixels * aspect)
                raw_h = math.sqrt(max_pixels / aspect)

                scaled_w = max(128, int(round(raw_w / 16.0) * 16))
                scaled_h = max(128, int(round(raw_h / 16.0) * 16))

                while scaled_w * scaled_h > max_pixels and (scaled_w > 128 or scaled_h > 128):
                    if scaled_w / float(orig_w) >= scaled_h / float(orig_h) and scaled_w > 128:
                        scaled_w = max(128, scaled_w - 16)
                    elif scaled_h > 128:
                        scaled_h = max(128, scaled_h - 16)
                    else:
                        break

                logger.info(
                    f"[CcC Krea2] [{node_name}] Role-based output resolution for '{output_resolution}' "
                    f"limited from {out_w}x{out_h} ({aligned_area / 1e6:.2f} MP) to {scaled_w}x{scaled_h} "
                    f"({scaled_w * scaled_h / 1e6:.2f} MP, limit: {role_resolution_max_megapixels:.2f} MP)."
                )
                return scaled_w, scaled_h

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
