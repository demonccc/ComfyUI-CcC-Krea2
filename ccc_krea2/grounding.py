"""Qwen3-VL grounding image preprocessor and preset resolution logic."""

import torch
import torch.nn.functional as F
from .constants import LOGGER_PREFIX


def resolve_grounding_px(preset: str, custom_px: int) -> int:
    """Resolves grounding resolution based on preset selection."""
    if preset == "balanced":
        return 768
    elif preset == "max_identity":
        return 1024
    return custom_px


def resize_grounding_image(
    image: torch.Tensor,
    resize_mode: str = "normalize",
    grounding_px: int = 768,
    grounding_min_px: int = 512,
    grounding_max_px: int = 1024,
    grounding_preset: str = "balanced"
) -> torch.Tensor:
    """Preprocesses reference images for Qwen3-VL semantic grounding.

    Input: ComfyUI image tensor (B, H, W, C) with float values in [0.0, 1.0].
    Presets:
    - balanced: 768px (Default standard resolution)
    - max_identity: 1024px
    - custom: Uses grounding_px value directly

    Modes:
    - none: Returns the original image unchanged.
    - downscale_only: Reduces longest edge to effective_px only if larger.
    - normalize: Resizes longest edge to exactly effective_px.
    - clamp: Upscales if longest edge < grounding_min_px; downscales if > grounding_max_px.

    Aspect ratio is strictly preserved in all modes.
    """
    if image is None:
        raise ValueError(f"{LOGGER_PREFIX} grounding image input cannot be None.")

    if resize_mode not in ("none", "downscale_only", "normalize", "clamp"):
        raise ValueError(
            f"{LOGGER_PREFIX} Invalid grounding_resize_mode '{resize_mode}'. "
            "Allowed modes: 'none', 'downscale_only', 'normalize', 'clamp'."
        )

    if resize_mode == "none":
        return image

    if image.ndim == 3:
        image = image.unsqueeze(0)

    effective_px = resolve_grounding_px(grounding_preset, grounding_px)

    if image.shape[-1] in (1, 3, 4):
        img_bchw = image.movedim(-1, 1).float()
    else:
        img_bchw = image.float()

    bs, c, h, w = img_bchw.shape
    longest_edge = max(h, w)
    target_longest: int = longest_edge

    if resize_mode == "downscale_only":
        if longest_edge > effective_px:
            target_longest = effective_px

    elif resize_mode == "normalize":
        target_longest = effective_px

    elif resize_mode == "clamp":
        if longest_edge < grounding_min_px:
            target_longest = grounding_min_px
        elif longest_edge > grounding_max_px:
            target_longest = grounding_max_px

    if target_longest == longest_edge or target_longest <= 0:
        return img_bchw.movedim(1, -1).clamp(0.0, 1.0)

    scale = target_longest / float(longest_edge)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if scale < 1.0:
        res = F.interpolate(img_bchw, size=(new_h, new_w), mode="area")
    else:
        res = F.interpolate(img_bchw, size=(new_h, new_w), mode="bicubic", antialias=True)

    return res.movedim(1, -1).clamp(0.0, 1.0)
