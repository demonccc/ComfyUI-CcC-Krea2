"""Deterministic Krea-aligned character-sheet composer for Krea2 CcC Edit."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY
from .latent_node import (
    DEFAULT_KREA_PRESET_SIZE,
    KREA_PRESET_GEOMETRIES,
    KREA_PRESET_SIZES,
)


PRESET_1_PORTRAIT = "1 portrait"
PRESET_2_PORTRAITS = "2 portraits"
PRESET_3_PORTRAITS = "3 portraits"
PRESET_4_PORTRAITS = "4 portraits"
PRESET_1_BODY = "1 body"
PRESET_2_BODIES = "2 bodies"
PRESET_1_PORTRAIT_2_BODIES = "1 portrait + 2 bodies"
PRESET_3_PORTRAITS_1_BODY = "3 portraits + 1 body"
PRESET_4_PORTRAITS_1_BODY = "4 portraits + 1 body"

CHARACTER_SHEET_PRESETS = (
    PRESET_1_PORTRAIT,
    PRESET_2_PORTRAITS,
    PRESET_3_PORTRAITS,
    PRESET_4_PORTRAITS,
    PRESET_1_BODY,
    PRESET_2_BODIES,
    PRESET_1_PORTRAIT_2_BODIES,
    PRESET_3_PORTRAITS_1_BODY,
    PRESET_4_PORTRAITS_1_BODY,
)

PRESET_INPUTS = {
    PRESET_1_PORTRAIT: ("portrait_1",),
    PRESET_2_PORTRAITS: ("portrait_1", "portrait_2"),
    PRESET_3_PORTRAITS: ("portrait_1", "portrait_2", "portrait_3"),
    PRESET_4_PORTRAITS: ("portrait_1", "portrait_2", "portrait_3", "portrait_4"),
    PRESET_1_BODY: ("body_1",),
    PRESET_2_BODIES: ("body_1", "body_2"),
    PRESET_1_PORTRAIT_2_BODIES: ("portrait_1", "body_1", "body_2"),
    PRESET_3_PORTRAITS_1_BODY: ("portrait_1", "portrait_2", "portrait_3", "body_1"),
    PRESET_4_PORTRAITS_1_BODY: (
        "portrait_1",
        "portrait_2",
        "portrait_3",
        "portrait_4",
        "body_1",
    ),
}

BACKGROUND_VALUES = {
    "white": 1.0,
    "gray": 0.5,
    "black": 0.0,
}


def _split_extent(total: int, parts: int, gap: int) -> Tuple[int, ...]:
    """Split an extent into nearly equal integer parts while preserving every pixel."""
    usable = total - gap * (parts - 1)
    if usable <= 0:
        raise ValueError("Character Sheet padding leaves no room for image slots.")
    base, remainder = divmod(usable, parts)
    return tuple(base + (1 if i < remainder else 0) for i in range(parts))


def _inner_geometry(
    width: int,
    height: int,
    outer_margin: int,
) -> Tuple[int, int, int, int]:
    inner_w = width - 2 * outer_margin
    inner_h = height - 2 * outer_margin
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Character Sheet outer_margin leaves no usable canvas.")
    return outer_margin, outer_margin, inner_w, inner_h


def _row_layout(
    names: Tuple[str, ...],
    width: int,
    height: int,
    outer_margin: int,
    padding: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    x0, y0, inner_w, inner_h = _inner_geometry(width, height, outer_margin)
    widths = _split_extent(inner_w, len(names), padding)
    rects: Dict[str, Tuple[int, int, int, int]] = {}
    x = x0
    for name, slot_w in zip(names, widths):
        rects[name] = (x, y0, slot_w, inner_h)
        x += slot_w + padding
    return rects


def _portrait_grid_layout(
    names: Tuple[str, ...],
    width: int,
    height: int,
    outer_margin: int,
    padding: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    if len(names) <= 3:
        return _row_layout(names, width, height, outer_margin, padding)

    x0, y0, inner_w, inner_h = _inner_geometry(width, height, outer_margin)
    widths = _split_extent(inner_w, 2, padding)
    heights = _split_extent(inner_h, 2, padding)
    rects: Dict[str, Tuple[int, int, int, int]] = {}
    index = 0
    y = y0
    for slot_h in heights:
        x = x0
        for slot_w in widths:
            rects[names[index]] = (x, y, slot_w, slot_h)
            index += 1
            x += slot_w + padding
        y += slot_h + padding
    return rects


def _portraits_plus_body_layout(
    portrait_names: Tuple[str, ...],
    body_name: str,
    width: int,
    height: int,
    outer_margin: int,
    padding: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    x0, y0, inner_w, inner_h = _inner_geometry(width, height, outer_margin)
    columns = _split_extent(inner_w, 3, padding)
    portrait_w = columns[0] + padding + columns[1]
    body_w = columns[2]
    body_x = x0 + portrait_w + padding

    if len(portrait_names) == 4:
        portrait_widths = _split_extent(portrait_w, 2, padding)
        portrait_heights = _split_extent(inner_h, 2, padding)
        rects: Dict[str, Tuple[int, int, int, int]] = {}
        index = 0
        y = y0
        for slot_h in portrait_heights:
            x = x0
            for slot_w in portrait_widths:
                rects[portrait_names[index]] = (x, y, slot_w, slot_h)
                index += 1
                x += slot_w + padding
            y += slot_h + padding
    else:
        portrait_heights = _split_extent(inner_h, 2, padding)
        portrait_widths = _split_extent(portrait_w, 2, padding)
        rects = {
            portrait_names[0]: (x0, y0, portrait_widths[0], portrait_heights[0]),
            portrait_names[1]: (
                x0 + portrait_widths[0] + padding,
                y0,
                portrait_widths[1],
                portrait_heights[0],
            ),
            portrait_names[2]: (
                x0,
                y0 + portrait_heights[0] + padding,
                portrait_w,
                portrait_heights[1],
            ),
        }

    rects[body_name] = (body_x, y0, body_w, inner_h)
    return rects


def _layout_rects(
    preset: str,
    width: int,
    height: int,
    outer_margin: int,
    padding: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Return slot rectangles as x, y, width, height."""
    x0, y0, inner_w, inner_h = _inner_geometry(width, height, outer_margin)

    if preset == PRESET_1_PORTRAIT:
        return {"portrait_1": (x0, y0, inner_w, inner_h)}
    if preset == PRESET_2_PORTRAITS:
        return _portrait_grid_layout(
            ("portrait_1", "portrait_2"),
            width,
            height,
            outer_margin,
            padding,
        )
    if preset == PRESET_3_PORTRAITS:
        return _portrait_grid_layout(
            ("portrait_1", "portrait_2", "portrait_3"),
            width,
            height,
            outer_margin,
            padding,
        )
    if preset == PRESET_4_PORTRAITS:
        return _portrait_grid_layout(
            ("portrait_1", "portrait_2", "portrait_3", "portrait_4"),
            width,
            height,
            outer_margin,
            padding,
        )
    if preset == PRESET_1_BODY:
        return {"body_1": (x0, y0, inner_w, inner_h)}
    if preset == PRESET_2_BODIES:
        return _row_layout(
            ("body_1", "body_2"),
            width,
            height,
            outer_margin,
            padding,
        )
    if preset == PRESET_1_PORTRAIT_2_BODIES:
        return _row_layout(
            ("portrait_1", "body_1", "body_2"),
            width,
            height,
            outer_margin,
            padding,
        )
    if preset == PRESET_3_PORTRAITS_1_BODY:
        return _portraits_plus_body_layout(
            ("portrait_1", "portrait_2", "portrait_3"),
            "body_1",
            width,
            height,
            outer_margin,
            padding,
        )
    if preset == PRESET_4_PORTRAITS_1_BODY:
        return _portraits_plus_body_layout(
            ("portrait_1", "portrait_2", "portrait_3", "portrait_4"),
            "body_1",
            width,
            height,
            outer_margin,
            padding,
        )

    raise ValueError(f"Unsupported Character Sheet preset: {preset!r}")


def _normalize_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(image):
        raise TypeError(f"Character Sheet input {name!r} must be an IMAGE tensor.")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            f"Character Sheet input {name!r} must be HWC or BHWC; got shape {tuple(image.shape)}."
        )
    if image.shape[0] != 1:
        raise ValueError(
            f"Character Sheet input {name!r} must contain exactly one image; "
            f"received batch size {image.shape[0]}."
        )
    if image.shape[-1] < 3:
        raise ValueError(f"Character Sheet input {name!r} must have at least 3 channels.")
    return image[..., :3].to(dtype=torch.float32).clamp(0.0, 1.0)


def _fit_image(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    background_value: float,
) -> torch.Tensor:
    """Fit the complete source inside one slot without cropping or distortion."""
    image = _normalize_image(image, "slot")
    src_h, src_w = int(image.shape[1]), int(image.shape[2])
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Character Sheet input image has an empty spatial dimension.")

    scale = min(target_w / float(src_w), target_h / float(src_h))
    out_w = max(1, min(target_w, int(round(src_w * scale))))
    out_h = max(1, min(target_h, int(round(src_h * scale))))

    nchw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(
        nchw,
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    resized = resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    slot = torch.full(
        (1, target_h, target_w, 3),
        float(background_value),
        dtype=resized.dtype,
        device=resized.device,
    )
    y = (target_h - out_h) // 2
    x = (target_w - out_w) // 2
    slot[:, y:y + out_h, x:x + out_w, :] = resized
    return slot


class CcCKrea2CharacterSheet:
    """Compose generic portrait/body references into one Krea-aligned IMAGE."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("character_sheet",)
    FUNCTION = "compose"
    DESCRIPTION = (
        "Builds a Krea-aligned character reference sheet from generic portrait/body inputs. "
        "Every source is fitted completely inside its slot with preserved aspect ratio; "
        "no source crop, stretch, model, VAE, Qwen or latent processing is performed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheet_type": (
                    CHARACTER_SHEET_PRESETS,
                    {"default": PRESET_1_PORTRAIT},
                ),
                "sheet_geometry": (
                    KREA_PRESET_SIZES,
                    {
                        "default": DEFAULT_KREA_PRESET_SIZE,
                        "tooltip": "Final Character Sheet geometry. Uses the same curated Krea presets as Krea2 CcC Latent.",
                    },
                ),
                "padding": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Gap in pixels between Character Sheet slots.",
                    },
                ),
                "outer_margin": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Margin in pixels around the Character Sheet.",
                    },
                ),
                "background": (
                    tuple(BACKGROUND_VALUES),
                    {"default": "white"},
                ),
            },
            "optional": {
                "portrait_1": (
                    "IMAGE",
                    {"tooltip": "Primary portrait/identity reference. Any useful angle is valid."},
                ),
                "portrait_2": (
                    "IMAGE",
                    {"tooltip": "Additional portrait/identity reference. Any useful angle is valid."},
                ),
                "portrait_3": (
                    "IMAGE",
                    {"tooltip": "Additional portrait/identity reference. Any useful angle is valid."},
                ),
                "portrait_4": (
                    "IMAGE",
                    {"tooltip": "Additional portrait/identity reference. Any useful angle is valid."},
                ),
                "body_1": (
                    "IMAGE",
                    {"tooltip": "Primary full-body reference. Any useful view is valid."},
                ),
                "body_2": (
                    "IMAGE",
                    {"tooltip": "Additional full-body reference. Any useful view is valid."},
                ),
            },
        }

    def compose(
        self,
        sheet_type: str,
        sheet_geometry: str,
        padding: int,
        outer_margin: int,
        background: str,
        portrait_1: Optional[torch.Tensor] = None,
        portrait_2: Optional[torch.Tensor] = None,
        portrait_3: Optional[torch.Tensor] = None,
        portrait_4: Optional[torch.Tensor] = None,
        body_1: Optional[torch.Tensor] = None,
        body_2: Optional[torch.Tensor] = None,
    ):
        try:
            width, height = KREA_PRESET_GEOMETRIES[sheet_geometry]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Character Sheet geometry: {sheet_geometry!r}."
            ) from exc

        images = {
            "portrait_1": portrait_1,
            "portrait_2": portrait_2,
            "portrait_3": portrait_3,
            "portrait_4": portrait_4,
            "body_1": body_1,
            "body_2": body_2,
        }

        required_inputs = PRESET_INPUTS.get(sheet_type)
        if required_inputs is None:
            raise ValueError(f"Unsupported Character Sheet preset: {sheet_type!r}.")

        missing = [name for name in required_inputs if images.get(name) is None]
        if missing:
            raise ValueError(
                f"Character Sheet preset {sheet_type!r} requires: {', '.join(required_inputs)}. "
                f"Missing: {', '.join(missing)}."
            )

        rects = _layout_rects(
            preset=sheet_type,
            width=int(width),
            height=int(height),
            outer_margin=int(outer_margin),
            padding=int(padding),
        )

        normalized = {
            name: _normalize_image(images[name], name)
            for name in required_inputs
        }
        first = normalized[required_inputs[0]]
        bg = BACKGROUND_VALUES.get(background)
        if bg is None:
            raise ValueError(f"Unsupported Character Sheet background: {background!r}.")

        canvas = torch.full(
            (1, int(height), int(width), 3),
            float(bg),
            dtype=first.dtype,
            device=first.device,
        )

        for name, (x, y, slot_w, slot_h) in rects.items():
            source = normalized[name].to(device=canvas.device, dtype=canvas.dtype)
            fitted = _fit_image(
                source,
                target_h=slot_h,
                target_w=slot_w,
                background_value=float(bg),
            )
            canvas[:, y:y + slot_h, x:x + slot_w, :] = fitted

        return (canvas.clamp(0.0, 1.0),)
