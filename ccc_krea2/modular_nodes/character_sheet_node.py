"""Deterministic 1:1 character-sheet composer for Krea2 CcC Edit."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY


PRESET_4_HEADS_1_BODY = "4 heads + 1 body (2x2 + 1)"
PRESET_3_HEADS_1_BODY = "3 heads + 1 body"
PRESET_2_HEADS_1_BODY = "2 heads + 1 body"
PRESET_4_HEADS = "4 heads (2x2)"

CHARACTER_SHEET_PRESETS = (
    PRESET_4_HEADS_1_BODY,
    PRESET_3_HEADS_1_BODY,
    PRESET_2_HEADS_1_BODY,
    PRESET_4_HEADS,
)

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


def _body_column_layout(
    resolution: int,
    outer_margin: int,
    padding: int,
    head_count: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Left head column(s) plus one full-height body column."""
    inner = resolution - 2 * outer_margin
    if inner <= 0:
        raise ValueError("Character Sheet outer_margin leaves no usable canvas.")

    thirds = _split_extent(inner, 3, padding)
    left_w = thirds[0] + padding + thirds[1]
    body_w = thirds[2]
    body_x = outer_margin + left_w + padding

    rects: Dict[str, Tuple[int, int, int, int]] = {}
    if head_count == 4:
        head_ws = _split_extent(left_w, 2, padding)
        head_hs = _split_extent(inner, 2, padding)
        names = ("front_view", "three_quarter_view", "profile_view", "extra_view")
        index = 0
        y = outer_margin
        for row_h in head_hs:
            x = outer_margin
            for col_w in head_ws:
                rects[names[index]] = (x, y, col_w, row_h)
                index += 1
                x += col_w + padding
            y += row_h + padding
    else:
        head_hs = _split_extent(inner, head_count, padding)
        names = ("front_view", "three_quarter_view", "profile_view")[:head_count]
        y = outer_margin
        for name, row_h in zip(names, head_hs):
            rects[name] = (outer_margin, y, left_w, row_h)
            y += row_h + padding

    rects["full_body_view"] = (body_x, outer_margin, body_w, inner)
    return rects


def _layout_rects(
    preset: str,
    resolution: int,
    outer_margin: int,
    padding: int,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Return semantic slot rectangles as x, y, width, height."""
    inner = resolution - 2 * outer_margin
    if inner <= 0:
        raise ValueError("Character Sheet outer_margin leaves no usable canvas.")

    if preset == PRESET_4_HEADS_1_BODY:
        return _body_column_layout(resolution, outer_margin, padding, head_count=4)
    if preset == PRESET_3_HEADS_1_BODY:
        return _body_column_layout(resolution, outer_margin, padding, head_count=3)
    if preset == PRESET_2_HEADS_1_BODY:
        return _body_column_layout(resolution, outer_margin, padding, head_count=2)
    if preset == PRESET_4_HEADS:
        ws = _split_extent(inner, 2, padding)
        hs = _split_extent(inner, 2, padding)
        names = ("front_view", "three_quarter_view", "profile_view", "extra_view")
        rects: Dict[str, Tuple[int, int, int, int]] = {}
        index = 0
        y = outer_margin
        for row_h in hs:
            x = outer_margin
            for col_w in ws:
                rects[names[index]] = (x, y, col_w, row_h)
                index += 1
                x += col_w + padding
            y += row_h + padding
        return rects

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
    mode: str,
    background_value: float,
) -> torch.Tensor:
    """Fit one BHWC image into a slot using contain or centered cover."""
    image = _normalize_image(image, "slot")
    src_h, src_w = int(image.shape[1]), int(image.shape[2])
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Character Sheet input image has an empty spatial dimension.")

    if mode == "contain":
        scale = min(target_w / float(src_w), target_h / float(src_h))
        out_w = max(1, min(target_w, int(round(src_w * scale))))
        out_h = max(1, min(target_h, int(round(src_h * scale))))
        nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(nchw, size=(out_h, out_w), mode="bicubic", align_corners=False)
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

    if mode == "cover":
        scale = max(target_w / float(src_w), target_h / float(src_h))
        out_w = max(target_w, int(round(src_w * scale)))
        out_h = max(target_h, int(round(src_h * scale)))
        nchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(nchw, size=(out_h, out_w), mode="bicubic", align_corners=False)
        resized = resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        y = max(0, (out_h - target_h) // 2)
        x = max(0, (out_w - target_w) // 2)
        return resized[:, y:y + target_h, x:x + target_w, :]

    raise ValueError(f"Unsupported Character Sheet slot_fit: {mode!r}")


class CcCKrea2CharacterSheet:
    """Compose semantic character views into one deterministic square IMAGE."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("character_sheet",)
    FUNCTION = "compose"
    DESCRIPTION = (
        "Builds a 1:1 character reference sheet from semantic view inputs using only "
        "resize/crop/composition. No model, VAE, Qwen or latent processing is performed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_preset": (
                    CHARACTER_SHEET_PRESETS,
                    {"default": PRESET_4_HEADS_1_BODY},
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 16,
                        "tooltip": "Square output size. The node always returns resolution x resolution.",
                    },
                ),
                "slot_fit": (
                    ("contain", "cover"),
                    {
                        "default": "contain",
                        "tooltip": "contain preserves the full source view; cover fills the slot with a centered crop.",
                    },
                ),
                "padding": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Gap in pixels between character-sheet slots.",
                    },
                ),
                "outer_margin": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Margin in pixels around the square sheet.",
                    },
                ),
                "background": (
                    tuple(BACKGROUND_VALUES),
                    {"default": "white"},
                ),
            },
            "optional": {
                "front_view": (
                    "IMAGE",
                    {"tooltip": "Front-facing head/upper-body reference."},
                ),
                "three_quarter_view": (
                    "IMAGE",
                    {"tooltip": "Three-quarter head/upper-body reference."},
                ),
                "profile_view": (
                    "IMAGE",
                    {"tooltip": "Profile head/upper-body reference."},
                ),
                "extra_view": (
                    "IMAGE",
                    {"tooltip": "Additional useful head view, angle or neutral expression."},
                ),
                "full_body_view": (
                    "IMAGE",
                    {"tooltip": "Full-body reference, ideally head-to-toe."},
                ),
            },
        }

    def compose(
        self,
        layout_preset: str,
        resolution: int,
        slot_fit: str,
        padding: int,
        outer_margin: int,
        background: str,
        front_view: Optional[torch.Tensor] = None,
        three_quarter_view: Optional[torch.Tensor] = None,
        profile_view: Optional[torch.Tensor] = None,
        extra_view: Optional[torch.Tensor] = None,
        full_body_view: Optional[torch.Tensor] = None,
    ):
        images = {
            "front_view": front_view,
            "three_quarter_view": three_quarter_view,
            "profile_view": profile_view,
            "extra_view": extra_view,
            "full_body_view": full_body_view,
        }

        rects = _layout_rects(
            preset=layout_preset,
            resolution=int(resolution),
            outer_margin=int(outer_margin),
            padding=int(padding),
        )
        missing = [name for name in rects if images.get(name) is None]
        if missing:
            raise ValueError(
                f"Character Sheet preset {layout_preset!r} requires: {', '.join(rects)}. "
                f"Missing: {', '.join(missing)}."
            )

        normalized = {name: _normalize_image(images[name], name) for name in rects}
        first = normalized[next(iter(rects))]
        bg = BACKGROUND_VALUES.get(background)
        if bg is None:
            raise ValueError(f"Unsupported Character Sheet background: {background!r}.")

        canvas = torch.full(
            (1, int(resolution), int(resolution), 3),
            float(bg),
            dtype=first.dtype,
            device=first.device,
        )

        for name, (x, y, width, height) in rects.items():
            source = normalized[name].to(device=canvas.device, dtype=canvas.dtype)
            fitted = _fit_image(
                source,
                target_h=height,
                target_w=width,
                mode=slot_fit,
                background_value=float(bg),
            )
            canvas[:, y:y + height, x:x + width, :] = fitted

        return (canvas.clamp(0.0, 1.0),)
