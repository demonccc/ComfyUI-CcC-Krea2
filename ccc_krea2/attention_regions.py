"""Tagged target attention regions and spatial transform helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


REFERENCE_ATTENTION_SCOPES = ("global", "boost in region", "only in region")


@dataclass(frozen=True)
class AttentionRegion:
    """One user-declared target box in normalized source/canvas coordinates."""

    tag: str
    x: float
    y: float
    width: float
    height: float

    @property
    def box_normalized(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class AttentionRegionChain:
    """Ordered chain of uniquely tagged target regions."""

    entries: Tuple[AttentionRegion, ...] = ()

    def append(self, entry: AttentionRegion) -> "AttentionRegionChain":
        if any(existing.tag == entry.tag for existing in self.entries):
            raise ValueError(
                f"[Krea2 CcC Attention Region] Duplicate region tag '{entry.tag}'. "
                "Each attention region tag must be unique."
            )
        return AttentionRegionChain(self.entries + (entry,))

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class ResolvedAttentionRegion:
    """A target region after target pixel-space transformations have been applied."""

    tag: str
    source_box_normalized: Tuple[float, float, float, float]
    target_box_px: Tuple[float, float, float, float]
    target_box_normalized: Tuple[float, float, float, float]


def build_attention_region(
    tag: str,
    x_percent: float,
    y_percent: float,
    width_percent: float,
    height_percent: float,
) -> AttentionRegion:
    tag = str(tag or "").strip()
    if not tag:
        raise ValueError("[Krea2 CcC Attention Region] tag cannot be empty.")

    x = float(x_percent) / 100.0
    y = float(y_percent) / 100.0
    width = float(width_percent) / 100.0
    height = float(height_percent) / 100.0

    if width <= 0.0 or height <= 0.0:
        raise ValueError("[Krea2 CcC Attention Region] width and height must be greater than zero.")
    if x < 0.0 or y < 0.0 or x >= 1.0 or y >= 1.0:
        raise ValueError("[Krea2 CcC Attention Region] x and y must be inside the 0..100 canvas.")
    if x + width > 1.0 + 1e-9 or y + height > 1.0 + 1e-9:
        raise ValueError(
            "[Krea2 CcC Attention Region] the box must remain completely inside the 0..100 canvas."
        )

    return AttentionRegion(tag=tag, x=x, y=y, width=width, height=height)


def _normalize_target_box(
    box_px: Tuple[float, float, float, float],
    target_w: int,
    target_h: int,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_px
    return (
        x0 / float(target_w),
        y0 / float(target_h),
        x1 / float(target_w),
        y1 / float(target_h),
    )


def resolve_attention_regions(
    regions: Optional[AttentionRegionChain],
    target_w: int,
    target_h: int,
    placement: Dict[str, Any],
) -> Tuple[ResolvedAttentionRegion, ...]:
    """Carry tagged boxes through target pixel transforms before token-grid projection.

    Regions are expressed in normalized target coordinates when Latent content is empty.
    With image content they are expressed relative to the original content image and are
    transformed with stretch/contain/crop. Crop is intentionally strict: if it removes
    any part of any declared attention box, generation fails instead of silently changing
    the attention region.
    """
    if regions is None or not regions.entries:
        return ()

    mode = str((placement or {}).get("mode", "empty"))
    resolved = []

    for region in regions.entries:
        sx0, sy0, sx1, sy1 = region.box_normalized

        if mode in ("empty", "stretch"):
            target_box = (
                sx0 * target_w,
                sy0 * target_h,
                sx1 * target_w,
                sy1 * target_h,
            )

        elif mode in ("contain", "crop"):
            fitted_w, fitted_h = placement["fitted_size"]
            fx0, fy0, fx1, fy1 = (
                sx0 * fitted_w,
                sy0 * fitted_h,
                sx1 * fitted_w,
                sy1 * fitted_h,
            )

            if mode == "contain":
                pad_left, pad_top, _pad_right, _pad_bottom = placement["padding"]
                target_box = (
                    fx0 + pad_left,
                    fy0 + pad_top,
                    fx1 + pad_left,
                    fy1 + pad_top,
                )
            else:
                crop_x, crop_y, crop_w, crop_h = placement["crop"]
                crop_x1 = crop_x + crop_w
                crop_y1 = crop_y + crop_h
                eps = 1e-6
                if (
                    fx0 < crop_x - eps
                    or fy0 < crop_y - eps
                    or fx1 > crop_x1 + eps
                    or fy1 > crop_y1 + eps
                ):
                    raise ValueError(
                        f"[Krea2 CcC Attention Region] crop touches attention region '{region.tag}'. "
                        "Attention boxes must remain completely inside the retained crop. "
                        "Choose another target geometry/content fit or adjust the region."
                    )
                target_box = (
                    fx0 - crop_x,
                    fy0 - crop_y,
                    fx1 - crop_x,
                    fy1 - crop_y,
                )
        else:
            raise ValueError(
                f"[Krea2 CcC Attention Region] Unsupported target placement mode '{mode}'."
            )

        tx0, ty0, tx1, ty1 = target_box
        eps = 1e-5
        if (
            tx0 < -eps
            or ty0 < -eps
            or tx1 > target_w + eps
            or ty1 > target_h + eps
            or tx1 <= tx0
            or ty1 <= ty0
        ):
            raise ValueError(
                f"[Krea2 CcC Attention Region] region '{region.tag}' resolved outside "
                f"the target canvas {target_w}x{target_h}."
            )

        target_box = (
            max(0.0, tx0),
            max(0.0, ty0),
            min(float(target_w), tx1),
            min(float(target_h), ty1),
        )
        resolved.append(
            ResolvedAttentionRegion(
                tag=region.tag,
                source_box_normalized=region.box_normalized,
                target_box_px=target_box,
                target_box_normalized=_normalize_target_box(target_box, target_w, target_h),
            )
        )

    return tuple(resolved)


def resolved_regions_by_tag(
    regions: Tuple[ResolvedAttentionRegion, ...],
) -> Dict[str, ResolvedAttentionRegion]:
    return {region.tag: region for region in regions}
