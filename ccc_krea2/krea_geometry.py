"""Shared Krea/Krea2 pixel geometry presets and resolution helpers."""

from typing import Tuple


GEOMETRY_POLICIES = ("nearest_krea_aspect", "preserve_aspect_krea_bounds")

KREA_MIN_DIMENSION = 1024
KREA_MAX_DIMENSION = 2048

KREA_PRESET_GEOMETRIES = {
    "1024 x 1024 | 1:1 | ~1.05 MP": (1024, 1024),
    "1216 x 832 | ~3:2 | ~1.01 MP": (1216, 832),
    "832 x 1216 | ~2:3 | ~1.01 MP": (832, 1216),
    "1536 x 1024 | 3:2 | ~1.57 MP": (1536, 1024),
    "1024 x 1536 | 2:3 | ~1.57 MP": (1024, 1536),
    "1536 x 1152 | 4:3 | ~1.77 MP": (1536, 1152),
    "1152 x 1536 | 3:4 | ~1.77 MP": (1152, 1536),
    "2048 x 1536 | 4:3 | ~3.15 MP": (2048, 1536),
    "1536 x 2048 | 3:4 | ~3.15 MP": (1536, 2048),
    "2048 x 1152 | 16:9 | ~2.36 MP": (2048, 1152),
    "1152 x 2048 | 9:16 | ~2.36 MP": (1152, 2048),
    "2048 x 2048 | 1:1 | ~4.19 MP": (2048, 2048),
}

KREA_PRESET_SIZES = tuple(KREA_PRESET_GEOMETRIES)
DEFAULT_KREA_PRESET_SIZE = "1024 x 1024 | 1:1 | ~1.05 MP"


def align_dimension(value: float) -> int:
    """Align a pixel dimension to Krea's /16 pixel grid."""
    return max(16, int(round(float(value) / 16.0)) * 16)


def align_geometry(width: float, height: float) -> Tuple[int, int]:
    """Align width and height to the shared /16 Krea pixel grid."""
    return align_dimension(width), align_dimension(height)


def preset_geometry(preset_size: str) -> Tuple[int, int]:
    """Resolve a curated Krea preset label into pixel width and height."""
    try:
        return KREA_PRESET_GEOMETRIES[preset_size]
    except KeyError as exc:
        raise ValueError(
            f"[Krea2 CcC] Invalid Krea preset size '{preset_size}'. "
            f"Expected one of: {', '.join(KREA_PRESET_SIZES)}."
        ) from exc


def nearest_krea_geometry(width: int, height: int) -> Tuple[int, int]:
    """Choose the curated Krea preset closest to the requested aspect and size."""
    if width <= 0 or height <= 0:
        raise ValueError("[Krea2 CcC] Source geometry must be greater than zero.")

    source_ratio = width / float(height)

    def score(geometry: Tuple[int, int]) -> Tuple[float, float]:
        candidate_w, candidate_h = geometry
        candidate_ratio = candidate_w / float(candidate_h)
        ratio_error = abs((candidate_ratio / source_ratio) - 1.0)
        size_error = (
            abs(candidate_w - width) / float(max(1, width))
            + abs(candidate_h - height) / float(max(1, height))
        )
        return ratio_error, size_error

    return min(KREA_PRESET_GEOMETRIES.values(), key=score)


def preserve_aspect_krea_bounds(width: int, height: int) -> Tuple[int, int]:
    """Preserve aspect ratio while fitting both axes inside Krea's supported bounds."""
    if width <= 0 or height <= 0:
        raise ValueError("[Krea2 CcC] Source geometry must be greater than zero.")

    lower_scale = max(
        KREA_MIN_DIMENSION / float(width),
        KREA_MIN_DIMENSION / float(height),
    )
    upper_scale = min(
        KREA_MAX_DIMENSION / float(width),
        KREA_MAX_DIMENSION / float(height),
    )

    if lower_scale > upper_scale:
        raise ValueError(
            "[Krea2 CcC] preserve_aspect_krea_bounds cannot fit this aspect ratio "
            f"inside {KREA_MIN_DIMENSION}..{KREA_MAX_DIMENSION} px on both axes. "
            "Use nearest_krea_aspect or choose another source geometry."
        )

    scale = min(max(1.0, lower_scale), upper_scale)
    target_w, target_h = align_geometry(width * scale, height * scale)
    target_w = min(KREA_MAX_DIMENSION, max(KREA_MIN_DIMENSION, target_w))
    target_h = min(KREA_MAX_DIMENSION, max(KREA_MIN_DIMENSION, target_h))
    return target_w, target_h


def resolve_krea_geometry(
    width: int,
    height: int,
    geometry_policy: str,
) -> Tuple[int, int]:
    """Resolve an arbitrary requested geometry through a shared Krea policy."""
    if geometry_policy == "nearest_krea_aspect":
        return nearest_krea_geometry(width, height)
    if geometry_policy == "preserve_aspect_krea_bounds":
        return preserve_aspect_krea_bounds(width, height)

    raise ValueError(
        f"[Krea2 CcC] Invalid geometry_policy '{geometry_policy}'. "
        f"Expected one of: {', '.join(GEOMETRY_POLICIES)}."
    )
