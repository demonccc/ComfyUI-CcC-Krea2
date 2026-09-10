"""Native visual-reference sizing for Krea2 Edit."""

import math

from .. import edit_engine
from ..krea2edit_geometry import ResolvedGeometry, resolve_krea2edit_geometry as _canonical_resolver


def resolve_krea2_reference_geometry(
    src_h: int,
    src_w: int,
    tgt_h: int,
    tgt_w: int,
    fit_mode: str = "auto",
) -> ResolvedGeometry:
    """Add a native-size reference mode while delegating all other modes to the canonical resolver."""
    if fit_mode != "native":
        return _canonical_resolver(
            src_h=src_h,
            src_w=src_w,
            tgt_h=tgt_h,
            tgt_w=tgt_w,
            fit_mode=fit_mode,
        )

    vae_input_w = max(16, int(math.ceil(src_w / 16.0)) * 16)
    vae_input_h = max(16, int(math.ceil(src_h / 16.0)) * 16)
    vae_lat_w = vae_input_w // 8
    vae_lat_h = vae_input_h // 8
    tgt_lat_w = tgt_w // 8
    tgt_lat_h = tgt_h // 8

    return ResolvedGeometry(
        mode_requested="native",
        mode_resolved="native",
        source_size=(src_w, src_h),
        crop_rectangle=(0, 0, src_w, src_h),
        vae_input_pixel_size=(vae_input_w, vae_input_h),
        vae_latent_grid_size=(vae_lat_w, vae_lat_h),
        target_grid_size=(tgt_lat_w, tgt_lat_h),
        centered_fractional_offset=(
            (tgt_lat_h - vae_lat_h) / 2.0,
            (tgt_lat_w - vae_lat_w) / 2.0,
        ),
        interpolation_method="pad" if (vae_input_w, vae_input_h) != (src_w, src_h) else "none",
        whether_interpolation_occurred=False,
    )


def install_krea2_reference_fit() -> None:
    """Install native reference sizing into the orchestrator runtime."""
    edit_engine.resolve_krea2edit_geometry = resolve_krea2_reference_geometry
