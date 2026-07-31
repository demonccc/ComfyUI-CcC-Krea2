"""Dual-path reference image preprocessor with process_latent_in support."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch

from .constants import ReferenceRole
from . import geometry
from .grounding import resize_grounding_image


@dataclass
class ReferenceConfig:
    role: ReferenceRole
    image: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor] = None
    boost: float = 1.0
    mask_invert: bool = False
    grounding_preset: str = "balanced"
    grounding_resize_mode: str = "normalize"
    grounding_px: int = 768
    grounding_min_px: int = 128
    grounding_max_px: int = 4096
    grounding_resize_method: str = "auto"
    reference_fit_mode: str = "fit"
    reference_resize_method: str = "auto"


@dataclass
class PreparedReference:
    role: ReferenceRole
    grounding_image: Optional[torch.Tensor]
    vae_latent: Optional[torch.Tensor]
    spatial_attention_mask: Optional[torch.Tensor]
    boost: float
    spatial_hw: Tuple[int, int]
    lat_hw: Tuple[int, int]
    mask_mode: str = "hard"
    ref_fit_meta: Optional[Dict[str, Any]] = None


def prepare_reference(
    config: ReferenceConfig,
    vae: Any,
    model: Any,
    target_h: int,
    target_w: int,
    reference_fit_mode: Optional[str] = None,
    attention_mask_mode: str = "hard"
) -> Optional[PreparedReference]:
    """Prepare reference for dual-path pipeline: Qwen3-VL grounding and VAE reference latent."""
    if config.image is None:
        return None

    if vae is None:
        raise ValueError(f"VAE model is required for encoding reference image (role: {config.role.value}).")

    fit_mode = reference_fit_mode if reference_fit_mode is not None else config.reference_fit_mode

    # 1. Qwen3-VL Grounding Path
    grounding_img = resize_grounding_image(
        image=config.image,
        resize_mode=config.grounding_resize_mode,
        grounding_preset=config.grounding_preset,
        grounding_px=config.grounding_px,
        grounding_min_px=config.grounding_min_px,
        grounding_max_px=config.grounding_max_px,
        resize_method=config.grounding_resize_method
    )

    # 2. VAE Reference Latent Path
    mask_interp = "nearest-exact" if attention_mask_mode == "hard" else "bilinear"
    fitted_img, fitted_mask, ref_fit_meta = geometry.apply_reference_fit_transform(
        image=config.image,
        target_h=target_h,
        target_w=target_w,
        mode=fit_mode,
        mask=config.attention_mask,
        mask_interpolation=mask_interp,
        resize_method=config.reference_resize_method
    )

    # Invert spatial mask if requested
    if fitted_mask is not None and config.mask_invert:
        fitted_mask = 1.0 - fitted_mask

    # VAE encode reference image
    raw_vae_latent = vae.encode(fitted_img)

    lat_h = raw_vae_latent.shape[-2]
    lat_w = raw_vae_latent.shape[-1]
    spatial_h = fitted_img.shape[1]
    spatial_w = fitted_img.shape[2]

    return PreparedReference(
        role=config.role,
        grounding_image=grounding_img,
        vae_latent=raw_vae_latent,
        spatial_attention_mask=fitted_mask,
        boost=config.boost,
        spatial_hw=(spatial_h, spatial_w),
        lat_hw=(lat_h, lat_w),
        mask_mode=attention_mask_mode,
        ref_fit_meta=ref_fit_meta
    )


def _process_latent_in_if_available(model: Any, vae_latent: torch.Tensor) -> torch.Tensor:
    """Pass VAE latent through model.model.process_latent_in if available."""
    if model is None:
        return vae_latent

    # Check for ComfyUI ModelPatcher or underlying model instance
    inner_model = getattr(model, "model", model)
    if hasattr(inner_model, "process_latent_in"):
        return inner_model.process_latent_in(vae_latent)

    return vae_latent
