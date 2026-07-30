"""Reference image & attention mask preparation and VAE encoding."""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
import torch

from .constants import ReferenceRole
from .geometry import apply_reference_fit_transform
from .masks import process_attention_mask
from .grounding import resize_grounding_image


@dataclass
class ReferenceConfig:
    role: ReferenceRole
    image: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    boost: float = 1.0
    mask_invert: bool = False
    grounding_preset: str = "balanced"
    grounding_resize_mode: str = "normalize"
    grounding_px: int = 768
    grounding_min_px: int = 512
    grounding_max_px: int = 1024
    reference_fit_mode: str = "fit"


@dataclass
class PreparedReference:
    role: ReferenceRole
    grounding_image: torch.Tensor
    pixel_image: torch.Tensor
    vae_latent: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    boost: float = 1.0
    attention_token_grid: Tuple[int, int] = (0, 0)
    reference_fit_mode: str = "fit"


def prepare_reference(
    config: ReferenceConfig,
    vae: Any,
    target_h: int,
    target_w: int,
    reference_fit_mode: str = "fit",
    attention_mask_mode: str = "hard",
    patch_size: int = 2
) -> PreparedReference:
    """Prepares a single reference image through both Qwen3-VL grounding and VAE reference paths.

    1. Grounding Path: Resizes original image using grounding controls for Qwen3-VL.
    2. VAE Path: Resizes original image to target output pixel grid (target_h x target_w)
       using reference_fit_mode (fit, crop), applies identical transformation
       to the attention mask, and encodes the image with VAE.
    """
    if config.image is None:
        raise ValueError(f"Reference image for role '{config.role.value}' cannot be None.")

    # 1. Qwen3-VL Grounding Path
    grounding_img = resize_grounding_image(
        image=config.image,
        resize_mode=config.grounding_resize_mode,
        grounding_px=config.grounding_px,
        grounding_min_px=config.grounding_min_px,
        grounding_max_px=config.grounding_max_px,
        grounding_preset=config.grounding_preset
    )

    # 2. VAE Reference Path (uses reference_fit_mode: fit or crop)
    pixel_img, transformed_attn_mask = apply_reference_fit_transform(
        image=config.image,
        target_h=target_h,
        target_w=target_w,
        mode=reference_fit_mode,
        mask=config.attention_mask,
        mask_interpolation="nearest" if attention_mask_mode == "hard" else "bicubic"
    )

    # Encode transformed pixel image with VAE if VAE is provided
    vae_latent = None
    if vae is not None:
        rgb_img = pixel_img[..., :3].clamp(0.0, 1.0)
        vae_latent = vae.encode(rgb_img)

    # Compute reference token grid dimensions based on latent sample spatial size
    gh, gw = 0, 0
    if vae_latent is not None and isinstance(vae_latent, dict) and "samples" in vae_latent:
        lat_samples = vae_latent["samples"]
        gh = lat_samples.shape[-2] // patch_size
        gw = lat_samples.shape[-1] // patch_size
    else:
        gh = target_h // 16
        gw = target_w // 16

    # 3. Reference Attention Mask Processing
    processed_attn_mask = None
    if transformed_attn_mask is not None:
        processed_attn_mask = process_attention_mask(
            mask=transformed_attn_mask,
            invert=config.mask_invert,
            mode=attention_mask_mode,
            token_grid=(gh, gw)
        )

    return PreparedReference(
        role=config.role,
        grounding_image=grounding_img,
        pixel_image=pixel_img,
        vae_latent=vae_latent,
        attention_mask=processed_attn_mask,
        boost=config.boost,
        attention_token_grid=(gh, gw),
        reference_fit_mode=reference_fit_mode
    )
