"""Latent output generation for empty, image-based, and inpainting modes."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional


def create_empty_latent(width: int, height: int, batch_size: int = 1) -> Dict[str, Any]:
    """Create an empty Krea 2 compatible latent dictionary (16-channel DiT format)."""
    lat_h = height // 8
    lat_w = width // 8
    samples = torch.zeros((batch_size, 16, lat_h, lat_w), dtype=torch.float32)
    return {"samples": samples}


def create_image_latent(vae: Any, image: torch.Tensor) -> Dict[str, Any]:
    """Encode an image tensor into a VAE latent dictionary."""
    if vae is None:
        raise ValueError("VAE must be provided to create an image latent.")
    if image is None:
        raise ValueError("Image tensor cannot be None when creating an image latent.")

    rgb_image = image[..., :3].clamp(0.0, 1.0)
    latent = vae.encode(rgb_image)
    if isinstance(latent, dict):
        return latent
    return {"samples": latent}


def create_inpaint_latent(
    vae: Any,
    image: torch.Tensor,
    noise_mask: torch.Tensor
) -> Dict[str, Any]:
    """Create an inpainting latent containing the encoded base image and noise mask metadata.

    The inpainting LATENT contains:
    - samples: VAE-encoded base image
    - noise_mask: 2D/3D/4D tensor attached to the latent dict so the sampler noises only editable regions.
    """
    latent = create_image_latent(vae, image)

    if noise_mask is not None:
        m = noise_mask.float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 4:
            m = m.squeeze(1)

        samples = latent["samples"]
        lat_h, lat_w = samples.shape[-2], samples.shape[-1]

        if (m.shape[-2], m.shape[-1]) != (lat_h, lat_w):
            m_4d = m.unsqueeze(1)
            # Use nearest-neighbor to maintain sharp mask boundaries
            m_res = F.interpolate(m_4d, size=(lat_h, lat_w), mode="nearest")
            m = m_res.squeeze(1)

        latent["noise_mask"] = m.clamp(0.0, 1.0)

    return latent
