"""Active Krea2 target image dimension and VAE output helpers.

Target geometry and image fitting belong to modular_nodes.latent_node.
Historical subject/scene/megapixel latent builders were removed.
"""

from typing import Any, Tuple

import torch


def get_image_dims(image_tensor: torch.Tensor) -> Tuple[int, int]:
    """Helper to extract (height, width) safely from 3D or 4D image tensor."""
    if image_tensor.ndim == 4:
        return int(image_tensor.shape[1]), int(image_tensor.shape[2])
    elif image_tensor.ndim == 3:
        return int(image_tensor.shape[0]), int(image_tensor.shape[1])
    else:
        raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")

def normalize_vae_output(encoded: Any, batch_size: int) -> torch.Tensor:
    """Normalize VAE encode output into a 4D or 5D tensor and expand batch dimension if necessary."""
    if isinstance(encoded, torch.Tensor):
        latent = encoded
    elif isinstance(encoded, dict) and "samples" in encoded:
        latent = encoded["samples"]
    elif hasattr(encoded, "samples"):
        latent = getattr(encoded, "samples")
    elif hasattr(encoded, "sample") and callable(getattr(encoded, "sample")):
        latent = encoded.sample()
    else:
        raise ValueError(f"Unsupported VAE return format: {type(encoded)}")

    if not isinstance(latent, torch.Tensor) or latent.ndim not in (4, 5):
        raise ValueError(f"Normalized VAE latent must be a 4D or 5D tensor, got shape {getattr(latent, 'shape', None)}")

    b = latent.shape[0]
    if b == 1 and batch_size > 1:
        repeats = [batch_size] + [1] * (latent.ndim - 1)
        latent = latent.repeat(*repeats)
    elif b == batch_size:
        pass
    else:
        raise ValueError(f"Encoded latent batch size ({b}) does not match requested batch size ({batch_size}).")

    return latent

