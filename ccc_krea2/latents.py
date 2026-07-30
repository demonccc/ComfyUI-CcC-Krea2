"""Latent generation utilities using model-driven empty latents and batch_size repetition."""

from typing import Dict, Any, Optional
import torch

from .geometry import apply_sampling_transform
from .masks import process_inpaint_mask
from .references import _process_latent_in_if_available


def generate_krea2_latent(
    model: Any,
    vae: Optional[Any],
    width: int,
    height: int,
    batch_size: int = 1,
    latent_source: str = "empty",
    base_image: Optional[torch.Tensor] = None,
    inpaint_mask: Optional[torch.Tensor] = None,
    inpaint_mask_invert: bool = False,
    inpaint_mask_grow: int = 0,
    inpaint_mask_blur: int = 0,
    sampling_resize_mode: str = "fit"
) -> Dict[str, Any]:
    """Generate KSampler target LATENT dictionary."""
    if latent_source == "empty" or base_image is None:
        return _generate_empty_latent(model, width, height, batch_size)

    if vae is None:
        raise ValueError("VAE model is required to encode base image for non-empty latent generation.")

    # 1. Transform base image and mask to target width & height
    transformed_img, transformed_mask = apply_sampling_transform(
        image=base_image,
        target_h=height,
        target_w=width,
        mode=sampling_resize_mode,
        mask=inpaint_mask
    )

    # 2. VAE encode base image
    raw_latent = vae.encode(transformed_img)

    # 3. Apply model.model.process_latent_in
    lat = _process_latent_in_if_available(model, raw_latent)

    # 4. Honor requested batch_size (tile/repeat along batch dimension if needed)
    if lat.shape[0] < batch_size:
        repeat_factor = batch_size // lat.shape[0]
        lat = lat.repeat(repeat_factor, 1, 1, 1)

    latent_dict: Dict[str, Any] = {"samples": lat}

    # 5. Process inpaint noise mask if provided
    if transformed_mask is not None:
        processed_mask = process_inpaint_mask(
            mask=transformed_mask,
            invert=inpaint_mask_invert,
            grow=inpaint_mask_grow,
            blur=inpaint_mask_blur
        )
        if processed_mask.shape[0] < batch_size:
            processed_mask = processed_mask.repeat(batch_size // processed_mask.shape[0], 1, 1)
        latent_dict["noise_mask"] = processed_mask

    return latent_dict


def _generate_empty_latent(
    model: Any,
    width: int,
    height: int,
    batch_size: int
) -> Dict[str, Any]:
    """Generate model-driven empty latent using ComfyUI latent format."""
    # Attempt ComfyUI model.get_empty_latent API
    if model is not None and hasattr(model, "get_empty_latent"):
        try:
            return model.get_empty_latent(width, height, batch_size=batch_size)
        except Exception:
            pass

    # Fallback to model.model.latent_format or SD3/Krea2 16-channel format
    lat_h = height // 8
    lat_w = width // 8

    device = torch.device("cpu")
    dtype = torch.float32

    if model is not None:
        inner_model = getattr(model, "model", model)
        if hasattr(inner_model, "load_device"):
            device = inner_model.load_device
        if hasattr(inner_model, "model_dtype"):
            dtype = inner_model.model_dtype
        elif hasattr(inner_model, "dtype"):
            dtype = inner_model.dtype

        if hasattr(inner_model, "latent_format") and hasattr(inner_model.latent_format, "generate_empty"):
            empty_tensor = inner_model.latent_format.generate_empty(batch_size, 16, lat_h, lat_w, device, dtype)
            return {"samples": empty_tensor}

    empty_tensor = torch.zeros((batch_size, 16, lat_h, lat_w), device=device, dtype=dtype)
    return {"samples": empty_tensor}
