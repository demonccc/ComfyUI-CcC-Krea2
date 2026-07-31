"""Target LATENT dictionary generator for KSampler integration."""

from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from .geometry import apply_sampling_transform


def generate_krea2_latent(
    model: Any,
    vae: Any,
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
    """Generate model-driven LATENT dictionary returned to KSampler.

    Key principles:
    - KSampler latents are RAW VAE-encoded latents (do NOT apply process_latent_in).
    - Image-based latents and masks match batch_size via repeat/trim.
    - Inpainting latents construct noise_mask.
    """
    if latent_source == "image" or base_image is not None:
        if base_image is None:
            raise ValueError("base_image is required when latent_source is set to an image role.")

        # Apply sampling geometric transformation
        trans_img, trans_mask = apply_sampling_transform(
            image=base_image,
            target_h=height,
            target_w=width,
            mode=sampling_resize_mode,
            mask=inpaint_mask
        )

        # VAE encode base image directly (RAW VAE latent)
        raw_latent = vae.encode(trans_img)

        # Trim or repeat along batch dimension to match batch_size
        raw_latent = _repeat_or_trim_to_batch_size(raw_latent, batch_size)

        latent_dict = {"samples": raw_latent}

        # Process inpainting mask if present
        if inpaint_mask is not None or trans_mask is not None:
            active_mask = trans_mask if trans_mask is not None else inpaint_mask
            processed_mask = _process_inpaint_mask(
                mask=active_mask,
                invert=inpaint_mask_invert,
                grow=inpaint_mask_grow,
                blur=inpaint_mask_blur,
                target_h=height,
                target_w=width
            )
            processed_mask = _repeat_or_trim_to_batch_size(processed_mask, batch_size)
            latent_dict["noise_mask"] = processed_mask

        return latent_dict

    # Default empty latent generation using model/vae
    return _generate_empty_latent(model, vae, width, height, batch_size)


def _generate_empty_latent(
    model: Any,
    vae: Any,
    width: int,
    height: int,
    batch_size: int
) -> Dict[str, Any]:
    """Generate model-driven empty latent using model.get_empty_latent or latent_format."""
    if hasattr(model, "get_empty_latent"):
        try:
            empty_lat = model.get_empty_latent(width, height, batch_size=batch_size)
            if isinstance(empty_lat, dict) and "samples" in empty_lat:
                return empty_lat
            elif isinstance(empty_lat, torch.Tensor):
                return {"samples": empty_lat}
        except Exception:
            pass

    inner_model = getattr(model, "model", model)
    if hasattr(inner_model, "latent_format") and hasattr(inner_model.latent_format, "generate_empty"):
        try:
            lat = inner_model.latent_format.generate_empty(width, height, batch_size=batch_size)
            return {"samples": lat}
        except Exception:
            pass

    # CPU fallback float32
    lat_h = height // 8
    lat_w = width // 8
    samples = torch.zeros((batch_size, 16, lat_h, lat_w), dtype=torch.float32)
    return {"samples": samples}


def _repeat_or_trim_to_batch_size(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Trim or repeat tensor along batch dimension (dim 0)."""
    curr_b = tensor.shape[0]
    if curr_b == batch_size:
        return tensor
    elif curr_b > batch_size:
        return tensor[:batch_size]
    else:
        repeats = (batch_size + curr_b - 1) // curr_b
        tiled = tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))
        return tiled[:batch_size]


def _process_inpaint_mask(
    mask: torch.Tensor,
    invert: bool,
    grow: int,
    blur: int,
    target_h: int,
    target_w: int
) -> torch.Tensor:
    """Process inpainting noise_mask tensor using pure PyTorch operations."""
    if mask.ndim == 2:
        mask_bchw = mask.unsqueeze(0).unsqueeze(0).float()
    elif mask.ndim == 3:
        mask_bchw = mask.unsqueeze(1).float()
    else:
        mask_bchw = mask.float()

    if invert:
        mask_bchw = 1.0 - mask_bchw

    if mask_bchw.shape[-2:] != (target_h, target_w):
        mask_bchw = F.interpolate(mask_bchw, size=(target_h, target_w), mode="bicubic", antialias=True)

    if grow > 0:
        kernel_size = 2 * grow + 1
        mask_bchw = F.max_pool2d(mask_bchw, kernel_size=kernel_size, stride=1, padding=grow)

    if blur > 0:
        kernel_size = 2 * blur + 1
        sigma = blur / 2.0
        coords = torch.arange(kernel_size, dtype=torch.float32) - blur
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = g / g.sum()

        kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
        kernel_y = kernel_1d.view(1, 1, kernel_size, 1)

        mask_bchw = F.conv2d(mask_bchw, kernel_x, padding=(0, blur))
        mask_bchw = F.conv2d(mask_bchw, kernel_y, padding=(blur, 0))

    return mask_bchw.squeeze(1).clamp(0.0, 1.0)
