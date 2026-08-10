"""Target latent creation and geometry resolution for Krea 2 modular pipeline."""

import torch
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from .reference_specs import PreparedVisionImage
from .geometry import resize_tensor


def get_image_dims(image_tensor: torch.Tensor) -> Tuple[int, int]:
    """Helper to extract (height, width) safely from 3D or 4D image tensor."""
    if image_tensor.ndim == 4:
        return int(image_tensor.shape[1]), int(image_tensor.shape[2])
    elif image_tensor.ndim == 3:
        return int(image_tensor.shape[0]), int(image_tensor.shape[1])
    else:
        raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")


def calculate_target_latent_resolution(
    geometry_mode: str = "fixed",
    target_megapixels: float = 1.0,
    fixed_megapixels: float = 2.0,
    aspect_ratio: str = "1:1",
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    **kwargs: Any
) -> Tuple[int, int, str, float, Optional[Tuple[int, int]], list]:
    """Calculate target latent pixel dimensions [H, W] and metadata."""
    if "target_geometry" in kwargs:
        geometry_mode = kwargs["target_geometry"]
    if "maximum_mp" in kwargs:
        target_megapixels = kwargs["maximum_mp"]
    if "fixed_mp" in kwargs:
        fixed_megapixels = kwargs["fixed_mp"]
    if "fixed_aspect_ratio" in kwargs:
        aspect_ratio = kwargs["fixed_aspect_ratio"]
    """Calculate target latent pixel dimensions [H, W] and metadata.

    Returns:
        (target_h, target_w, geometry_source, active_mp, source_dims, warnings)
    """
    warnings = []
    source_dims = None
    geometry_source = "custom_aspect_ratio"

    if geometry_mode == "favor_subject" and subject_image is not None:
        ih, iw = get_image_dims(subject_image.original_image)
        source_dims = (ih, iw)
        src_ar = iw / float(ih)
        geometry_source = "subject_original_image"
        src_mp = (ih * iw) / 1_000_000.0
        active_mp = min(src_mp, target_megapixels)
    elif geometry_mode == "favor_scene" and scene_image is not None:
        ih, iw = get_image_dims(scene_image.original_image)
        source_dims = (ih, iw)
        src_ar = iw / float(ih)
        geometry_source = "scene_original_image"
        src_mp = (ih * iw) / 1_000_000.0
        active_mp = min(src_mp, target_megapixels)
    elif geometry_mode == "fixed" or geometry_mode == "crop_subject":
        geometry_source = "fixed_megapixels"
        active_mp = fixed_megapixels
        if fixed_megapixels > 2.0:
            warnings.append(f"Warning: Fixed MP is set to {fixed_megapixels:.2f} MP, exceeding recommended 2.0 MP limit.")

        # Parse aspect ratio string (e.g. "1:1", "16:9", "custom")
        if ":" in aspect_ratio:
            parts = aspect_ratio.split(":")
            try:
                src_ar = float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                src_ar = 1.0
        else:
            src_ar = 1.0
    else:
        # Fallback to subject or scene if available
        if subject_image is not None:
            ih, iw = get_image_dims(subject_image.original_image)
            source_dims = (ih, iw)
            src_ar = iw / float(ih)
            geometry_source = "subject_original_image"
            active_mp = min((ih * iw) / 1_000_000.0, target_megapixels)
        elif scene_image is not None:
            ih, iw = get_image_dims(scene_image.original_image)
            source_dims = (ih, iw)
            src_ar = iw / float(ih)
            geometry_source = "scene_original_image"
            active_mp = min((ih * iw) / 1_000_000.0, target_megapixels)
        else:
            geometry_source = "fixed_megapixels"
            active_mp = fixed_megapixels
            src_ar = 1.0

    target_pixels = int(active_mp * 1_000_000)
    raw_h = math.sqrt(target_pixels / src_ar)
    raw_w = raw_h * src_ar

    # Align to 16-pixel multiples
    target_h = max(128, int(round(raw_h / 16.0)) * 16)
    target_w = max(128, int(round(raw_w / 16.0)) * 16)

    # For favor_subject / favor_scene, ensure alignment doesn't materially exceed target_megapixels limit
    if geometry_mode in ("favor_subject", "favor_scene") and (target_h * target_w) / 1_000_000.0 > target_megapixels + 0.05:
        alt_h = max(128, int(math.floor(raw_h / 16.0)) * 16)
        alt_w = max(128, int(math.floor(raw_w / 16.0)) * 16)
        if alt_h >= 128 and alt_w >= 128:
            target_h, target_w = alt_h, alt_w

    return target_h, target_w, geometry_source, active_mp, source_dims, warnings


@dataclass(frozen=True)
class TargetContentTransform:
    source_size: Tuple[int, int]              # (W, H)
    crop_rectangle: Tuple[int, int, int, int] # (left, top, crop_w, crop_h)
    target_size: Tuple[int, int]              # (target_w, target_h)
    interpolation: str                         # "bicubic"
    interpolation_applied: bool


def adapt_target_content_image(
    image: torch.Tensor,
    target_w: int,
    target_h: int
) -> Tuple[torch.Tensor, TargetContentTransform]:
    """Deterministically adapt a source pixel image to fill target geometry for target latent initialization."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    bs, src_h, src_w, c = image.shape
    tgt_ar = target_w / float(target_h)
    src_ar = src_w / float(src_h)

    if src_ar > tgt_ar:
        crop_h = src_h
        crop_w = int(round(src_h * tgt_ar))
    else:
        crop_w = src_w
        crop_h = int(round(src_w / tgt_ar))

    left = (src_w - crop_w) // 2
    top = (src_h - crop_h) // 2

    cropped = image[:, top : top + crop_h, left : left + crop_w, :]

    interp_applied = (crop_w, crop_h) != (target_w, target_h)
    if interp_applied:
        adapted = resize_tensor(cropped, target_h=target_h, target_w=target_w, method="bicubic")
    else:
        adapted = cropped

    adapted = torch.clamp(adapted, 0.0, 1.0)

    transform = TargetContentTransform(
        source_size=(src_w, src_h),
        crop_rectangle=(left, top, crop_w, crop_h),
        target_size=(target_w, target_h),
        interpolation="bicubic",
        interpolation_applied=interp_applied
    )
    return adapted, transform


def normalize_vae_output(encoded: Any, batch_size: int) -> torch.Tensor:
    """Normalize VAE encode output into a 4D tensor and expand batch dimension if necessary."""
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

    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError(f"Normalized VAE latent must be a 4D tensor, got shape {getattr(latent, 'shape', None)}")

    b = latent.shape[0]
    if b == 1 and batch_size > 1:
        latent = latent.repeat(batch_size, 1, 1, 1)
    elif b == batch_size:
        pass
    else:
        raise ValueError(f"Encoded latent batch size ({b}) does not match requested batch size ({batch_size}).")

    return latent


# Backward-compatibility alias functions
resolve_target_geometry = calculate_target_latent_resolution


def create_target_latent(
    target_latent_content: str = "empty",
    target_geometry: str = "favor_subject",
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    maximum_mp: float = 2.0,
    fixed_mp: float = 2.0,
    fixed_aspect_ratio: str = "1:1",
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
    batch_size: int = 1,
    vae: Any = None,
    **kwargs: Any
) -> Tuple[Dict[str, Any], str]:
    """Backward compatibility alias for build_target_latent."""
    aspect = f"{custom_aspect_width}:{custom_aspect_height}" if fixed_aspect_ratio == "custom" else fixed_aspect_ratio
    return build_target_latent(
        vae=vae,
        target_content=target_latent_content,
        geometry_mode=target_geometry,
        target_megapixels=maximum_mp,
        fixed_megapixels=fixed_mp,
        aspect_ratio=aspect,
        batch_size=batch_size,
        subject_image=subject_image,
        scene_image=scene_image,
        **kwargs
    )


def build_target_latent(
    vae: Any = None,
    target_content: str = "empty",
    geometry_mode: str = "favor_subject",
    target_megapixels: float = 2.0,
    fixed_megapixels: float = 2.0,
    aspect_ratio: str = "1:1",
    batch_size: int = 1,
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    **kwargs: Any
) -> Tuple[Dict[str, Any], str]:
    """Build formatted target LATENT dict and latent_info string."""
    if "target_latent_content" in kwargs:
        target_content = kwargs["target_latent_content"]
    if "target_geometry" in kwargs:
        geometry_mode = kwargs["target_geometry"]
    if "maximum_mp" in kwargs:
        target_megapixels = kwargs["maximum_mp"]
    if "fixed_mp" in kwargs:
        fixed_megapixels = kwargs["fixed_mp"]
    if "fixed_aspect_ratio" in kwargs:
        aspect_ratio = kwargs["fixed_aspect_ratio"]

    target_h, target_w, geom_src, active_mp, src_dims, warnings = calculate_target_latent_resolution(
        geometry_mode=geometry_mode,
        target_megapixels=target_megapixels,
        fixed_megapixels=fixed_megapixels,
        aspect_ratio=aspect_ratio,
        subject_image=subject_image,
        scene_image=scene_image
    )

    latent_h = target_h // 8
    latent_w = target_w // 8
    vae_applied = False
    transform_info: Optional[TargetContentTransform] = None
    content_src_name = "N/A"

    if target_content == "empty":
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)
    elif target_content == "subject":
        if subject_image is None:
            raise ValueError("Subject image is required when target_content is 'subject'.")
        if vae is None:
            raise ValueError("VAE is required when target_content is 'subject'.")

        orig_img = subject_image.original_image
        adapted_img, transform_info = adapt_target_content_image(orig_img, target_w=target_w, target_h=target_h)
        raw_encoded = vae.encode(adapted_img)
        samples = normalize_vae_output(raw_encoded, batch_size=batch_size)
        vae_applied = True
        content_src_name = "Subject original_image"
    elif target_content == "scene":
        if scene_image is None:
            raise ValueError("Scene image is required when target_content is 'scene'.")
        if vae is None:
            raise ValueError("VAE is required when target_content is 'scene'.")

        orig_img = scene_image.original_image
        adapted_img, transform_info = adapt_target_content_image(orig_img, target_w=target_w, target_h=target_h)
        raw_encoded = vae.encode(adapted_img)
        samples = normalize_vae_output(raw_encoded, batch_size=batch_size)
        vae_applied = True
        content_src_name = "Scene original_image"
    else:
        raise ValueError(f"Unknown target_content mode: '{target_content}'. Expected 'empty', 'subject', or 'scene'.")

    latent_dict = {
        "samples": samples,
        "batch_index": list(range(batch_size)),
    }

    # Format latent_info string
    src_size_str = f"{src_dims[1]} x {src_dims[0]}" if src_dims else "N/A"
    actual_mp = (target_h * target_w) / 1_000_000.0

    lines = [
        f"Latent Content: {target_content}",
        f"Geometry Strategy: {geometry_mode}",
        f"Geometry Source: {geom_src}",
        f"Content Source: {content_src_name}",
        f"Content Source Size: {f'{transform_info.source_size[0]} x {transform_info.source_size[1]}' if transform_info else src_size_str}",
        f"Content Crop Rectangle: {transform_info.crop_rectangle if transform_info else 'N/A'}",
        f"Content Target Size: {f'{transform_info.target_size[0]} x {transform_info.target_size[1]}' if transform_info else f'{target_w} x {target_h}'}",
        f"Content Interpolation: {transform_info.interpolation if transform_info else 'none'}",
        f"VAE Encode Applied: {'yes' if vae_applied else 'no'}",
    ]

    if geometry_mode in ("favor_subject", "favor_scene"):
        lines.append(f"Maximum MP: {target_megapixels:.2f} MP")
    elif geometry_mode == "fixed":
        lines.append(f"Fixed MP: {fixed_megapixels:.2f} MP")
        lines.append(f"Fixed Aspect Ratio: {aspect_ratio}")

    lines.extend([
        f"Target Pixel Size: {target_w} x {target_h}",
        f"Target MP: {actual_mp:.3f} MP",
        f"Target Latent Size: {latent_w} x {latent_h}",
        f"Batch Size: {batch_size}",
        f"Warnings: {'; '.join(warnings) if warnings else 'none'}"
    ])

    latent_info = "\n".join(lines)
    return latent_dict, latent_info
