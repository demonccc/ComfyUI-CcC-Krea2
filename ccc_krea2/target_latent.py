"""Target latent creation and geometry resolution for Krea 2 modular pipeline."""

import torch
import math
from typing import Dict, Any, Tuple, Optional
from ccc_krea2.reference_specs import PreparedVisionImage


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


# Backward-compatibility alias functions
resolve_target_geometry = calculate_target_latent_resolution


def create_target_latent(
    target_latent_content: str = "empty",
    target_geometry: str = "favor_subject",
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    maximum_mp: float = 1.0,
    fixed_mp: float = 2.0,
    fixed_aspect_ratio: str = "1:1",
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
    batch_size: int = 1,
    vae: Any = None,
    **kwargs: Any
) -> Tuple[Dict[str, Any], str]:
    """Backward compatibility alias for build_target_latent."""
    if vae is None:
        raise ValueError("VAE is required for target latent creation.")
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
    vae: Any,
    target_content: str = "empty",
    geometry_mode: str = "favor_subject",
    target_megapixels: float = 1.0,
    fixed_megapixels: float = 2.0,
    aspect_ratio: str = "1:1",
    batch_size: int = 1,
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    **kwargs: Any
) -> Tuple[Dict[str, Any], str]:
    """Build formatted target LATENT dict and latent_info string."""
    if vae is None:
        raise ValueError("VAE is required for target latent creation.")

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
    """Build formatted target LATENT dict and latent_info string."""
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

    # Create empty latent tensor [B, 16, H//8, W//8] for SD3/Krea2
    samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)

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
        f"Source Size: {src_size_str}",
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
