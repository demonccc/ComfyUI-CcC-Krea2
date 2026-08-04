"""Target latent generation logic separating content and geometry decisions."""

import torch
import math
from typing import Tuple, Dict, Any, Optional
from ccc_krea2.reference_specs import PreparedVisionImage
from ccc_krea2.geometry import apply_sampling_transform


def resolve_target_geometry(
    target_geometry: str,
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    maximum_mp: float = 2.0,
    fixed_mp: float = 1.0,
    fixed_aspect_ratio: str = "1:1",
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
    alignment: int = 16
) -> Tuple[int, int, str]:
    """Resolve target spatial dimensions (target_h, target_w) and warnings."""
    warning = ""

    if target_geometry == "favor_subject":
        if subject_image is None:
            raise ValueError("Target geometry 'favor_subject' requires a valid subject_image.")
        img = subject_image.original_image
        ih, iw = img.shape[1], img.shape[2] if img.ndim == 4 else (img.shape[0], img.shape[1])
        aspect = iw / float(ih)
        target_pixels = ih * iw

    elif target_geometry == "favor_scene":
        if scene_image is None:
            raise ValueError("Target geometry 'favor_scene' requires a valid scene_image.")
        img = scene_image.original_image
        ih, iw = img.shape[1], img.shape[2] if img.ndim == 4 else (img.shape[0], img.shape[1])
        aspect = iw / float(ih)
        target_pixels = ih * iw

    elif target_geometry == "fixed":
        aspect_ratios = {
            "1:1": 1.0,
            "4:3": 4.0 / 3.0,
            "3:4": 3.0 / 4.0,
            "3:2": 3.0 / 2.0,
            "2:3": 2.0 / 3.0,
            "16:9": 16.0 / 9.0,
            "9:16": 9.0 / 16.0,
        }
        if fixed_aspect_ratio == "custom":
            aspect = float(custom_aspect_width) / float(max(1, custom_aspect_height))
        else:
            aspect = aspect_ratios.get(fixed_aspect_ratio, 1.0)

        target_pixels = int(fixed_mp * 1_000_000)

    else:
        raise ValueError(f"Unknown target_geometry '{target_geometry}'. Expected 'favor_subject', 'favor_scene', or 'fixed'.")

    # Cap by maximum_mp
    max_pixels = int(maximum_mp * 1_000_000)
    if target_pixels > max_pixels:
        target_pixels = max_pixels

    if target_pixels > 2_000_000:
        warning = f"Warning: Target area ({target_pixels / 1_000_000:.2f} MP) exceeds recommended Krea 2 range (<= 2.0 MP)."

    # Compute target dimensions matching aspect ratio
    raw_h = math.sqrt(target_pixels / aspect)
    raw_w = raw_h * aspect

    target_h = max(128, int(round(raw_h / alignment)) * alignment)
    target_w = max(128, int(round(raw_w / alignment)) * alignment)

    return target_h, target_w, warning


def create_target_latent(
    vae: Optional[Any],
    target_latent_content: str = "empty",
    target_geometry: str = "fixed",
    subject_image: Optional[PreparedVisionImage] = None,
    scene_image: Optional[PreparedVisionImage] = None,
    maximum_mp: float = 2.0,
    fixed_mp: float = 1.0,
    fixed_aspect_ratio: str = "1:1",
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
    batch_size: int = 1
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Create target LATENT dict and latent_info string."""
    target_h, target_w, warning = resolve_target_geometry(
        target_geometry=target_geometry,
        subject_image=subject_image,
        scene_image=scene_image,
        maximum_mp=maximum_mp,
        fixed_mp=fixed_mp,
        fixed_aspect_ratio=fixed_aspect_ratio,
        custom_aspect_width=custom_aspect_width,
        custom_aspect_height=custom_aspect_height
    )

    latent_h = target_h // 8
    latent_w = target_w // 8

    if target_latent_content == "empty":
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)

    elif target_latent_content in ("subject", "scene"):
        if vae is None:
            raise ValueError(f"VAE is required when target_latent_content is '{target_latent_content}'.")

        ref_prep = subject_image if target_latent_content == "subject" else scene_image
        if ref_prep is None:
            raise ValueError(f"Content source '{target_latent_content}' selected but corresponding image is missing.")

        orig = ref_prep.original_image
        transformed_img, _ = apply_sampling_transform(
            image=orig,
            target_h=target_h,
            target_w=target_w,
            mode="crop"
        )
        encoded = vae.encode(transformed_img)
        if isinstance(encoded, dict) and "samples" in encoded:
            samples = encoded["samples"]
        elif hasattr(encoded, "sample"):
            samples = encoded.sample()
        else:
            samples = encoded

        if samples.shape[0] != batch_size:
            samples = samples[:1].repeat(batch_size, 1, 1, 1)

    else:
        raise ValueError(f"Unknown target_latent_content '{target_latent_content}'. Expected 'empty', 'subject', or 'scene'.")

    target_mp = (target_h * target_w) / 1_000_000.0

    subj_size_str = f"{subject_image.original_image.shape[2]} x {subject_image.original_image.shape[1]}" if subject_image else "none"
    scne_size_str = f"{scene_image.original_image.shape[2]} x {scene_image.original_image.shape[1]}" if scene_image else "none"

    info_lines = [
        f"Latent Content: {target_latent_content.capitalize()}",
        f"Geometry Strategy: {target_geometry.replace('_', ' ').title()}",
        f"Subject Source: {subj_size_str}",
        f"Scene Source: {scne_size_str}",
        f"Maximum MP: {maximum_mp:.3f}",
        f"Target Pixels: {target_w} x {target_h}",
        f"Target MP: {target_mp:.3f}",
        f"Target Latent: {latent_w} x {latent_h}",
        f"Batch Size: {batch_size}"
    ]
    if warning:
        info_lines.append(f"Warning: {warning}")

    return {"samples": samples}, "\n".join(info_lines)
