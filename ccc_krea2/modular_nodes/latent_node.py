"""Krea2 CcC Latent node."""

from typing import Any, Dict, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from ..geometry import resize_tensor
from ..krea_geometry import (
    DEFAULT_KREA_PRESET_SIZE,
    GEOMETRY_POLICIES,
    KREA_PRESET_SIZES,
    preset_geometry,
    resolve_krea_geometry,
)
from ..target_latent import get_image_dims, normalize_vae_output


DIMENSION_MODES = ("from_image", "fixed", "preset")
CONTENT_MODES = ("empty", "from_image")
CONTENT_FIT_MODES = ("crop", "contain", "stretch")


def _resolve_dimensions(
    dimensions: str,
    dimensions_image: Optional[torch.Tensor],
    width: int,
    height: int,
    preset_size: str,
    geometry_policy: str,
) -> Tuple[int, int, str]:
    if dimensions == "preset":
        target_w, target_h = preset_geometry(preset_size)
        return target_w, target_h, f"preset {preset_size}"

    if dimensions == "from_image":
        if dimensions_image is None:
            raise ValueError("[Krea2 CcC Edit] dimensions='from_image' requires dimensions_image.")
        src_h, src_w = get_image_dims(dimensions_image)
        target_w, target_h = resolve_krea_geometry(src_w, src_h, geometry_policy)
        return (
            target_w,
            target_h,
            f"image {src_w} x {src_h} -> {geometry_policy}",
        )

    if dimensions == "fixed":
        source_w, source_h = int(width), int(height)
        if source_w <= 0 or source_h <= 0:
            raise ValueError("[Krea2 CcC Edit] fixed width and height must be greater than zero.")
        target_w, target_h = resolve_krea_geometry(source_w, source_h, geometry_policy)
        return (
            target_w,
            target_h,
            f"fixed {source_w} x {source_h} -> {geometry_policy}",
        )

    raise ValueError(
        f"[Krea2 CcC Edit] Invalid dimensions mode '{dimensions}'. "
        f"Expected from_image, fixed, or preset."
    )


def _fit_content_image(
    image: torch.Tensor,
    target_w: int,
    target_h: int,
    content_fit: str,
    resize_method: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)

    batch, src_h, src_w, channels = image.shape

    if content_fit == "stretch":
        fitted = resize_tensor(image, target_h=target_h, target_w=target_w, method=resize_method)
        return fitted.clamp(0.0, 1.0), {
            "mode": "stretch",
            "source_size": (src_w, src_h),
            "fitted_size": (target_w, target_h),
            "crop": (0, 0, target_w, target_h),
            "padding": (0, 0, 0, 0),
            "resize_method": resize_method,
        }

    if content_fit == "crop":
        scale = max(target_w / float(src_w), target_h / float(src_h))
        fitted_w = max(target_w, int(round(src_w * scale)))
        fitted_h = max(target_h, int(round(src_h * scale)))
        fitted = resize_tensor(image, target_h=fitted_h, target_w=fitted_w, method=resize_method)
        src_x0 = max(0, (fitted_w - target_w) // 2)
        src_y0 = max(0, (fitted_h - target_h) // 2)
        cropped = fitted[:, src_y0 : src_y0 + target_h, src_x0 : src_x0 + target_w, :]
        return cropped.clamp(0.0, 1.0), {
            "mode": "crop",
            "source_size": (src_w, src_h),
            "fitted_size": (fitted_w, fitted_h),
            "crop": (src_x0, src_y0, target_w, target_h),
            "padding": (0, 0, 0, 0),
            "resize_method": resize_method,
        }

    if content_fit == "contain":
        scale = min(target_w / float(src_w), target_h / float(src_h))
        fitted_w = max(1, min(target_w, int(round(src_w * scale))))
        fitted_h = max(1, min(target_h, int(round(src_h * scale))))
        fitted = resize_tensor(image, target_h=fitted_h, target_w=fitted_w, method=resize_method)
        dst_x0 = (target_w - fitted_w) // 2
        dst_y0 = (target_h - fitted_h) // 2
        canvas = torch.ones(
            (batch, target_h, target_w, channels),
            dtype=image.dtype,
            device=image.device,
        )
        canvas[:, dst_y0 : dst_y0 + fitted_h, dst_x0 : dst_x0 + fitted_w, :] = fitted
        return canvas.clamp(0.0, 1.0), {
            "mode": "contain",
            "source_size": (src_w, src_h),
            "fitted_size": (fitted_w, fitted_h),
            "crop": (0, 0, fitted_w, fitted_h),
            "padding": (
                dst_x0,
                dst_y0,
                target_w - dst_x0 - fitted_w,
                target_h - dst_y0 - fitted_h,
            ),
            "resize_method": resize_method,
        }

    raise ValueError(
        f"[Krea2 CcC Edit] Invalid content_fit '{content_fit}'. "
        f"Expected one of: {', '.join(CONTENT_FIT_MODES)}."
    )

def _build_target_latent(
    vae: Any,
    content: str,
    content_image: Optional[torch.Tensor],
    target_w: int,
    target_h: int,
    content_fit: str,
    batch_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latent_h = target_h // 8
    latent_w = target_w // 8

    preview_image = None
    if content == "empty":
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)
        placement: Dict[str, Any] = {"mode": "empty"}
        preview_image = torch.full((batch_size, target_h, target_w, 3), 0.5, dtype=torch.float32)
    elif content == "from_image":
        if content_image is None:
            raise ValueError("[Krea2 CcC Edit] content='from_image' requires content_image.")
        if vae is None:
            raise ValueError("[Krea2 CcC Edit] VAE is required for image content.")
        canvas, placement = _fit_content_image(
            image=content_image,
            target_w=target_w,
            target_h=target_h,
            content_fit=content_fit,
            resize_method="bicubic",
        )
        samples = normalize_vae_output(vae.encode(canvas), batch_size=batch_size)
        preview_image = canvas
    else:
        raise ValueError(
            f"[Krea2 CcC Edit] Invalid content mode '{content}'. Expected empty or from_image."
        )

    return {
        "samples": samples,
        "ccc_krea2_target_preview_image": preview_image,
        "batch_index": list(range(batch_size)),
    }, placement


class CcCKrea2Latent:
    """Build a Krea2 target latent with independent dimension and content controls."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("LATENT", "STRING", "IMAGE")
    RETURN_NAMES = ("latent", "latent_info", "target_image")
    FUNCTION = "process"
    DESCRIPTION = (
        "Builds the Krea2 Edit target latent. Dimensions can come from an image, fixed width/height, or a "
        "curated Krea-size preset. Fixed and image-derived dimensions resolve through a Krea geometry policy; "
        "presets bypass geometry resolution. Image content can use crop, contain, or stretch before VAE encoding. "
        "The target_image output matches the final target pixel grid and can be used to paint aligned attention masks."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "dimensions": (DIMENSION_MODES, {"default": "preset"}),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                    },
                ),
                "preset_size": (KREA_PRESET_SIZES, {"default": DEFAULT_KREA_PRESET_SIZE}),
                "geometry_policy": (GEOMETRY_POLICIES, {"default": "preserve_aspect_krea_bounds"}),
                "content": (CONTENT_MODES, {"default": "empty"}),
                "content_fit": (CONTENT_FIT_MODES, {"default": "crop"}),
                "latent_semantic": ("BOOLEAN", {"default": False}),
                "latent_semantic_instruction": ("STRING", {"multiline": True, "default": ""}),
                "latent_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "dimensions_image": ("IMAGE",),
                "content_image": ("IMAGE",),
            },
        }

    def process(
        self,
        vae,
        dimensions="preset",
        width=1024,
        height=1024,
        preset_size=DEFAULT_KREA_PRESET_SIZE,
        geometry_policy="preserve_aspect_krea_bounds",
        content="empty",
        content_fit="crop",
        latent_semantic=False,
        latent_semantic_instruction="",
        latent_grounding_px=768,
        batch_size=1,
        dimensions_image=None,
        content_image=None,
    ):
        target_w, target_h, dimensions_label = _resolve_dimensions(
            dimensions=dimensions,
            dimensions_image=dimensions_image,
            width=width,
            height=height,
            preset_size=preset_size,
            geometry_policy=geometry_policy,
        )

        if latent_semantic and (content != "from_image" or content_image is None):
            raise ValueError(
                "[Krea2 CcC Edit] latent_semantic requires content='from_image' and content_image."
            )

        latent, placement = _build_target_latent(
            vae=vae,
            content=content,
            content_image=content_image,
            target_w=target_w,
            target_h=target_h,
            content_fit=content_fit,
            batch_size=int(batch_size),
        )

        latent["ccc_krea2_latent_semantic"] = {
            "enabled": bool(latent_semantic),
            "image": content_image if latent_semantic else None,
            "instruction": latent_semantic_instruction.strip() if latent_semantic else "",
            "grounding_px": int(latent_grounding_px),
        }

        lines = [
            "=== Krea2 CcC Latent Report ===",
            f"Dimensions Mode: {dimensions}",
            f"Dimensions Source: {dimensions_label}",
            f"Target Pixel Geometry: {target_w} x {target_h}",
            f"Target Latent Geometry: {target_w // 8} x {target_h // 8}",
            f"Geometry Policy: {geometry_policy if dimensions != 'preset' else '<preset>'}",
            f"Content: {content}",
            f"Content Fit: {content_fit if content == 'from_image' else '<unused>'}",
            f"Resize Method: {'bicubic' if content == 'from_image' else '<unused>'}",
            f"Content Placement: {placement}",
            f"Batch Size: {int(batch_size)}",
            f"Latent Semantic: {'enabled' if latent_semantic else 'disabled'}",
        ]
        if dimensions == "preset":
            lines.append(f"Preset: {preset_size}")
        if latent_semantic:
            lines.append(f"Latent Grounding: {int(latent_grounding_px)} px")
            if latent_semantic_instruction.strip():
                lines.append(f"Latent Semantic Instruction: {latent_semantic_instruction.strip()}")

        latent_info = "\n".join(lines)
        latent["ccc_krea2_latent_info"] = latent_info
        return latent, latent_info, latent["ccc_krea2_target_preview_image"]
