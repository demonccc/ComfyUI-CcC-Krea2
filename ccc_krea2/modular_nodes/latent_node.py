"""Krea2 CcC Latent node."""

from typing import Any, Dict, Optional, Tuple

import torch

from ..constants import NODE_CATEGORY
from ..geometry import resize_tensor
from ..target_latent import TargetVisionContext, get_image_dims, normalize_vae_output


DIMENSION_MODES = ("from_image", "fixed", "preset")
CONTENT_MODES = ("empty", "from_image")
KREA_PRESET_GEOMETRIES = {
    "1024 x 1024 | 1:1 | ~1.05 MP": (1024, 1024),
    "1216 x 832 | ~3:2 | ~1.01 MP": (1216, 832),
    "832 x 1216 | ~2:3 | ~1.01 MP": (832, 1216),
    "1536 x 1024 | 3:2 | ~1.57 MP": (1536, 1024),
    "1024 x 1536 | 2:3 | ~1.57 MP": (1024, 1536),
    "1536 x 1152 | 4:3 | ~1.77 MP": (1536, 1152),
    "1152 x 1536 | 3:4 | ~1.77 MP": (1152, 1536),
    "2048 x 1536 | 4:3 | ~3.15 MP": (2048, 1536),
    "1536 x 2048 | 3:4 | ~3.15 MP": (1536, 2048),
    "2048 x 1152 | 16:9 | ~2.36 MP": (2048, 1152),
    "1152 x 2048 | 9:16 | ~2.36 MP": (1152, 2048),
    "2048 x 2048 | 1:1 | ~4.19 MP": (2048, 2048),
}
KREA_PRESET_SIZES = tuple(KREA_PRESET_GEOMETRIES)
DEFAULT_KREA_PRESET_SIZE = "1024 x 1024 | 1:1 | ~1.05 MP"
IMAGE_FIT_MODES = ("long_edge", "native", "stretch")
RESIZE_METHODS = ("auto", "nearest-exact", "bilinear", "bicubic", "area", "lanczos")


def _align_dimension(value: float) -> int:
    return max(16, int(round(float(value) / 16.0)) * 16)


def _align_geometry(width: float, height: float) -> Tuple[int, int]:
    return _align_dimension(width), _align_dimension(height)


def _preset_geometry(preset_size: str) -> Tuple[int, int]:
    try:
        return KREA_PRESET_GEOMETRIES[preset_size]
    except KeyError as exc:
        raise ValueError(
            f"[Krea2 CcC Edit] Invalid Krea preset size '{preset_size}'. "
            f"Expected one of: {', '.join(KREA_PRESET_SIZES)}."
        ) from exc


def _resolve_dimensions(
    dimensions: str,
    dimensions_image: Optional[torch.Tensor],
    width: int,
    height: int,
    preset_size: str,
) -> Tuple[int, int, str]:
    if dimensions == "from_image":
        if dimensions_image is None:
            raise ValueError("[Krea2 CcC Edit] dimensions='from_image' requires dimensions_image.")
        src_h, src_w = get_image_dims(dimensions_image)
        target_w, target_h = _align_geometry(src_w, src_h)
        return target_w, target_h, f"image dimensions {src_w} x {src_h}"

    if dimensions == "fixed":
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("[Krea2 CcC Edit] fixed width and height must be greater than zero.")
        target_w, target_h = _align_geometry(int(width), int(height))
        return target_w, target_h, f"fixed {int(width)} x {int(height)}"

    if dimensions == "preset":
        target_w, target_h = _preset_geometry(preset_size)
        return target_w, target_h, f"preset {preset_size}"

    raise ValueError(
        f"[Krea2 CcC Edit] Invalid dimensions mode '{dimensions}'. "
        f"Expected from_image, fixed, or preset."
    )


def _center_place(
    image: torch.Tensor,
    target_w: int,
    target_h: int,
    image_fit: str,
    resize_method: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)

    batch, src_h, src_w, channels = image.shape

    if image_fit == "stretch":
        placed = resize_tensor(image, target_h=target_h, target_w=target_w, method=resize_method)
        return placed.clamp(0.0, 1.0), {
            "mode": "stretch",
            "source_size": (src_w, src_h),
            "fitted_size": (target_w, target_h),
            "scale": (target_w / float(src_w), target_h / float(src_h)),
            "crop": (0, 0, target_w, target_h),
            "padding": (0, 0, 0, 0),
            "resize_method": resize_method,
        }

    if image_fit == "long_edge":
        scale = max(target_w, target_h) / float(max(src_w, src_h))
        fitted_w = max(1, int(round(src_w * scale)))
        fitted_h = max(1, int(round(src_h * scale)))
        fitted = (
            image
            if (fitted_w, fitted_h) == (src_w, src_h)
            else resize_tensor(image, target_h=fitted_h, target_w=fitted_w, method=resize_method)
        )
    elif image_fit == "native":
        scale = 1.0
        fitted_w, fitted_h = src_w, src_h
        fitted = image
    else:
        raise ValueError(
            f"[Krea2 CcC Edit] Invalid image_fit '{image_fit}'. "
            f"Expected long_edge, native, or stretch."
        )

    src_x0 = max(0, (fitted_w - target_w) // 2)
    src_y0 = max(0, (fitted_h - target_h) // 2)
    copy_w = min(target_w, fitted_w)
    copy_h = min(target_h, fitted_h)

    dst_x0 = max(0, (target_w - fitted_w) // 2)
    dst_y0 = max(0, (target_h - fitted_h) // 2)

    cropped = fitted[:, src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w, :]
    canvas = torch.ones(
        (batch, target_h, target_w, channels),
        dtype=image.dtype,
        device=image.device,
    )
    canvas[:, dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w, :] = cropped

    return canvas.clamp(0.0, 1.0), {
        "mode": image_fit,
        "source_size": (src_w, src_h),
        "fitted_size": (fitted_w, fitted_h),
        "scale": scale,
        "crop": (src_x0, src_y0, copy_w, copy_h),
        "padding": (
            dst_x0,
            dst_y0,
            target_w - dst_x0 - copy_w,
            target_h - dst_y0 - copy_h,
        ),
        "resize_method": resize_method if image_fit == "long_edge" else "none",
    }


def _build_target_latent(
    vae: Any,
    content: str,
    content_image: Optional[torch.Tensor],
    target_w: int,
    target_h: int,
    image_fit: str,
    resize_method: str,
    batch_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latent_h = target_h // 8
    latent_w = target_w // 8

    if content == "empty":
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)
        placement: Dict[str, Any] = {"mode": "empty"}
    elif content == "from_image":
        if content_image is None:
            raise ValueError("[Krea2 CcC Edit] content='from_image' requires content_image.")
        if vae is None:
            raise ValueError("[Krea2 CcC Edit] VAE is required for image content.")
        canvas, placement = _center_place(
            image=content_image,
            target_w=target_w,
            target_h=target_h,
            image_fit=image_fit,
            resize_method=resize_method,
        )
        samples = normalize_vae_output(vae.encode(canvas), batch_size=batch_size)
    else:
        raise ValueError(
            f"[Krea2 CcC Edit] Invalid content mode '{content}'. Expected empty or from_image."
        )

    return {
        "samples": samples,
        "batch_index": list(range(batch_size)),
        "target_vision_context": TargetVisionContext(include_in_vision="no"),
    }, placement


class CcCKrea2Latent:
    """Build a Krea2 target latent with independent dimension and content controls."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "latent_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Builds the Krea2 Edit target latent. Dimensions can come from an image, fixed width/height, or a "
        "curated Krea-size preset. Content can be empty or image-based. Presets are explicit /16 geometries; "
        "fixed and image-derived dimensions remain available separately."
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
                "content": (CONTENT_MODES, {"default": "empty"}),
                "image_fit": (IMAGE_FIT_MODES, {"default": "long_edge"}),
                "resize_method": (RESIZE_METHODS, {"default": "auto"}),
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
        content="empty",
        image_fit="long_edge",
        resize_method="auto",
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
            image_fit=image_fit,
            resize_method=resize_method,
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
            f"Content: {content}",
            f"Image Fit: {image_fit if content == 'from_image' else '<unused>'}",
            f"Resize Method: {resize_method if content == 'from_image' and image_fit != 'native' else '<unused>'}",
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
        return latent, latent_info
