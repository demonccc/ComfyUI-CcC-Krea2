"""Krea2 CcC Latent node."""

from typing import Any, Dict, Optional, Tuple

import torch

from ..attention_regions import resolve_attention_regions
from ..constants import NODE_CATEGORY
from ..geometry import resize_tensor
from ..target_latent import TargetVisionContext, get_image_dims, normalize_vae_output


DIMENSION_MODES = ("from_image", "fixed", "preset")
CONTENT_MODES = ("empty", "from_image")
GEOMETRY_POLICIES = ("nearest_krea_aspect", "preserve_aspect_krea_bounds")
KREA_MIN_DIMENSION = 1024
KREA_MAX_DIMENSION = 2048
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
CONTENT_FIT_MODES = ("crop", "contain", "stretch")
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


def _nearest_krea_geometry(width: int, height: int) -> Tuple[int, int]:
    source_ratio = width / float(height)

    def score(geometry: Tuple[int, int]) -> Tuple[float, float]:
        candidate_w, candidate_h = geometry
        candidate_ratio = candidate_w / float(candidate_h)
        ratio_error = abs((candidate_ratio / source_ratio) - 1.0)
        size_error = (
            abs(candidate_w - width) / float(max(1, width))
            + abs(candidate_h - height) / float(max(1, height))
        )
        return ratio_error, size_error

    return min(KREA_PRESET_GEOMETRIES.values(), key=score)


def _preserve_aspect_krea_bounds(width: int, height: int) -> Tuple[int, int]:
    lower_scale = max(
        KREA_MIN_DIMENSION / float(width),
        KREA_MIN_DIMENSION / float(height),
    )
    upper_scale = min(
        KREA_MAX_DIMENSION / float(width),
        KREA_MAX_DIMENSION / float(height),
    )

    if lower_scale > upper_scale:
        raise ValueError(
            "[Krea2 CcC Edit] preserve_aspect_krea_bounds cannot fit this aspect ratio "
            f"inside {KREA_MIN_DIMENSION}..{KREA_MAX_DIMENSION} px on both axes. "
            "Use nearest_krea_aspect or choose another source geometry."
        )

    scale = min(max(1.0, lower_scale), upper_scale)
    target_w, target_h = _align_geometry(width * scale, height * scale)
    target_w = min(KREA_MAX_DIMENSION, max(KREA_MIN_DIMENSION, target_w))
    target_h = min(KREA_MAX_DIMENSION, max(KREA_MIN_DIMENSION, target_h))
    return target_w, target_h


def _resolve_krea_geometry(width: int, height: int, geometry_policy: str) -> Tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("[Krea2 CcC Edit] Source geometry must be greater than zero.")

    if geometry_policy == "nearest_krea_aspect":
        return _nearest_krea_geometry(width, height)
    if geometry_policy == "preserve_aspect_krea_bounds":
        return _preserve_aspect_krea_bounds(width, height)

    raise ValueError(
        f"[Krea2 CcC Edit] Invalid geometry_policy '{geometry_policy}'. "
        f"Expected one of: {', '.join(GEOMETRY_POLICIES)}."
    )


def _resolve_dimensions(
    dimensions: str,
    dimensions_image: Optional[torch.Tensor],
    width: int,
    height: int,
    preset_size: str,
    geometry_policy: str,
) -> Tuple[int, int, str]:
    if dimensions == "preset":
        target_w, target_h = _preset_geometry(preset_size)
        return target_w, target_h, f"preset {preset_size}"

    if dimensions == "from_image":
        if dimensions_image is None:
            raise ValueError("[Krea2 CcC Edit] dimensions='from_image' requires dimensions_image.")
        src_h, src_w = get_image_dims(dimensions_image)
        target_w, target_h = _resolve_krea_geometry(src_w, src_h, geometry_policy)
        return (
            target_w,
            target_h,
            f"image {src_w} x {src_h} -> {geometry_policy}",
        )

    if dimensions == "fixed":
        source_w, source_h = int(width), int(height)
        if source_w <= 0 or source_h <= 0:
            raise ValueError("[Krea2 CcC Edit] fixed width and height must be greater than zero.")
        target_w, target_h = _resolve_krea_geometry(source_w, source_h, geometry_policy)
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
        canvas, placement = _fit_content_image(
            image=content_image,
            target_w=target_w,
            target_h=target_h,
            content_fit=content_fit,
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
        "curated Krea-size preset. Fixed and image-derived dimensions resolve through a Krea geometry policy; "
        "presets bypass geometry resolution. Image content can use crop, contain, or stretch before VAE encoding. "
        "Tagged Attention Regions travel through the same target transforms; crop is rejected if it touches any region."
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
                "resize_method": (RESIZE_METHODS, {"default": "auto"}),
                "latent_semantic": ("BOOLEAN", {"default": False}),
                "latent_semantic_instruction": ("STRING", {"multiline": True, "default": ""}),
                "latent_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "dimensions_image": ("IMAGE",),
                "content_image": ("IMAGE",),
                "attention_regions": ("KREA2_ATTENTION_REGION_CHAIN",),
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
        resize_method="auto",
        latent_semantic=False,
        latent_semantic_instruction="",
        latent_grounding_px=768,
        batch_size=1,
        dimensions_image=None,
        content_image=None,
        attention_regions=None,
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
            resize_method=resize_method,
            batch_size=int(batch_size),
        )

        resolved_attention_regions = resolve_attention_regions(
            regions=attention_regions,
            target_w=target_w,
            target_h=target_h,
            placement=placement,
        )
        latent["ccc_krea2_attention_regions"] = resolved_attention_regions

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
            f"Resize Method: {resize_method if content == 'from_image' else '<unused>'}",
            f"Content Placement: {placement}",
            f"Batch Size: {int(batch_size)}",
            f"Latent Semantic: {'enabled' if latent_semantic else 'disabled'}",
            f"Attention Regions: {len(resolved_attention_regions)}",
        ]
        for region in resolved_attention_regions:
            lines.append(
                f"Attention Region '{region.tag}': target_px={tuple(round(v, 2) for v in region.target_box_px)}, "
                f"target_norm={tuple(round(v, 4) for v in region.target_box_normalized)}"
            )
        if dimensions == "preset":
            lines.append(f"Preset: {preset_size}")
        if latent_semantic:
            lines.append(f"Latent Grounding: {int(latent_grounding_px)} px")
            if latent_semantic_instruction.strip():
                lines.append(f"Latent Semantic Instruction: {latent_semantic_instruction.strip()}")

        latent_info = "\n".join(lines)
        latent["ccc_krea2_latent_info"] = latent_info
        return latent, latent_info
