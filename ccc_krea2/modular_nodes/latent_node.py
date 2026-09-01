"""Krea2 CcC Latent node."""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY
from ..geometry import resize_tensor
from ..target_latent import TargetVisionContext, get_image_dims, normalize_vae_output


ASPECT_RATIOS = ("from source", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")
RESOLUTIONS = ("from source", "0.5 MP", "1.0 MP", "1.5 MP", "2.0 MP", "2.5 MP", "3.0 MP", "4.0 MP")


def _align_mp_geometry(megapixels: float, ratio: float) -> Tuple[int, int]:
    pixels = max(1, int(megapixels * 1_000_000))
    raw_h = math.sqrt(pixels / ratio)
    raw_w = raw_h * ratio
    return max(128, int(round(raw_w / 16.0)) * 16), max(128, int(round(raw_h / 16.0)) * 16)


def _resolve_target_geometry(
    target_image: Optional[torch.Tensor],
    aspect_ratio: str,
    resolution: str,
    grid_size_image: Optional[torch.Tensor],
    grid_geometry_image: Optional[torch.Tensor],
) -> Tuple[int, int, str, str]:
    size_image = grid_size_image if grid_size_image is not None else target_image
    geometry_image = grid_geometry_image if grid_geometry_image is not None else target_image

    if aspect_ratio == "from source":
        if geometry_image is None:
            raise ValueError(
                "[CcC Krea2] aspect_ratio='from source' requires grid_geometry_image or target_image."
            )
        gh, gw = get_image_dims(geometry_image)
        ratio = gw / float(gh)
        geometry_label = "grid geometry image" if grid_geometry_image is not None else "target image"
    else:
        rw, rh = (float(part) for part in aspect_ratio.split(":"))
        ratio = rw / rh
        geometry_label = f"explicit {aspect_ratio}"

    if resolution == "from source":
        if size_image is None:
            raise ValueError("[CcC Krea2] resolution='from source' requires grid_size_image or target_image.")
        sh, sw = get_image_dims(size_image)
        target_w, target_h = _align_mp_geometry((sw * sh) / 1_000_000.0, ratio)
        size_label = "grid size image" if grid_size_image is not None else "target image"
    else:
        megapixels = float(resolution.replace(" MP", ""))
        target_w, target_h = _align_mp_geometry(megapixels, ratio)
        size_label = f"explicit {resolution}"

    return target_w, target_h, size_label, geometry_label


def _contain_on_white(
    image: torch.Tensor,
    target_w: int,
    target_h: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    _, src_h, src_w, _ = image.shape
    scale = min(target_w / float(src_w), target_h / float(src_h))
    fitted_w = min(target_w, max(1, int(round(src_w * scale))))
    fitted_h = min(target_h, max(1, int(round(src_h * scale))))

    if (fitted_h, fitted_w) == (src_h, src_w):
        fitted = image
    else:
        fitted = resize_tensor(image, target_h=fitted_h, target_w=fitted_w, method="bicubic")

    left = (target_w - fitted_w) // 2
    right = target_w - fitted_w - left
    top = (target_h - fitted_h) // 2
    bottom = target_h - fitted_h - top

    canvas = F.pad(
        fitted.permute(0, 3, 1, 2),
        (left, right, top, bottom),
        mode="constant",
        value=1.0,
    ).permute(0, 2, 3, 1)

    return canvas.clamp(0.0, 1.0), {
        "source_size": (src_w, src_h),
        "fitted_size": (fitted_w, fitted_h),
        "scale": scale,
        "padding": (left, top, right, bottom),
    }


def _build_target_latent(
    vae: Any,
    target_image: Optional[torch.Tensor],
    target_w: int,
    target_h: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latent_h = target_h // 8
    latent_w = target_w // 8
    placement: Dict[str, Any] = {"mode": "empty"}

    if target_image is None:
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)
    else:
        if vae is None:
            raise ValueError("[CcC Krea2] VAE is required for an image-based target latent.")
        canvas, placement = _contain_on_white(target_image, target_w=target_w, target_h=target_h)
        samples = normalize_vae_output(vae.encode(canvas), batch_size=batch_size)
        placement["mode"] = "contain_on_white"

    return {
        "samples": samples,
        "batch_index": list(range(batch_size)),
        "target_vision_context": TargetVisionContext(include_in_vision="no"),
    }, placement


class CcCKrea2Latent:
    """Build the target latent and preserve its optional semantic reinterpretation metadata."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "latent_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Builds the Krea2 Edit target latent. It owns target image initialization, aspect ratio, resolution, "
        "grid size/geometry sources, batch size, and optional semantic reinterpretation of the target image."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (RESOLUTIONS, {"default": "2.0 MP"}),
                "latent_semantic": ("BOOLEAN", {"default": False}),
                "latent_semantic_instruction": ("STRING", {"multiline": True, "default": ""}),
                "latent_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "target_image": ("IMAGE",),
                "grid_size_image": ("IMAGE",),
                "grid_geometry_image": ("IMAGE",),
            },
        }

    def process(
        self,
        vae,
        aspect_ratio="1:1",
        resolution="2.0 MP",
        latent_semantic=False,
        latent_semantic_instruction="",
        latent_grounding_px=768,
        batch_size=1,
        target_image=None,
        grid_size_image=None,
        grid_geometry_image=None,
    ):
        target_w, target_h, size_label, geometry_label = _resolve_target_geometry(
            target_image=target_image,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            grid_size_image=grid_size_image,
            grid_geometry_image=grid_geometry_image,
        )

        if latent_semantic and target_image is None:
            raise ValueError("[CcC Krea2] latent_semantic requires target_image.")

        latent, placement = _build_target_latent(
            vae=vae,
            target_image=target_image,
            target_w=target_w,
            target_h=target_h,
            batch_size=int(batch_size),
        )

        latent["ccc_krea2_latent_semantic"] = {
            "enabled": bool(latent_semantic),
            "image": target_image if latent_semantic else None,
            "instruction": latent_semantic_instruction.strip() if latent_semantic else "",
            "grounding_px": int(latent_grounding_px),
        }

        lines = [
            "=== Krea2 CcC Latent Report ===",
            f"Target Latent: {'image' if target_image is not None else 'empty'}",
            f"Aspect Ratio: {aspect_ratio}",
            f"Resolution: {resolution}",
            f"Grid Size Source: {size_label}",
            f"Grid Geometry Source: {geometry_label}",
            f"Target Pixel Geometry: {target_w} x {target_h}",
            f"Target Latent Geometry: {target_w // 8} x {target_h // 8}",
            f"Target Placement: {placement}",
            f"Batch Size: {int(batch_size)}",
            f"Latent Semantic: {'enabled' if latent_semantic else 'disabled'}",
        ]
        if latent_semantic:
            lines.append(f"Latent Grounding: {int(latent_grounding_px)} px")
            if latent_semantic_instruction.strip():
                lines.append(f"Latent Semantic Instruction: {latent_semantic_instruction.strip()}")

        latent_info = "\n".join(lines)
        latent["ccc_krea2_latent_info"] = latent_info
        return latent, latent_info
