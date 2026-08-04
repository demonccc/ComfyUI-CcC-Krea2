"""Qwen Vision image preparation logic, encoder introspection, and formatting helpers."""

import torch
import math
from typing import Tuple, Dict, Any
from ccc_krea2.reference_specs import VisionPrepSpec, PreparedVisionImage
from ccc_krea2.geometry import resize_tensor


def introspect_qwen_clip(clip: Any) -> Dict[str, Any]:
    """Extract Qwen Vision encoder parameters from a ComfyUI CLIP instance if available."""
    info: Dict[str, Any] = {
        "encoder_signature": "Qwen-VL",
        "alignment": 28,
        "patch_size": 14,
        "merge_size": 2,
        "min_pixels": 256 * 28 * 28,   # Default ~200k px
        "max_pixels": 1280 * 28 * 28,  # Default ~1M px
    }

    if clip is None:
        return info

    try:
        # Check underlying clip_model or patcher
        cond_stage = getattr(clip, "cond_stage_model", None) or getattr(clip, "patcher", None)
        if cond_stage is not None:
            model = getattr(cond_stage, "model", cond_stage)
            visual = getattr(model, "visual", None) or getattr(model, "image_encoder", None)
            if visual is not None:
                patch_size = getattr(visual, "patch_size", 14)
                merge_size = getattr(visual, "merge_size", 2)
                if isinstance(patch_size, int) and isinstance(merge_size, int):
                    alignment = patch_size * merge_size
                    info["patch_size"] = patch_size
                    info["merge_size"] = merge_size
                    info["alignment"] = alignment
                    if hasattr(visual, "min_pixels") and isinstance(visual.min_pixels, (int, float)):
                        info["min_pixels"] = int(visual.min_pixels)
                    if hasattr(visual, "max_pixels") and isinstance(visual.max_pixels, (int, float)):
                        info["max_pixels"] = int(visual.max_pixels)
                    info["encoder_signature"] = type(visual).__name__
    except Exception:
        pass

    return info


def calculate_qwen_vision_resolution(
    image_h: int,
    image_w: int,
    mode: str,
    min_mp: float,
    max_mp: float,
    fixed_mp: float,
    alignment: int = 28
) -> Tuple[int, int, str, str]:
    """Calculate prepared vision image dimensions based on mode and Qwen encoder constraints.

    Returns:
        (target_h, target_w, resize_direction, resolved_method)
    """
    src_pixels = image_h * image_w
    src_ar = image_w / float(image_h)

    # Calculate native Qwen target area
    native_min_px = 256 * alignment * alignment
    native_max_px = 1280 * alignment * alignment
    target_pixels = max(native_min_px, min(src_pixels, native_max_px))

    if mode == "native":
        # Native Qwen behavior
        target_pixels = max(native_min_px, min(src_pixels, native_max_px))

    elif mode == "adaptive":
        # Constrain native target to user-defined MP range
        user_max_px = max(100_000, int(max_mp * 1_000_000))
        target_pixels = min(target_pixels, user_max_px)

        if min_mp > 0.0:
            user_min_px = int(min_mp * 1_000_000)
            target_pixels = max(target_pixels, user_min_px)

    elif mode == "fixed":
        target_pixels = int(fixed_mp * 1_000_000)

    # Compute height and width preserving aspect ratio
    raw_h = math.sqrt(target_pixels / src_ar)
    raw_w = raw_h * src_ar

    # Align to encoder alignment
    target_h = max(alignment, int(round(raw_h / alignment)) * alignment)
    target_w = max(alignment, int(round(raw_w / alignment)) * alignment)

    if (target_h, target_w) == (image_h, image_w):
        direction = "none"
    elif (target_h * target_w) < src_pixels:
        direction = "downscale"
    else:
        direction = "upscale"

    return target_h, target_w, direction, ("area" if direction == "downscale" else "bicubic")


def prepare_vision_image(
    image: torch.Tensor,
    clip: Any,
    mode: str = "native",
    min_mp: float = 0.0,
    max_mp: float = 1.0,
    fixed_mp: float = 1.0,
    downscale_method: str = "auto",
    upscale_method: str = "auto"
) -> PreparedVisionImage:
    """Prepare a derivative vision image for Qwen Vision while keeping the original image intact."""
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Input format [B, H, W, C]
    bs, ih, iw, c = image.shape

    encoder_info = introspect_qwen_clip(clip)
    alignment = encoder_info["alignment"]

    target_h, target_w, direction, auto_method = calculate_qwen_vision_resolution(
        image_h=ih,
        image_w=iw,
        mode=mode,
        min_mp=min_mp,
        max_mp=max_mp,
        fixed_mp=fixed_mp,
        alignment=alignment
    )

    resolved_method = auto_method
    if direction == "downscale" and downscale_method != "auto":
        resolved_method = downscale_method
    elif direction == "upscale" and upscale_method != "auto":
        resolved_method = upscale_method

    if (ih, iw) == (target_h, target_w):
        vision_image = image
    else:
        vision_image = resize_tensor(image, target_h=target_h, target_w=target_w, method=resolved_method)

    prep_spec = VisionPrepSpec(
        mode=mode,
        semantic_min_mp=min_mp,
        semantic_max_mp=max_mp,
        semantic_fixed_mp=fixed_mp,
        downscale_method_requested=downscale_method,
        upscale_method_requested=upscale_method,
        encoder_signature=encoder_info["encoder_signature"],
        resolved_alignment=alignment,
        resolved_native_limits={"min_pixels": encoder_info["min_pixels"], "max_pixels": encoder_info["max_pixels"]}
    )

    debug_meta = {
        "src_hw": (ih, iw),
        "target_hw": (target_h, target_w),
        "direction": direction,
        "resolved_method": resolved_method,
        "encoder_info": encoder_info
    }

    return PreparedVisionImage(
        original_image=image,
        vision_image=vision_image,
        prep_spec=prep_spec,
        debug_metadata=debug_meta
    )


def format_vision_info(prep_img: PreparedVisionImage) -> str:
    """Format human-readable vision_info string."""
    meta = prep_img.debug_metadata
    spec = prep_img.prep_spec
    enc = meta.get("encoder_info", {})

    ih, iw = meta["src_hw"]
    th, tw = meta["target_hw"]
    src_mp = (ih * iw) / 1_000_000.0
    prep_mp = (th * tw) / 1_000_000.0

    lines = [
        f"Vision Encoder: {enc.get('encoder_signature', 'Qwen-VL')}",
        f"Mode: {spec.mode}",
        f"Source Size: {iw} x {ih}",
        f"Source Area: {src_mp:.3f} MP",
        f"Configured Minimum: {spec.semantic_min_mp:.3f} MP",
        f"Configured Maximum: {spec.semantic_max_mp:.3f} MP",
        f"Prepared Size: {tw} x {th}",
        f"Prepared Area: {prep_mp:.3f} MP",
        f"Encoder Alignment: {spec.resolved_alignment}",
        f"Resize Applied: {'yes' if meta['direction'] != 'none' else 'no'}",
        f"Resize Direction: {meta['direction']}",
        f"Resize Method Requested: {spec.downscale_method_requested if meta['direction'] == 'downscale' else spec.upscale_method_requested}",
        f"Resize Method Resolved: {meta['resolved_method']}",
        "Expected Additional Geometry Adjustment: no"
    ]

    return "\n".join(lines)
