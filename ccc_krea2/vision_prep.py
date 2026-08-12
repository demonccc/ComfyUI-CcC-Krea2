import torch
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from .reference_specs import VisionPrepSpec, PreparedVisionImage
from .geometry import resize_tensor


@dataclass(frozen=True)
class QwenVisionEncoderConfig:
    encoder_signature: str
    patch_size: int
    merge_size: int
    factor: int
    min_pixels: int
    max_pixels: int
    interpolation: str
    introspection_status: str
    introspection_warnings: Tuple[str, ...]
    value_sources: Dict[str, str]


def resolve_qwen_encoder_config(clip: Any) -> QwenVisionEncoderConfig:
    """Resolve Qwen Vision encoder configuration with strict fallback hierarchy and explicit introspection warning tracking."""
    fallback_patch_size = 16
    fallback_merge_size = 2
    fallback_min_pixels = 3136
    fallback_max_pixels = 12845056

    sources = {
        "encoder_signature": "fallback",
        "patch_size": "fallback",
        "merge_size": "fallback",
        "factor": "fallback",
        "min_pixels": "fallback",
        "max_pixels": "fallback",
        "interpolation": "fallback",
    }

    sig = "Qwen3-VL"
    p_size = fallback_patch_size
    m_size = fallback_merge_size
    min_px = fallback_min_pixels
    max_px = fallback_max_pixels
    interp = "bilinear"  # Section 5.1: Native Qwen fallback interpolation is bilinear

    warnings: List[str] = []
    introspection_attempted = False
    introspection_succeeded = False

    if clip is not None:
        introspection_attempted = True
        try:
            cond_stage = getattr(clip, "cond_stage_model", None) or getattr(clip, "patcher", None)
            if cond_stage is not None:
                model = getattr(cond_stage, "model", cond_stage)
                visual = (
                    getattr(model, "visual", None)
                    or getattr(model, "image_encoder", None)
                    or getattr(model, "vision_model", None)
                )
                if visual is not None:
                    sig = type(visual).__name__
                    sources["encoder_signature"] = "introspected"
                    introspection_succeeded = True

                    found_p = getattr(visual, "patch_size", None)
                    if isinstance(found_p, int) and found_p > 0:
                        p_size = found_p
                        sources["patch_size"] = "introspected"

                    found_m = getattr(visual, "spatial_merge_size", None) or getattr(visual, "merge_size", None)
                    if isinstance(found_m, int) and found_m > 0:
                        m_size = found_m
                        sources["merge_size"] = "introspected"

                    found_min = getattr(visual, "min_pixels", None)
                    if isinstance(found_min, (int, float)) and found_min > 0:
                        min_px = int(found_min)
                        sources["min_pixels"] = "introspected"

                    found_max = getattr(visual, "max_pixels", None)
                    if isinstance(found_max, (int, float)) and found_max > 0:
                        max_px = int(found_max)
                        sources["max_pixels"] = "introspected"
                else:
                    warnings.append("CLIP model object has no visual / image_encoder attribute.")
            else:
                warnings.append("CLIP object has no cond_stage_model or patcher attribute.")
        except Exception as e:
            warnings.append(f"CLIP vision encoder introspection failed ({type(e).__name__}): {e}")

    factor = p_size * m_size
    sources["factor"] = "derived" if (sources["patch_size"] == "introspected" or sources["merge_size"] == "introspected") else "fallback"

    if introspection_succeeded:
        status = "succeeded" if not warnings else "partial"
    elif introspection_attempted:
        status = "fallback_only"
    else:
        status = "none"

    return QwenVisionEncoderConfig(
        encoder_signature=sig,
        patch_size=p_size,
        merge_size=m_size,
        factor=factor,
        min_pixels=min_px,
        max_pixels=max_px,
        interpolation=interp,
        introspection_status=status,
        introspection_warnings=tuple(warnings),
        value_sources=sources,
    )


def calculate_native_qwen_geometry(height: int, width: int, config: QwenVisionEncoderConfig) -> Tuple[int, int]:
    """Exact process_qwen2vl_images geometry calculation."""
    factor = config.factor
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)

    current_pixels = h_bar * w_bar
    original_pixels = height * width

    if current_pixels > config.max_pixels:
        beta = math.sqrt(original_pixels / float(config.max_pixels))
        target_h = max(factor, int(math.floor(height / beta / factor)) * factor)
        target_w = max(factor, int(math.floor(width / beta / factor)) * factor)
    elif current_pixels < config.min_pixels:
        beta = math.sqrt(float(config.min_pixels) / original_pixels)
        target_h = max(factor, int(math.ceil(height * beta / factor)) * factor)
        target_w = max(factor, int(math.ceil(width * beta / factor)) * factor)
    else:
        target_h = h_bar
        target_w = w_bar

    return target_h, target_w


def calculate_qwen_vision_resolution(
    image_h: int,
    image_w: int,
    mode: str,
    min_mp: float,
    max_mp: float,
    fixed_mp: float,
    config: QwenVisionEncoderConfig
) -> Tuple[int, int, int, int, str, str, str]:
    """Calculate prepared vision image dimensions based on mode and Qwen encoder constraints."""
    native_h, native_w = calculate_native_qwen_geometry(image_h, image_w, config)
    factor = config.factor

    if mode == "native":
        prep_h, prep_w = native_h, native_w
    elif mode == "adaptive":
        target_pixels = native_h * native_w
        if max_mp > 0.0:
            user_max_px = int(max_mp * 1_000_000)
            target_pixels = min(target_pixels, user_max_px)
        if min_mp > 0.0:
            user_min_px = int(min_mp * 1_000_000)
            target_pixels = max(target_pixels, user_min_px)

        src_ar = image_w / float(image_h)
        raw_h = math.sqrt(target_pixels / src_ar)
        raw_w = raw_h * src_ar

        prep_h = max(factor, int(round(raw_h / factor)) * factor)
        prep_w = max(factor, int(round(raw_w / factor)) * factor)
    elif mode == "fixed":
        target_pixels = int(fixed_mp * 1_000_000)
        src_ar = image_w / float(image_h)
        raw_h = math.sqrt(target_pixels / src_ar)
        raw_w = raw_h * src_ar

        prep_h = max(factor, int(round(raw_h / factor)) * factor)
        prep_w = max(factor, int(round(raw_w / factor)) * factor)
    else:
        prep_h, prep_w = native_h, native_w

    src_pixels = image_h * image_w
    prep_pixels = prep_h * prep_w

    if (prep_h, prep_w) == (image_h, image_w):
        direction = "none"
        auto_method = "none"
    elif prep_pixels < src_pixels:
        direction = "downscale"
        auto_method = "area"
    else:
        direction = "upscale"
        auto_method = "bicubic"

    # Check if Qwen vision processing on prep_h, prep_w will produce identical dimensions
    qwen_re_h, qwen_re_w = calculate_native_qwen_geometry(prep_h, prep_w, config)
    if (qwen_re_h, qwen_re_w) == (prep_h, prep_w):
        add_adj = "no"
    else:
        add_adj = f"yes ({qwen_re_w} x {qwen_re_h})"

    return native_h, native_w, prep_h, prep_w, direction, auto_method, add_adj


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

    bs, ih, iw, c = image.shape
    config = resolve_qwen_encoder_config(clip)

    native_h, native_w, prep_h, prep_w, direction, auto_method, add_adj = calculate_qwen_vision_resolution(
        image_h=ih,
        image_w=iw,
        mode=mode,
        min_mp=min_mp,
        max_mp=max_mp,
        fixed_mp=fixed_mp,
        config=config
    )

    resolved_method = auto_method
    if mode == "native":
        resolved_method = config.interpolation  # Section 5.1: native mode uses config.interpolation ("bilinear")
    elif direction == "downscale" and downscale_method != "auto":
        resolved_method = downscale_method
    elif direction == "upscale" and upscale_method != "auto":
        resolved_method = upscale_method

    if (ih, iw) == (prep_h, prep_w):
        vision_image = image
    else:
        vision_image = resize_tensor(image, target_h=prep_h, target_w=prep_w, method=resolved_method)

    prep_spec = VisionPrepSpec(
        mode=mode,
        semantic_min_mp=min_mp,
        semantic_max_mp=max_mp,
        semantic_fixed_mp=fixed_mp,
        downscale_method_requested=downscale_method,
        upscale_method_requested=upscale_method,
        encoder_signature=config.encoder_signature,
        resolved_alignment=config.factor,
        resolved_native_limits={"min_pixels": config.min_pixels, "max_pixels": config.max_pixels}
    )

    debug_meta = {
        "src_hw": (ih, iw),
        "native_hw": (native_h, native_w),
        "target_hw": (prep_h, prep_w),
        "prep_hw": (prep_h, prep_w),
        "direction": direction,
        "resolved_method": resolved_method,
        "additional_adjustment": add_adj,
        "config": config,
        "downscale_method_requested": downscale_method,
        "upscale_method_requested": upscale_method,
    }

    return PreparedVisionImage(
        original_image=image,
        vision_image=vision_image,
        prep_spec=prep_spec,
        debug_metadata=debug_meta
    )


def format_vision_info(prep_img: PreparedVisionImage) -> str:
    """Format human-readable vision_info string following exact Section 5.2 required key names."""
    meta = prep_img.debug_metadata
    spec = prep_img.prep_spec
    config: QwenVisionEncoderConfig = meta["config"]

    ih, iw = meta["src_hw"]
    nh, nw = meta["native_hw"]
    ph, pw = meta["prep_hw"]

    src_area = (ih * iw) / 1_000_000.0
    native_area = (nh * nw) / 1_000_000.0
    prep_area = (ph * pw) / 1_000_000.0

    req_method = (
        spec.downscale_method_requested if meta["direction"] == "downscale"
        else (spec.upscale_method_requested if meta["direction"] == "upscale" else "auto")
    )

    config_source_str = f"{config.value_sources.get('encoder_signature', 'fallback')} (patch: {config.value_sources.get('patch_size', 'fallback')}, limits: {config.value_sources.get('min_pixels', 'fallback')})"
    warnings_str = "; ".join(config.introspection_warnings) if config.introspection_warnings else "none"

    lines = [
        f"Vision Encoder: {config.encoder_signature}",
        f"Configuration Source: {config_source_str}",
        f"Patch Size: {config.patch_size}",
        f"Merge Size: {config.merge_size}",
        f"Encoder Factor: {config.factor}",
        f"Native Minimum Pixels: {config.min_pixels}",
        f"Native Maximum Pixels: {config.max_pixels}",
        f"Native Interpolation: {config.interpolation}",
        f"Introspection Status: {config.introspection_status}",
        f"Introspection Warnings: {warnings_str}",
        f"Mode: {spec.mode}",
        f"Source Size: {iw} x {ih}",
        f"Source Area: {src_area:.3f} MP",
        f"Native Target Size: {nw} x {nh}",
        f"Native Target Area: {native_area:.3f} MP",
        f"Configured Minimum MP: {spec.semantic_min_mp:.3f} MP",
        f"Configured Maximum MP: {spec.semantic_max_mp:.3f} MP",
        f"Configured Fixed MP: {spec.semantic_fixed_mp:.3f} MP",
        f"Prepared Size: {pw} x {ph}",
        f"Prepared Area: {prep_area:.3f} MP",
        f"Resize Applied: {'yes' if meta['direction'] != 'none' else 'no'}",
        f"Resize Direction: {meta['direction']}",
        f"Resize Method Requested: {req_method}",
        f"Resize Method Resolved: {meta['resolved_method']}",
        f"Expected Additional Geometry Adjustment: {meta['additional_adjustment']}"
    ]

    return "\n".join(lines)


# Alias for prepared vision image creation
prepare_image_for_qwen = prepare_vision_image


