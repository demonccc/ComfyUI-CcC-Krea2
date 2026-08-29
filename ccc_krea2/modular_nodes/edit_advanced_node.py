"""Single-node laboratory for explicit Krea2 Edit routing and geometry experiments."""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..constants import NODE_CATEGORY
from ..edit_engine import run_krea2_edit_orchestrator
from ..geometry import resize_tensor
from ..grounding import resize_grounding_image
from ..reference_specs import ReferenceChain, ReferenceSpec, StyleReferenceSpec
from ..target_latent import TargetVisionContext, get_image_dims, normalize_vae_output
from ..vision_prep import prepare_image_for_qwen


IMAGE_SOURCES = ("none", "subject", "scene", "outfit", "style")
LATENT_SOURCES = ("empty", "subject", "scene", "outfit", "style")
ASPECT_RATIOS = ("from source", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")
RESOLUTIONS = ("from source", "0.5 MP", "1.0 MP", "1.5 MP", "2.0 MP", "2.5 MP", "3.0 MP", "4.0 MP")
ROPE_POSITIONS = ("none", "up", "down", "left", "right")
SEMANTIC_MODES = ("semantic_only", "style_direct", "style_indirect")
STYLE_PROCESSING = ("full", "2x2", "4x4")


def _channel_alias(channel: str, source_name: str) -> str:
    """Return one readable, parser-safe alias unique to an Advanced channel."""
    return f"{channel} ({source_name} image)"


def _source_image(name: str, sources: Dict[str, Optional[torch.Tensor]], *, allow_none: bool = False):
    if name in ("none", "empty", ""):
        if allow_none:
            return None
        raise ValueError(f"[CcC Krea2] Source '{name}' does not select an image.")
    image = sources.get(name)
    if image is None:
        raise ValueError(f"[CcC Krea2] The '{name}' image must be connected when selected.")
    return image


def _align_mp_geometry(megapixels: float, ratio: float) -> Tuple[int, int]:
    pixels = max(1, int(megapixels * 1_000_000))
    raw_h = math.sqrt(pixels / ratio)
    raw_w = raw_h * ratio
    return max(128, int(round(raw_w / 16.0)) * 16), max(128, int(round(raw_h / 16.0)) * 16)


def _resolve_target_geometry(
    latent_source: str,
    aspect_ratio: str,
    resolution: str,
    grid_size_source: str,
    grid_geometry_source: str,
    sources: Dict[str, Optional[torch.Tensor]],
) -> Tuple[int, int, str, str]:
    latent_image = None if latent_source == "empty" else _source_image(latent_source, sources)
    size_image = _source_image(grid_size_source, sources, allow_none=True)
    if size_image is None:
        size_image = latent_image
    geometry_image = _source_image(grid_geometry_source, sources, allow_none=True)
    if geometry_image is None:
        geometry_image = latent_image

    if aspect_ratio == "from source":
        if geometry_image is None:
            raise ValueError(
                "[CcC Krea2] aspect_ratio='from source' requires grid_geometry_source or an image latent source."
            )
        gh, gw = get_image_dims(geometry_image)
        ratio = gw / float(gh)
        geometry_label = grid_geometry_source if grid_geometry_source != "none" else latent_source
    else:
        rw, rh = (float(part) for part in aspect_ratio.split(":"))
        ratio = rw / rh
        geometry_label = f"explicit {aspect_ratio}"

    if resolution == "from source":
        if size_image is None:
            raise ValueError(
                "[CcC Krea2] resolution='from source' requires grid_size_source or an image latent source."
            )
        sh, sw = get_image_dims(size_image)
        # Preserve the source image's pixel budget while applying the selected geometry.
        # Image-based latents are then contained on this canvas and padded with white.
        target_w, target_h = _align_mp_geometry((sw * sh) / 1_000_000.0, ratio)
        size_label = grid_size_source if grid_size_source != "none" else latent_source
    else:
        megapixels = float(resolution.replace(" MP", ""))
        target_w, target_h = _align_mp_geometry(megapixels, ratio)
        size_label = f"explicit {resolution}"

    return target_w, target_h, size_label, geometry_label


def _contain_on_white(image: torch.Tensor, target_w: int, target_h: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    _, src_h, src_w, channels = image.shape
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
    latent_source: str,
    latent_image: Optional[torch.Tensor],
    target_w: int,
    target_h: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latent_h = target_h // 8
    latent_w = target_w // 8
    placement: Dict[str, Any] = {"mode": "empty"}

    if latent_source == "empty":
        samples = torch.zeros((batch_size, 16, latent_h, latent_w), dtype=torch.float32)
    else:
        if vae is None:
            raise ValueError("[CcC Krea2] VAE is required for an image-based latent.")
        canvas, placement = _contain_on_white(latent_image, target_w=target_w, target_h=target_h)
        samples = normalize_vae_output(vae.encode(canvas), batch_size=batch_size)
        placement["mode"] = "contain_on_white"

    return {
        "samples": samples,
        "batch_index": list(range(batch_size)),
        "target_vision_context": TargetVisionContext(include_in_vision="no"),
    }, placement


def _prepare_qwen_image(image: torch.Tensor, clip: Any, grounding_px: int):
    if grounding_px == 0:
        vision_input = image
    else:
        vision_input = resize_grounding_image(
            image=image,
            resize_mode="downscale_only",
            grounding_px=grounding_px,
            grounding_preset="custom",
        )
    return prepare_image_for_qwen(image=vision_input, clip=clip, original_image=image)


def _append_semantic(
    chain: ReferenceChain,
    source_name: str,
    image: torch.Tensor,
    clip: Any,
    instruction: str,
    grounding_px: int,
    mode: str = "semantic_only",
    processing: str = "2x2",
    fidelity: float = 1.0,
    alias: Optional[str] = None,
) -> ReferenceChain:
    prep = _prepare_qwen_image(image=image, clip=clip, grounding_px=grounding_px)
    resolved_alias = alias or source_name
    if mode == "semantic_only":
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=resolved_alias,
            vision_instruction=instruction.strip(),
            appearance_reference=False,
            include_in_vision=True,
            _legacy_role=resolved_alias,
        )
    else:
        spec = StyleReferenceSpec(
            reference_path="style",
            prepared_image=prep,
            alias=resolved_alias,
            vision_instruction=instruction.strip(),
            appearance_reference=False,
            include_in_vision=True,
            style_processing=processing,
            style_fidelity=fidelity,
            indirect_style_transfer=(mode == "style_indirect"),
            _legacy_role="style",
        )
    return chain.append(spec)


class CcCKrea2EditAdvanced:
    """Explicit Krea2 Edit laboratory with no presets or automatic prompt routing."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Manual Krea2 Edit laboratory. Explicitly selects latent, two VAE references, target grid sources, "
        "Qwen Vision conditioning, boosts, and experimental RoPE positions."
    )

    @classmethod
    def INPUT_TYPES(cls):
        semantic_common = {
            "semantic_1_source": (IMAGE_SOURCES, {"default": "none"}),
            "semantic_1_mode": (SEMANTIC_MODES, {"default": "semantic_only"}),
            "semantic_1_instruction": ("STRING", {"multiline": True, "default": ""}),
            "semantic_1_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
            "semantic_1_processing": (STYLE_PROCESSING, {"default": "2x2"}),
            "semantic_1_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "semantic_2_source": (IMAGE_SOURCES, {"default": "none"}),
            "semantic_2_mode": (SEMANTIC_MODES, {"default": "semantic_only"}),
            "semantic_2_instruction": ("STRING", {"multiline": True, "default": ""}),
            "semantic_2_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
            "semantic_2_processing": (STYLE_PROCESSING, {"default": "2x2"}),
            "semantic_2_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "latent": (LATENT_SOURCES, {"default": "empty"}),
                "reference_1": (LATENT_SOURCES, {"default": "empty"}),
                "reference_1_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "reference_1_rope_position": (ROPE_POSITIONS, {"default": "none"}),
                "reference_1_semantic": ("BOOLEAN", {"default": True}),
                "reference_1_instruction": ("STRING", {"multiline": True, "default": ""}),
                "reference_1_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                "reference_2": (LATENT_SOURCES, {"default": "empty"}),
                "reference_2_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "reference_2_rope_position": (ROPE_POSITIONS, {"default": "none"}),
                "reference_2_semantic": ("BOOLEAN", {"default": True}),
                "reference_2_instruction": ("STRING", {"multiline": True, "default": ""}),
                "reference_2_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (RESOLUTIONS, {"default": "2.0 MP"}),
                "grid_size_source": (IMAGE_SOURCES, {"default": "none"}),
                "grid_geometry_source": (IMAGE_SOURCES, {"default": "none"}),
                "latent_semantic": ("BOOLEAN", {"default": False}),
                "latent_semantic_instruction": ("STRING", {"multiline": True, "default": ""}),
                "latent_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                **semantic_common,
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "apply_krea2_edit_patch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "subject": ("IMAGE",),
                "scene": ("IMAGE",),
                "outfit": ("IMAGE",),
                "style": ("IMAGE",),
            },
        }

    def process(self, model, clip, vae, positive_prompt="", negative_prompt="", **kwargs):
        sources = {
            "subject": kwargs.get("subject"),
            "scene": kwargs.get("scene"),
            "outfit": kwargs.get("outfit"),
            "style": kwargs.get("style"),
        }
        latent_source = kwargs.get("latent", "empty")
        target_w, target_h, size_label, geometry_label = _resolve_target_geometry(
            latent_source=latent_source,
            aspect_ratio=kwargs.get("aspect_ratio", "1:1"),
            resolution=kwargs.get("resolution", "2.0 MP"),
            grid_size_source=kwargs.get("grid_size_source", "none"),
            grid_geometry_source=kwargs.get("grid_geometry_source", "none"),
            sources=sources,
        )
        latent_image = None if latent_source == "empty" else _source_image(latent_source, sources)
        target_latent, placement = _build_target_latent(
            vae=vae,
            latent_source=latent_source,
            latent_image=latent_image,
            target_w=target_w,
            target_h=target_h,
            batch_size=int(kwargs.get("batch_size", 1)),
        )

        chain = ReferenceChain()
        for index in (1, 2):
            source_name = kwargs.get(f"reference_{index}", "empty")
            if source_name == "empty":
                continue
            image = _source_image(source_name, sources)
            prep = _prepare_qwen_image(
                image=image,
                clip=clip,
                grounding_px=int(kwargs.get(f"reference_{index}_grounding_px", 768)),
            )
            chain = chain.append(
                ReferenceSpec(
                    reference_path="edit",
                    prepared_image=prep,
                    alias=_channel_alias(f"reference {index}", source_name),
                    vision_instruction=kwargs.get(f"reference_{index}_instruction", "").strip(),
                    appearance_reference=True,
                    include_in_vision=bool(kwargs.get(f"reference_{index}_semantic", True)),
                    attention_boost=float(kwargs.get(f"reference_{index}_boost", 1.0)),
                    visual_reference_fit="fit",
                    rope_position=kwargs.get(f"reference_{index}_rope_position", "none"),
                    _legacy_role=f"reference_{index}",
                )
            )

        if kwargs.get("latent_semantic", False):
            if latent_image is None:
                raise ValueError("[CcC Krea2] latent_semantic requires an image-based latent.")
            chain = _append_semantic(
                chain=chain,
                source_name=latent_source,
                image=latent_image,
                clip=clip,
                instruction=kwargs.get("latent_semantic_instruction", ""),
                grounding_px=int(kwargs.get("latent_grounding_px", 768)),
                alias=_channel_alias("latent semantic", latent_source),
            )

        pending_styles = []
        for index in (1, 2):
            source_name = kwargs.get(f"semantic_{index}_source", "none")
            if source_name == "none":
                continue
            mode = kwargs.get(f"semantic_{index}_mode", "semantic_only")
            payload = dict(
                source_name=source_name,
                image=_source_image(source_name, sources),
                clip=clip,
                instruction=kwargs.get(f"semantic_{index}_instruction", ""),
                grounding_px=int(kwargs.get(f"semantic_{index}_grounding_px", 768)),
                mode=mode,
                processing=kwargs.get(f"semantic_{index}_processing", "2x2"),
                fidelity=float(kwargs.get(f"semantic_{index}_fidelity", 1.0)),
                alias=_channel_alias(f"semantic {index}", source_name),
            )
            if mode == "semantic_only":
                chain = _append_semantic(chain=chain, **payload)
            else:
                pending_styles.append(payload)

        # Style paths must follow every edit/semantic-only path because they expand into physical Qwen images.
        for payload in pending_styles:
            chain = _append_semantic(chain=chain, **payload)

        patched_model, positive, negative, latent_out, pipeline_info = run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=chain,
            target_latent=target_latent,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_method="krea2_edit",
            apply_model_patch=bool(kwargs.get("apply_krea2_edit_patch", True)),
        )

        advanced_lines = [
            "=== CcC Krea2 Edit Advanced Report ===",
            f"Latent Source: {latent_source}",
            f"Aspect Ratio: {kwargs.get('aspect_ratio', '1:1')}",
            f"Resolution: {kwargs.get('resolution', '2.0 MP')}",
            f"Grid Size Source: {size_label}",
            f"Grid Geometry Source: {geometry_label}",
            f"Target Pixel Geometry: {target_w} x {target_h}",
            f"Target Latent Geometry: {target_w // 8} x {target_h // 8}",
            f"Latent Placement: {placement}",
            f"Reference 1: {kwargs.get('reference_1', 'empty')}",
            f"Reference 1 RoPE Position: {kwargs.get('reference_1_rope_position', 'none')}",
            f"Reference 2: {kwargs.get('reference_2', 'empty')}",
            f"Reference 2 RoPE Position: {kwargs.get('reference_2_rope_position', 'none')}",
            "",
            pipeline_info,
        ]
        return patched_model, positive, negative, latent_out, "\n".join(advanced_lines)
