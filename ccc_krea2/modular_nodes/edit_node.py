"""Krea2 CcC Edit node built from the former Edit Advanced laboratory."""

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
from .edit_reference_types import SemanticReferenceChain, VisualReferenceChain


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
    image: torch.Tensor,
    clip: Any,
    instruction: str,
    grounding_px: int,
    mode: str = "semantic_only",
    processing: str = "2x2",
    fidelity: float = 1.0,
    alias: str = "",
) -> ReferenceChain:
    prep = _prepare_qwen_image(image=image, clip=clip, grounding_px=grounding_px)

    if mode == "semantic_only":
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=alias,
            vision_instruction=instruction.strip(),
            appearance_reference=False,
            include_in_vision=True,
            _legacy_role=alias or "semantic",
        )
    else:
        spec = StyleReferenceSpec(
            reference_path="style",
            prepared_image=prep,
            alias=alias,
            vision_instruction=instruction.strip(),
            appearance_reference=False,
            include_in_vision=True,
            style_processing=processing,
            style_fidelity=fidelity,
            indirect_style_transfer=(mode == "style_indirect"),
            _legacy_role="style",
        )

    return chain.append(spec)


def _combine_reference_chains(
    clip: Any,
    visual_references: Optional[VisualReferenceChain],
    semantic_references: Optional[SemanticReferenceChain],
    target_image: Optional[torch.Tensor],
    latent_semantic: bool,
    latent_semantic_instruction: str,
    latent_grounding_px: int,
) -> ReferenceChain:
    chain = ReferenceChain()

    for index, entry in enumerate((visual_references or VisualReferenceChain()).entries, start=1):
        prep = _prepare_qwen_image(
            image=entry.image,
            clip=clip,
            grounding_px=entry.grounding_px if entry.semantic else 0,
        )
        chain = chain.append(
            ReferenceSpec(
                reference_path="edit",
                prepared_image=prep,
                alias=entry.semantic_role if entry.semantic else "",
                vision_instruction=entry.instruction if entry.semantic else "",
                appearance_reference=True,
                include_in_vision=entry.semantic,
                attention_boost=entry.boost,
                visual_reference_fit="fit",
                rope_position=entry.rope_position,
                _legacy_role=f"reference_{index}",
            )
        )

    if latent_semantic:
        if target_image is None:
            raise ValueError("[CcC Krea2] latent_semantic requires target_image.")
        chain = _append_semantic(
            chain=chain,
            image=target_image,
            clip=clip,
            instruction=latent_semantic_instruction,
            grounding_px=latent_grounding_px,
            alias="target image",
        )

    pending_styles = []
    for entry in (semantic_references or SemanticReferenceChain()).entries:
        payload = {
            "image": entry.image,
            "clip": clip,
            "instruction": entry.instruction,
            "grounding_px": entry.grounding_px,
            "mode": entry.mode,
            "processing": entry.processing,
            "fidelity": entry.fidelity,
            "alias": "",
        }
        if entry.mode == "semantic_only":
            chain = _append_semantic(chain=chain, **payload)
        else:
            pending_styles.append(payload)

    for payload in pending_styles:
        chain = _append_semantic(chain=chain, **payload)

    return chain


class CcCKrea2Edit:
    """Split Krea2 Edit orchestrator using visual and semantic reference chains."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 Edit orchestrator. Visual and semantic references are declared in dedicated nodes; "
        "this node owns prompt, target latent geometry, target semantic participation, batch, and patching."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (RESOLUTIONS, {"default": "2.0 MP"}),
                "latent_semantic": ("BOOLEAN", {"default": False}),
                "latent_semantic_instruction": ("STRING", {"multiline": True, "default": ""}),
                "latent_grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "apply_krea2_edit_patch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "target_image": ("IMAGE",),
                "grid_size_image": ("IMAGE",),
                "grid_geometry_image": ("IMAGE",),
                "visual_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
                "semantic_references": ("KREA2_SEMANTIC_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        model,
        clip,
        vae,
        positive_prompt="",
        negative_prompt="",
        aspect_ratio="1:1",
        resolution="2.0 MP",
        latent_semantic=False,
        latent_semantic_instruction="",
        latent_grounding_px=768,
        batch_size=1,
        apply_krea2_edit_patch=True,
        target_image=None,
        grid_size_image=None,
        grid_geometry_image=None,
        visual_references=None,
        semantic_references=None,
    ):
        target_w, target_h, size_label, geometry_label = _resolve_target_geometry(
            target_image=target_image,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            grid_size_image=grid_size_image,
            grid_geometry_image=grid_geometry_image,
        )

        target_latent, placement = _build_target_latent(
            vae=vae,
            target_image=target_image,
            target_w=target_w,
            target_h=target_h,
            batch_size=int(batch_size),
        )

        chain = _combine_reference_chains(
            clip=clip,
            visual_references=visual_references,
            semantic_references=semantic_references,
            target_image=target_image,
            latent_semantic=bool(latent_semantic),
            latent_semantic_instruction=latent_semantic_instruction,
            latent_grounding_px=int(latent_grounding_px),
        )

        patched_model, positive, negative, latent_out, pipeline_info = run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=chain,
            target_latent=target_latent,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_method="krea2_edit",
            apply_model_patch=bool(apply_krea2_edit_patch),
        )

        visual_entries = (visual_references or VisualReferenceChain()).entries
        semantic_entries = (semantic_references or SemanticReferenceChain()).entries

        lines = [
            "=== Krea2 CcC Edit Report ===",
            f"Target Latent: {'image' if target_image is not None else 'empty'}",
            f"Aspect Ratio: {aspect_ratio}",
            f"Resolution: {resolution}",
            f"Grid Size Source: {size_label}",
            f"Grid Geometry Source: {geometry_label}",
            f"Target Pixel Geometry: {target_w} x {target_h}",
            f"Target Latent Geometry: {target_w // 8} x {target_h // 8}",
            f"Target Placement: {placement}",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
        ]

        for index, entry in enumerate(visual_entries, start=1):
            role = entry.semantic_role if entry.semantic_role else "<positional>"
            lines.append(
                f"Visual Reference {index}: boost={entry.boost}, rope={entry.rope_position}, "
                f"semantic={entry.semantic}, semantic_role={role}"
            )

        lines.extend(["", pipeline_info])
        return patched_model, positive, negative, latent_out, "\n".join(lines)
