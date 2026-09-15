"""Krea2 CcC Edit node."""

from typing import Any, Optional

import torch

from .. import edit_engine as edit_engine_runtime
from ..constants import NODE_CATEGORY
from ..grounding import resize_grounding_image
from ..patch import attach_reference_runtime_to_conditioning
from ..rednode_contract import build_grounded_negative_user_content, build_grounded_positive_user_content
from ..rednode_identity_path import encode_visual_identity_direct
from ..rednode_runtime import patch_krea2_model_rednode
from ..reference_specs import (
    PreparedVisionImage,
    ReferenceChain,
    ReferenceSpec,
    StyleReferenceSpec,
    VisionPrepSpec,
)
from .edit_reference_types import SemanticReferenceChain, VisualReferenceChain
from .rope_position import install_krea2_rope_positioning


# Generic semantic/style fallback keeps the same text contract. Pure visual Identity
# workflows bypass edit_engine entirely and use rednode_identity_path instead.
edit_engine_runtime.build_krea2_user_content = build_grounded_positive_user_content
edit_engine_runtime.build_krea2_negative_user_content = build_grounded_negative_user_content
install_krea2_rope_positioning()


def _prepare_qwen_image(image: torch.Tensor, clip: Any, grounding_px: int):
    """Prepare Qwen grounding like RedNode before CLIP tokenization."""
    del clip
    if image.ndim == 3:
        image = image.unsqueeze(0)
    original = image

    if grounding_px == 0:
        vision_input = image
        resize_applied = False
    else:
        vision_input = resize_grounding_image(
            image=image,
            resize_mode="downscale_only",
            grounding_px=grounding_px,
            grounding_preset="custom",
            resize_method="area",
        )
        resize_applied = vision_input.shape[1:3] != image.shape[1:3]

    vision_input = vision_input[..., :3].clamp(0.0, 1.0)
    src_h, src_w = int(original.shape[1]), int(original.shape[2])
    prep_h, prep_w = int(vision_input.shape[1]), int(vision_input.shape[2])

    return PreparedVisionImage(
        original_image=original,
        vision_image=vision_input,
        prep_spec=VisionPrepSpec(
            mode="native",
            semantic_min_mp=0.0,
            semantic_max_mp=0.0,
            semantic_fixed_mp=0.0,
            downscale_method_requested="area",
            upscale_method_requested="none",
            encoder_signature="Qwen tokenizer-native",
            resolved_alignment=None,
            resolved_native_limits=None,
        ),
        debug_metadata={
            "src_hw": (src_h, src_w),
            "prep_hw": (prep_h, prep_w),
            "target_hw": (prep_h, prep_w),
            "direction": "downscale" if resize_applied else "none",
            "resolved_method": "area" if resize_applied else "none",
            "additional_adjustment": "owned by Qwen tokenizer",
        },
    )


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


def _runtime_latent(latent: dict) -> dict:
    runtime = {
        "samples": latent["samples"],
        "batch_index": latent.get("batch_index"),
        "target_vision_context": latent.get("target_vision_context"),
    }
    if runtime["batch_index"] is None:
        runtime.pop("batch_index")
    if runtime["target_vision_context"] is None:
        runtime.pop("target_vision_context")
    return runtime


def _combine_reference_chains(
    clip: Any,
    visual_references: Optional[VisualReferenceChain],
    semantic_references: Optional[SemanticReferenceChain],
    latent: dict,
) -> ReferenceChain:
    """Build the generic chain used only when semantic/style extensions are active."""
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

    latent_semantic = latent.get("ccc_krea2_latent_semantic") or {}
    if latent_semantic.get("enabled"):
        latent_image = latent_semantic.get("image")
        if latent_image is None:
            raise ValueError("[CcC Krea2] Latent semantic metadata is enabled but does not contain an image.")
        chain = _append_semantic(
            chain=chain,
            image=latent_image,
            clip=clip,
            instruction=str(latent_semantic.get("instruction", "")),
            grounding_px=int(latent_semantic.get("grounding_px", 768)),
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


def _normalize_pipeline_report(pipeline_info: str, patch_applied: bool) -> str:
    if not patch_applied:
        return pipeline_info
    lines = []
    skipped_warning = (
        "CcC Krea2 model patch was skipped; reference latents attached to conditioning require compatible runtime support."
    )
    for line in pipeline_info.splitlines():
        if skipped_warning in line:
            continue
        if line == "CcC Model Patch: skipped":
            line = "CcC Model Patch: applied"
        elif line == "Reference Transport: standard ComfyUI reference_latents":
            line = "Reference Transport: conditioning reference_latents (RedNode-compatible)"
        lines.append(line)
    if lines and lines[-1] == "Warnings:":
        lines.pop()
    return "\n".join(lines)


def _first_conditioning_extras(conditioning) -> dict:
    if not conditioning or not isinstance(conditioning, (list, tuple)):
        return {}
    first = conditioning[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2 or not isinstance(first[1], dict):
        return {}
    return first[1]


def _flatten_tensor_values(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_tensor_values(item))
        return out
    return []


def _format_runtime_conditioning(conditioning) -> str:
    extras = _first_conditioning_extras(conditioning)
    tensors = _flatten_tensor_values(extras.get("reference_latents"))
    shapes = [tuple(int(v) for v in tensor.shape) for tensor in tensors]
    return (
        f"ref_latents={len(tensors)} shapes={shapes}, "
        f"reference_fit={extras.get('reference_fit', '<missing>')}, "
        f"reference_boosts={extras.get('reference_boosts', '<implicit 1.0>')}, "
        f"rope={extras.get('reference_rope_positions', '<missing>')}"
    )


def _runtime_forward_owner() -> str:
    try:
        from comfy.ldm.krea2.model import SingleStreamDiT

        fn = SingleStreamDiT._forward
        return (
            f"{getattr(fn, '__module__', '<unknown>')}."
            f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '<unknown>'))}"
        )
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def _runtime_patch_markers() -> str:
    try:
        from comfy.ldm.krea2.model import SingleStreamDiT

        return (
            f"rednode={bool(getattr(SingleStreamDiT, '_krea2_identity_patched', False))}, "
            f"ccc={bool(getattr(SingleStreamDiT, '_ccc_rednode_identity_patched', False))}"
        )
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


class CcCKrea2Edit:
    """Krea2 Edit orchestrator consuming a pre-built target latent."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 Edit orchestrator. Target latent construction lives in Krea2 CcC Latent. "
        "Every visual reference is fitted to that resolved target using the Identity Edit v1.2 "
        "pixel-space geometry. Pure visual Identity workflows use a direct RedNode-compatible "
        "conditioning path; semantic/style extensions use the generic orchestrator."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "apply_krea2_edit_patch": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "visual_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
                "semantic_references": ("KREA2_SEMANTIC_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        model,
        clip,
        vae,
        latent,
        positive_prompt="",
        negative_prompt="",
        apply_krea2_edit_patch=True,
        visual_references=None,
        semantic_references=None,
    ):
        visual_entries = (visual_references or VisualReferenceChain()).entries
        semantic_entries = (semantic_references or SemanticReferenceChain()).entries
        latent_semantic = latent.get("ccc_krea2_latent_semantic") or {}
        runtime_latent = _runtime_latent(latent)

        positive_boosts = [float(entry.boost) for entry in visual_entries]
        negative_boosts = [1.0] * len(positive_boosts)
        rope_positions = [entry.rope_position for entry in visual_entries]

        # Pure visual Identity workflows get the direct proven encode path. No generic slots,
        # span extraction, style post-processing, prompt augmentation, or PreparedReference layer.
        direct_identity = bool(visual_entries) and not semantic_entries and not latent_semantic.get("enabled")
        direct_result = None

        if direct_identity:
            direct_result = encode_visual_identity_direct(
                clip=clip,
                vae=vae,
                visual_entries=visual_entries,
                target_latent=runtime_latent,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
            )
            positive = direct_result.positive
            negative = direct_result.negative
            latent_out = runtime_latent
            pipeline_info = (
                "=== CcC Krea2 Edit Pipeline Report ===\n"
                "Reference Contract: krea2_edit\n"
                "Conditioning Path: direct RedNode Identity encode\n"
                "Generic Edit Engine: bypassed\n"
                f"Target Pixel Geometry: {runtime_latent['samples'].shape[-1] * 8} x "
                f"{runtime_latent['samples'].shape[-2] * 8}\n"
                f"Target Latent Geometry: {runtime_latent['samples'].shape[-1]} x "
                f"{runtime_latent['samples'].shape[-2]}"
            )
        else:
            chain = _combine_reference_chains(
                clip=clip,
                visual_references=visual_references,
                semantic_references=semantic_references,
                latent=latent,
            )
            _, positive, negative, latent_out, pipeline_info = edit_engine_runtime.run_krea2_edit_orchestrator(
                model=model,
                clip=clip,
                vae=vae,
                references=chain,
                target_latent=runtime_latent,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                reference_method="krea2_edit",
                apply_model_patch=False,
            )

            positive = attach_reference_runtime_to_conditioning(
                positive,
                reference_count=len(visual_entries),
                rope_positions=rope_positions,
                reference_boosts=positive_boosts,
            )
            negative = attach_reference_runtime_to_conditioning(
                negative,
                reference_count=len(visual_entries),
                rope_positions=rope_positions,
                reference_boosts=None,
            )
            pipeline_info = _normalize_pipeline_report(pipeline_info, bool(apply_krea2_edit_patch))

        patched_model = patch_krea2_model_rednode(model) if apply_krea2_edit_patch else model

        lines = [
            "=== Krea2 CcC Edit Report ===",
            "Target Latent: external Krea2 CcC Latent",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
            f"Conditioning Path: {'direct RedNode Identity encode' if direct_identity else 'generic semantic/style orchestrator'}",
        ]

        if visual_entries:
            lines.append("Reference Geometry: mandatory Identity Edit v1.2 pixel-space fit to resolved target latent")
            lines.append("Reference Transport: CONDITIONING metadata (reference_latents/reference_fit)")
            lines.append(f"Runtime Forward Owner: {_runtime_forward_owner()}")
            lines.append(f"Runtime Patch Markers: {_runtime_patch_markers()}")
            lines.append("Positive Grounding: VISION_BLOCK * N + user prompt")
            lines.append("Visual Reference semantic_role/instruction: metadata-only on Identity grounding path")
            lines.append("Qwen Grounding Resize: RedNode AREA longest-side cap")
            lines.append(f"Positive Reference Boosts: {positive_boosts}")
            lines.append(f"Negative Reference Boosts: {negative_boosts}")
            lines.append("Negative Grounding: same visual Qwen images; no positive semantic annotations")
            lines.append(f"Positive Conditioning Runtime: {_format_runtime_conditioning(positive)}")
            lines.append(f"Negative Conditioning Runtime: {_format_runtime_conditioning(negative)}")

        for index, entry in enumerate(visual_entries, start=1):
            role = entry.semantic_role if entry.semantic_role else "<positional>"
            base = (
                f"Visual Reference {index}: boost={entry.boost}, fit=krea2_v1.2, "
                f"rope={entry.rope_position}, semantic={entry.semantic}, semantic_role={role}"
            )
            if direct_result is not None and index <= len(direct_result.geometries):
                geom = direct_result.geometries[index - 1]
                base += (
                    f", source={geom.source_size[0]}x{geom.source_size[1]}, "
                    f"vae={geom.vae_input_pixel_size[0]}x{geom.vae_input_pixel_size[1]}, "
                    f"latent={geom.vae_latent_grid_size[0]}x{geom.vae_latent_grid_size[1]}"
                )
            lines.append(base)

        if direct_result is not None:
            lines.append(f"Direct Qwen Image Sizes: {list(direct_result.qwen_sizes)}")
            lines.append(f"Direct Reference Latent Shapes: {list(direct_result.reference_latent_shapes)}")

        latent_info = latent.get("ccc_krea2_latent_info")
        if latent_info:
            lines.extend(["", str(latent_info)])

        lines.extend(["", pipeline_info])
        return patched_model, positive, negative, latent_out, "\n".join(lines)
