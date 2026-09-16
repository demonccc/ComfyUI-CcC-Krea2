"""Krea2 CcC Edit node."""

from typing import Any, Optional

import torch

from .. import edit_engine as edit_engine_runtime
from ..constants import NODE_CATEGORY
from ..grounding import resize_grounding_image
from ..orchestrator_runtime import patch_krea2_orchestrated_model
from ..reference_specs import (
    PreparedVisionImage,
    ReferenceChain,
    ReferenceSpec,
    StyleReferenceSpec,
    VisionPrepSpec,
)
from .edit_reference_types import SemanticReferenceChain, VisualReferenceChain
from .rope_position import install_krea2_rope_positioning


# Keep the split public node surface, but restore the original orchestrator transport:
# prepared appearance references are captured by the MODEL wrapper instead of bypassing
# the orchestrator or moving appearance state into CONDITIONING.
edit_engine_runtime.patch_krea2_model = patch_krea2_orchestrated_model
install_krea2_rope_positioning()


def _prepare_qwen_image(image: torch.Tensor, clip: Any, grounding_px: int):
    """Prepare the semantic copy sent to Qwen while preserving the original image for VAE fit."""
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
    """Translate the split public nodes back into the original orchestrator ReferenceChain."""
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
                attention_boost=float(entry.boost),
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

    # Style references remain last because a single logical style reference can expand
    # into multiple physical Qwen images.
    for payload in pending_styles:
        chain = _append_semantic(chain=chain, **payload)

    return chain


def _format_model_runtime(model) -> str:
    return (
        f"refs={getattr(model, '_ccc_orchestrated_reference_count', 0)} "
        f"shapes={getattr(model, '_ccc_orchestrated_reference_shapes', [])}, "
        f"boosts={getattr(model, '_ccc_orchestrated_reference_boosts', [])}, "
        f"rope={getattr(model, '_ccc_orchestrated_reference_rope', [])}"
    )


class CcCKrea2Edit:
    """Krea2 Edit orchestrator consuming the split CcC Latent/Reference node outputs."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 Edit orchestrator. The public Latent and Reference nodes stay modular, while "
        "execution uses the original CcC orchestrator contract: Qwen conditioning, reference "
        "preparation and the MODEL appearance wrapper are resolved together."
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

        chain = _combine_reference_chains(
            clip=clip,
            visual_references=visual_references,
            semantic_references=semantic_references,
            latent=latent,
        )

        patched_model, positive, negative, latent_out, pipeline_info = (
            edit_engine_runtime.run_krea2_edit_orchestrator(
                model=model,
                clip=clip,
                vae=vae,
                references=chain,
                target_latent=runtime_latent,
                positive_prompt=positive_prompt,
                negative_prompt=negative_prompt,
                reference_method="krea2_edit",
                apply_model_patch=bool(apply_krea2_edit_patch),
            )
        )

        lines = [
            "=== Krea2 CcC Edit Report ===",
            "Target Latent: external Krea2 CcC Latent",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
            "Conditioning Path: orchestrated Krea2 Edit",
            "Generic Edit Engine: active",
            (
                "Appearance Transport: MODEL wrapper closure"
                if apply_krea2_edit_patch
                else "Appearance Transport: external runtime / conditioning fallback"
            ),
        ]

        if visual_entries and apply_krea2_edit_patch:
            lines.append(f"Model Appearance Runtime: {_format_model_runtime(patched_model)}")

        for index, entry in enumerate(visual_entries, start=1):
            role = entry.semantic_role if entry.semantic_role else "<positional>"
            lines.append(
                f"Visual Reference {index}: boost={entry.boost}, fit=krea2_v1.2, "
                f"rope={entry.rope_position}, semantic={entry.semantic}, semantic_role={role}"
            )

        latent_info = latent.get("ccc_krea2_latent_info")
        if latent_info:
            lines.extend(["", str(latent_info)])

        lines.extend(["", pipeline_info])
        return patched_model, positive, negative, latent_out, "\n".join(lines)
