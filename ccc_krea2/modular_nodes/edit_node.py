"""Krea2 CcC Edit node."""

from typing import Any, Optional

import torch

from ..constants import NODE_CATEGORY
from ..edit_engine import run_krea2_edit_orchestrator
from ..grounding import resize_grounding_image
from ..reference_specs import ReferenceChain, ReferenceSpec, StyleReferenceSpec
from ..vision_prep import prepare_image_for_qwen
from .edit_reference_types import SemanticReferenceChain, VisualReferenceChain


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
    latent: dict,
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


class CcCKrea2Edit:
    """Krea2 Edit orchestrator consuming a pre-built target latent."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 Edit orchestrator. Target latent construction lives in Krea2 CcC Latent; "
        "this node combines the latent with visual and semantic references and applies the Krea2 Edit patch."
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
        chain = _combine_reference_chains(
            clip=clip,
            visual_references=visual_references,
            semantic_references=semantic_references,
            latent=latent,
        )

        patched_model, positive, negative, latent_out, pipeline_info = run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=chain,
            target_latent=latent,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            reference_method="krea2_edit",
            apply_model_patch=bool(apply_krea2_edit_patch),
        )

        visual_entries = (visual_references or VisualReferenceChain()).entries
        semantic_entries = (semantic_references or SemanticReferenceChain()).entries
        latent_semantic = latent.get("ccc_krea2_latent_semantic") or {}

        lines = [
            "=== Krea2 CcC Edit Report ===",
            "Target Latent: external Krea2 CcC Latent",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
        ]

        for index, entry in enumerate(visual_entries, start=1):
            role = entry.semantic_role if entry.semantic_role else "<positional>"
            lines.append(
                f"Visual Reference {index}: boost={entry.boost}, rope={entry.rope_position}, "
                f"semantic={entry.semantic}, semantic_role={role}"
            )

        latent_info = latent.get("ccc_krea2_latent_info")
        if latent_info:
            lines.extend(["", str(latent_info)])

        lines.extend(["", pipeline_info])
        return patched_model, positive, negative, latent_out, "\n".join(lines)
