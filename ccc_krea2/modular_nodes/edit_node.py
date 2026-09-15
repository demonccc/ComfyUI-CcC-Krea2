"""Krea2 CcC Edit node."""

from typing import Any, Optional

import torch

from .. import edit_engine as edit_engine_runtime
from ..constants import NODE_CATEGORY
from ..grounding import resize_grounding_image
from ..patch import attach_reference_runtime_to_conditioning, patch_krea2_model
from ..rednode_contract import build_grounded_negative_user_content
from ..reference_specs import ReferenceChain, ReferenceSpec, StyleReferenceSpec
from ..vision_prep import prepare_image_for_qwen
from .edit_reference_types import SemanticReferenceChain, VisualReferenceChain
from .rope_position import install_krea2_rope_positioning


# Keep the existing orchestrator, but make its negative builder match the proven Identity Edit
# contract: same Qwen appearance images, no positive aliases/instructions in the negative text.
edit_engine_runtime.build_krea2_negative_user_content = build_grounded_negative_user_content
install_krea2_rope_positioning()


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


def _runtime_latent(latent: dict) -> dict:
    """Restore the exact target-latent contract used before the Latent/Edit split."""
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
                # Target geometry may come from preset, fixed dimensions, Size Resolver, or image.
                # Once resolved, every visual reference uses the mandatory Identity Edit v1.2 fit.
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
    # Avoid leaving an empty Warnings section when the skipped-patch warning was the only warning.
    if lines and lines[-1] == "Warnings:":
        lines.pop()
    return "\n".join(lines)


class CcCKrea2Edit:
    """Krea2 Edit orchestrator consuming a pre-built target latent."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 Edit orchestrator. Target latent construction lives in Krea2 CcC Latent. "
        "Every visual reference is fitted to that resolved target using the Identity Edit v1.2 "
        "pixel-space geometry. Reference latents and fit metadata travel through CONDITIONING; "
        "RoPE positioning can move the fitted reference coordinates without changing sizing."
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
        chain = _combine_reference_chains(
            clip=clip,
            visual_references=visual_references,
            semantic_references=semantic_references,
            latent=latent,
        )
        runtime_latent = _runtime_latent(latent)

        # Ask the orchestrator for its standard reference_latents conditioning transport. We then
        # install the RedNode-compatible Krea2 runtime wrapper without capturing any refs in MODEL.
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

        positive_boosts = [float(entry.boost) for entry in visual_entries]
        negative_boosts = [1.0] * len(positive_boosts)
        rope_positions = [entry.rope_position for entry in visual_entries]

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

        patched_model = patch_krea2_model(model, prepared_refs=[]) if apply_krea2_edit_patch else model
        pipeline_info = _normalize_pipeline_report(pipeline_info, bool(apply_krea2_edit_patch))

        semantic_entries = (semantic_references or SemanticReferenceChain()).entries
        latent_semantic = latent.get("ccc_krea2_latent_semantic") or {}
        lines = [
            "=== Krea2 CcC Edit Report ===",
            "Target Latent: external Krea2 CcC Latent",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
        ]

        if visual_entries:
            lines.append("Reference Geometry: mandatory Identity Edit v1.2 pixel-space fit to resolved target latent")
            lines.append("Reference Transport: CONDITIONING metadata (reference_latents/reference_fit)")
            lines.append(f"Positive Reference Boosts: {positive_boosts}")
            lines.append(f"Negative Reference Boosts: {negative_boosts}")
            lines.append("Negative Grounding: same visual refs; semantic aliases/instructions excluded")

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
