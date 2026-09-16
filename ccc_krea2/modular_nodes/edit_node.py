"""Krea2 CcC Edit node."""

from typing import Any, Optional

import torch

from .. import edit_engine as edit_engine_runtime
from ..constants import NODE_CATEGORY
from ..grounding import resize_grounding_image
from ..identity_contract import (
    build_grounded_negative_user_content,
    build_grounded_positive_user_content,
)
from ..patch import attach_reference_runtime_to_conditioning, patch_krea2_model
from ..reference_specs import (
    PreparedVisionImage,
    ReferenceChain,
    ReferenceSpec,
    StyleReferenceSpec,
    VisionPrepSpec,
)
from .edit_reference_types import SemanticReferenceChain, VisualReferenceChain
from .rope_position import install_krea2_rope_positioning


# Krea2 Edit uses a grounded unconditional branch rather than a classic SD-style
# negative prompt. Keep the negative text empty by contract; visual references remain
# attached to both branches and negative reference boosts stay neutral at 1.0.
KREA2_EDIT_NEGATIVE_PROMPT = ""


# Identity appearance references are positional in the proven Krea2 grounding contract:
# VISION_BLOCK * N + instruction. Semantic/style-only CcC extensions may still add their
# annotations after the complete visual prefix.
edit_engine_runtime.build_krea2_user_content = build_grounded_positive_user_content
edit_engine_runtime.build_krea2_negative_user_content = build_grounded_negative_user_content
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
    """Translate the split public nodes into the common orchestrator ReferenceChain."""
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
                # Kept as metadata for reports/routing. Identity grounding builders deliberately
                # do not inject appearance-reference labels or instructions into Qwen text.
                alias=entry.semantic_role if entry.semantic else "",
                vision_instruction=entry.instruction if entry.semantic else "",
                appearance_reference=True,
                include_in_vision=entry.semantic,
                attention_boost=float(entry.boost),
                visual_reference_fit=entry.resolved_fit_mode,
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


def _conditioning_extras(conditioning):
    if not conditioning:
        return {}
    first = conditioning[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
        return first[1]
    return {}


def _format_conditioning_runtime(conditioning, reference_count: int) -> str:
    extras = _conditioning_extras(conditioning)
    refs = list(extras.get("reference_latents") or [])
    shapes = [tuple(int(v) for v in ref.shape) for ref in refs if torch.is_tensor(ref)]
    fit = list(extras.get("reference_fit") or [])
    rope = list(extras.get("reference_rope_positions") or [])
    boosts = extras.get("reference_boosts")
    if boosts is None:
        boosts = [1.0] * reference_count
    else:
        boosts = [float(v) for v in boosts]
    return f"refs={len(refs)} shapes={shapes}, fit={fit}, boosts={boosts}, rope={rope}"


def _external_moodboard_runtime_active() -> bool:
    """Return True when Krea2Moodboard already owns the proven Identity/Edit DiT forward."""
    try:
        from comfy.ldm.krea2.model import SingleStreamDiT
    except (ImportError, AttributeError):
        return False
    return bool(getattr(SingleStreamDiT, "_krea2_identity_patched", False))


def _external_moodboard_runtime_owner() -> str:
    try:
        from comfy.ldm.krea2.model import SingleStreamDiT
    except (ImportError, AttributeError):
        return "unavailable"
    forward = getattr(SingleStreamDiT, "_forward", None)
    if forward is None:
        return "unknown"
    return f"{getattr(forward, '__module__', '<unknown>')}.{getattr(forward, '__qualname__', getattr(forward, '__name__', '<unknown>'))}"


def _uses_only_standard_center_rope(visual_entries) -> bool:
    """Moodboard natively implements the centered in-target reference placement."""
    return all(
        getattr(entry, "rope_position", "none") in ("none", "inside:center:center")
        for entry in visual_entries
    )


def _rewrite_pipeline_report_for_conditioning_runtime(report: str, runtime_mode: str) -> str:
    if runtime_mode == "none":
        return report

    lines = []
    skip_warning = (
        "CcC Krea2 model patch was skipped; reference latents attached to conditioning require compatible runtime support."
    )
    for line in report.splitlines():
        if skip_warning in line:
            continue
        if line == "CcC Model Patch: skipped":
            if runtime_mode == "external_moodboard":
                lines.append("CcC Model Patch: not needed (external Krea2Moodboard runtime reused)")
            else:
                lines.append("CcC Model Patch: applied")
        elif line == "Reference Transport: standard ComfyUI reference_latents":
            if runtime_mode == "external_moodboard":
                lines.append("Reference Transport: CONDITIONING metadata -> Krea2Moodboard runtime")
            else:
                lines.append("Reference Transport: CONDITIONING metadata (refs/fit/boost/RoPE)")
        else:
            lines.append(line)
    return "\n".join(lines)


class CcCKrea2Edit:
    """Krea2 Edit orchestrator consuming the split CcC Latent/Reference node outputs."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 Edit orchestrator. CcC resolves target geometry and reference preparation while "
        "appearance refs, fit state, per-pass boosts and RoPE placement travel with CONDITIONING. "
        "The grounded negative branch always uses an empty text prompt and neutral reference boosts. "
        "For standard centered references, an installed Krea2Moodboard Identity runtime is reused directly."
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

        # Preparation stays in CcC: target geometry, Qwen images, VAE refs and per-pass metadata.
        # Runtime selection is intentionally separate. The centered baseline delegates to the
        # already-installed Krea2Moodboard Identity forward when available because that is the
        # proven behavior this custom node extends rather than replaces.
        _, positive, negative, latent_out, pipeline_info = edit_engine_runtime.run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=chain,
            target_latent=runtime_latent,
            positive_prompt=positive_prompt,
            negative_prompt=KREA2_EDIT_NEGATIVE_PROMPT,
            reference_method="krea2_edit",
            apply_model_patch=False,
        )

        reference_count = len(visual_entries)
        rope_positions = [entry.rope_position for entry in visual_entries]
        positive_boosts = [float(entry.boost) for entry in visual_entries]

        if reference_count:
            positive = attach_reference_runtime_to_conditioning(
                positive,
                reference_count=reference_count,
                rope_positions=rope_positions,
                reference_boosts=positive_boosts,
            )
            negative = attach_reference_runtime_to_conditioning(
                negative,
                reference_count=reference_count,
                rope_positions=rope_positions,
                reference_boosts=None,
            )

        external_moodboard = _external_moodboard_runtime_active()
        standard_center = _uses_only_standard_center_rope(visual_entries)
        runtime_mode = "none"

        if apply_krea2_edit_patch and reference_count:
            if external_moodboard and standard_center:
                # Do NOT install a CcC diffusion wrapper here. CONDITIONING already contains
                # the exact refs/fit/boost contract consumed by Krea2Moodboard, so leaving MODEL
                # untouched lets its proven SingleStreamDiT._forward execute byte-for-byte.
                patched_model = model
                runtime_mode = "external_moodboard"
            else:
                # CcC only owns the forward when Moodboard is unavailable or a non-standard
                # RoPE placement (outside/left/right/up/down) requires our extension.
                patched_model = patch_krea2_model(model=model)
                runtime_mode = "ccc_extended"
        else:
            patched_model = model

        pipeline_info = _rewrite_pipeline_report_for_conditioning_runtime(
            pipeline_info,
            runtime_mode=runtime_mode,
        )

        lines = [
            "=== Krea2 CcC Edit Report ===",
            "Target Latent: external Krea2 CcC Latent",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
            "Conditioning Path: orchestrated Krea2 Edit preparation",
            "Generic Edit Engine: active",
            (
                "Appearance Transport: CONDITIONING metadata"
                if reference_count
                else "Appearance Transport: none"
            ),
            "Identity Qwen Contract: positional visual blocks; Visual Reference labels/instructions are metadata-only",
            "Negative Prompt: fixed empty string",
            (
                "Identity Runtime: external Krea2Moodboard forward"
                if runtime_mode == "external_moodboard"
                else (
                    "Identity Runtime: CcC extended forward"
                    if runtime_mode == "ccc_extended"
                    else "Identity Runtime: no CcC runtime patch requested"
                )
            ),
            f"External Moodboard Runtime Detected: {'yes' if external_moodboard else 'no'}",
            f"Runtime Forward Owner: {_external_moodboard_runtime_owner()}",
            f"CcC Forward Override: {'yes' if runtime_mode == 'ccc_extended' else 'no'}",
        ]

        if reference_count:
            lines.append(
                f"Positive Appearance Runtime: {_format_conditioning_runtime(positive, reference_count)}"
            )
            lines.append(
                f"Negative Appearance Runtime: {_format_conditioning_runtime(negative, reference_count)}"
            )

        for index, entry in enumerate(visual_entries, start=1):
            role = entry.semantic_role if entry.semantic_role else "<positional>"
            lines.append(
                f"Visual Reference {index}: boost={entry.boost}, fit={entry.fit_mode}, "
                f"rope={entry.rope_position}, semantic={entry.semantic}, semantic_role={role}"
            )

        latent_info = latent.get("ccc_krea2_latent_info")
        if latent_info:
            lines.extend(["", str(latent_info)])

        lines.extend(["", pipeline_info])
        return patched_model, positive, negative, latent_out, "\n".join(lines)