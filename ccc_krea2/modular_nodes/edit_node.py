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

SEMANTIC_SUBJECT_DIRECTIVE = (
    "Use this reference image as a content and composition guide. Preserve its pose, action, "
    "clothing, surrounding people, interactions, objects, background, environment, framing, "
    "and spatial composition. Do not use the reference subject's identity as the target identity; "
    "identity is controlled by the edit identity references."
)


# Visual references are positional in the Krea2 grounding contract:
# VISION_BLOCK * N + instruction. Semantic/style-only references may append
# annotations after the complete visual prefix.
edit_engine_runtime.build_krea2_user_content = build_grounded_positive_user_content
edit_engine_runtime.build_krea2_negative_user_content = build_grounded_negative_user_content
install_krea2_rope_positioning()


def _prepare_qwen_image(
    image: torch.Tensor,
    clip: Any,
    grounding_px: int,
    semantic_resize: bool = True,
    resize_method: str = "lanczos",
):
    """Prepare the Qwen copy while preserving the untouched original image for VAE use."""
    del clip
    if image.ndim == 3:
        image = image.unsqueeze(0)
    original = image

    if semantic_resize:
        vision_input = resize_grounding_image(
            image=image,
            resize_mode="downscale_only",
            grounding_px=grounding_px,
            grounding_preset="custom",
            resize_method=resize_method,
        )
        resize_applied = vision_input.shape[1:3] != image.shape[1:3]
    else:
        vision_input = image
        resize_applied = False

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
            downscale_method_requested=resize_method if semantic_resize else "none",
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
            "resolved_method": resize_method if resize_applied else "none",
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
    prep = _prepare_qwen_image(
        image=image,
        clip=clip,
        grounding_px=grounding_px,
        semantic_resize=True,
        resize_method="lanczos",
    )
    if mode == "semantic_only":
        user_instruction = instruction.strip()
        semantic_instruction = (
            f"{SEMANTIC_SUBJECT_DIRECTIVE} {user_instruction}".strip()
            if user_instruction
            else SEMANTIC_SUBJECT_DIRECTIVE
        )
        spec = ReferenceSpec(
            reference_path="edit",
            prepared_image=prep,
            alias=alias,
            vision_instruction=semantic_instruction,
            appearance_reference=False,
            include_in_vision=True,
            semantic_extract="subject",
            semantic_strength=float(fidelity),
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
        if entry.cache is not None:
            cache = entry.cache
            chain = chain.append(
                ReferenceSpec(
                    reference_path="edit",
                    prepared_image=None,
                    alias="",
                    vision_instruction=entry.prompt_annotation if entry.semantic else "",
                    appearance_reference=True,
                    include_in_vision=entry.semantic,
                    attention_boost=float(entry.boost),
                    visual_reference_fit=entry.resolved_fit_mode,
                    visual_resize_method=entry.resize_method,
                    rope_position=entry.rope_position,
                    cached_appearance_latent=cache.appearance_latent,
                    cached_geometry=cache.geometry,
                    cached_qwen_visual=cache.qwen_visual,
                )
            )
            continue

        if entry.image is None:
            raise ValueError(f"[CcC Krea2] Visual reference {index} has neither IMAGE nor cache.")
        prep = _prepare_qwen_image(
            image=entry.image,
            clip=clip,
            grounding_px=entry.semantic_grounding_px,
            semantic_resize=entry.semantic_resize if entry.semantic else False,
            resize_method=entry.semantic_resize_method,
        )
        chain = chain.append(
            ReferenceSpec(
                reference_path="edit",
                prepared_image=prep,
                alias="",
                vision_instruction=entry.prompt_annotation if entry.semantic else "",
                appearance_reference=True,
                include_in_vision=entry.semantic,
                attention_boost=float(entry.boost),
                visual_reference_fit=entry.resolved_fit_mode,
                visual_resize_method=entry.resize_method,
                rope_position=entry.rope_position,
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
        "CcC Krea2 owns a single edit runtime for all visual references."
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

        # CcC prepares geometry, Qwen images and VAE reference latents.
        _, positive, negative, latent_out, pipeline_info = edit_engine_runtime.run_krea2_edit_orchestrator(
            model=model,
            clip=clip,
            vae=vae,
            references=chain,
            target_latent=runtime_latent,
            positive_prompt=positive_prompt,
            negative_prompt=KREA2_EDIT_NEGATIVE_PROMPT,
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

        if apply_krea2_edit_patch and reference_count:
            patched_model = patch_krea2_model(model=model)
            runtime_mode = "ccc"
        else:
            patched_model = model
            runtime_mode = "none"

        lines = [
            "=== Krea2 CcC Edit Report ===",
            "Target Latent: external Krea2 CcC Latent",
            f"Visual References: {len(visual_entries)}",
            f"Semantic References: {len(semantic_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
            "Conditioning Path: orchestrated Krea2 Edit preparation",
            (
                "Appearance Transport: CONDITIONING metadata"
                if reference_count
                else "Appearance Transport: none"
            ),
            "Qwen Contract: positional visual blocks with optional Image N prompt annotations",
            "Negative Prompt: fixed empty string",
            (
                "Edit Runtime: CcC Krea2"
                if runtime_mode == "ccc"
                else "Edit Runtime: patch disabled"
            ),
            f"CcC Forward Override: {'yes' if runtime_mode == 'ccc' else 'no'}",
        ]

        if reference_count:
            lines.append(
                f"Positive Appearance Runtime: {_format_conditioning_runtime(positive, reference_count)}"
            )
            lines.append(
                f"Negative Appearance Runtime: {_format_conditioning_runtime(negative, reference_count)}"
            )

        for index, entry in enumerate(visual_entries, start=1):
            annotation = entry.prompt_annotation if entry.prompt_annotation else "<none>"
            lines.append(
                f"Visual Reference {index}: cached={'yes' if entry.cache is not None else 'no'}, "
                f"boost={entry.boost}, fit={entry.reference_fit}, "
                f"resize_method={entry.resize_method}, rope={entry.rope_position}, semantic={entry.semantic}, "
                f"semantic_resize={entry.semantic_resize}, semantic_grounding_px={entry.semantic_grounding_px}, "
                f"semantic_resize_method={entry.semantic_resize_method}, prompt_annotation={annotation}"
            )

        latent_info = latent.get("ccc_krea2_latent_info")
        if latent_info:
            lines.extend(["", str(latent_info)])

        lines.extend(["", pipeline_info])
        return patched_model, positive, negative, latent_out, "\n".join(lines)
