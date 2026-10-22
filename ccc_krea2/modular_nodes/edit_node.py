"""Krea2 CcC Edit node."""

from typing import Any, Optional
from dataclasses import replace

import torch

from .. import edit_engine as edit_engine_runtime
from ..constants import NODE_CATEGORY
from ..prompt_augmentation import CCC_KREA2_PROMPT_AUGMENTATION, apply_prompt_augmentation
from ..geometry import resize_tensor
from ..patch import attach_reference_runtime_to_conditioning, patch_krea2_model
from ..reference_specs import (
    PreparedVisionImage,
    ReferenceChain,
    ReferenceSpec,
    StyleReferenceSpec,
    VisionPrepSpec,
)
from .edit_reference_types import EditReferenceChain, VisualReferenceEntry, VisionReferenceEntry
from ..vision_reference_runtime import build_vision_reference_spec
from .rope_position import install_krea2_rope_positioning
from .rope_preview import render_rope_preview


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


# Visual references are positional in the Krea2 grounding contract.
# Prompt construction is owned directly by edit_engine.py / identity_contract.py.
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

    vision_input = image
    max_edge = max(int(image.shape[1]), int(image.shape[2]))
    resize_applied = bool(semantic_resize and grounding_px > 0 and max_edge > grounding_px)
    if resize_applied:
        scale = grounding_px / float(max_edge)
        new_h = max(1, int(round(image.shape[1] * scale)))
        new_w = max(1, int(round(image.shape[2] * scale)))
        vision_input = resize_tensor(image, target_h=new_h, target_w=new_w, method=resize_method)

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
    runtime = {"samples": latent["samples"]}
    if latent.get("batch_index") is not None:
        runtime["batch_index"] = latent["batch_index"]
    return runtime


def _combine_reference_chains(
    clip: Any,
    references: Optional[EditReferenceChain],
    latent: dict,
) -> ReferenceChain:
    """Translate the split public nodes into the common orchestrator ReferenceChain."""
    chain = ReferenceChain()

    entries = (references or EditReferenceChain()).entries
    for index, entry in enumerate(entries, start=1):
        if isinstance(entry, VisionReferenceEntry):
            if entry.paired_visual:
                if index < 2 or not isinstance(entries[index - 2], VisualReferenceEntry):
                    raise ValueError("[Krea2 CcC Edit] Paired Vision must immediately follow its Visual source.")
                # The preceding Visual has already created the VAE reference
                # and its single Qwen image. No duplicate slot should be added.
                continue
            chain = chain.append(build_vision_reference_spec(entry))
            continue
        if not isinstance(entry, VisualReferenceEntry):
            raise TypeError(f"Unexpected edit reference type: {type(entry).__name__}")
        if entry.image is None:
            raise ValueError(f"[Krea2 CcC Edit] Visual reference {index} has no IMAGE.")
        paired = (
            entries[index]
            if index < len(entries)
            and isinstance(entries[index], VisionReferenceEntry)
            and entries[index].paired_visual
            else None
        )
        prep = _prepare_qwen_image(
            image=entry.image,
            clip=clip,
            grounding_px=entry.semantic_grounding_px,
            semantic_resize=entry.semantic_resize if entry.semantic else False,
            resize_method="bicubic",
        )
        vision_spec = build_vision_reference_spec(paired) if paired else None
        if vision_spec is not None:
            # Keep the untouched original image for VAE geometry while taking
            # the Vision path's resized image and per-mode extraction policy.
            prep = replace(prep, vision_image=vision_spec.prepared_image.vision_image)
        chain = chain.append(
            ReferenceSpec(
                reference_path="edit",
                prepared_image=prep,
                alias="",
                vision_instruction=(
                    vision_spec.vision_instruction if vision_spec is not None
                    else entry.prompt_annotation if entry.semantic else ""
                ),
                appearance_reference=True,
                include_in_vision=entry.semantic,
                attention_boost=float(entry.boost),
                attention_mask=entry.reference_boost_mask,
                visual_reference_fit=entry.resolved_fit_mode,
                visual_resize_method="bicubic",
                rope_position=entry.rope_position,
                vision_reference_mode=vision_spec.vision_reference_mode if vision_spec else "",
                vision_summary_rows=vision_spec.vision_summary_rows if vision_spec else 0,
                vision_indirect=vision_spec.vision_indirect if vision_spec else False,
                semantic_extract=vision_spec.semantic_extract if vision_spec else "none",
                semantic_strength=vision_spec.semantic_strength if vision_spec else 1.0,
                style_fidelity=vision_spec.style_fidelity if vision_spec else 0.5,
                style_processing=vision_spec.style_processing if vision_spec else "full",
                indirect_style_transfer=vision_spec.indirect_style_transfer if vision_spec else False,
            )
        )

    latent_semantic = latent.get("ccc_krea2_latent_semantic") or {}
    if latent_semantic.get("enabled"):
        latent_image = latent_semantic.get("image")
        if latent_image is None:
            raise ValueError("[Krea2 CcC Edit] Latent semantic metadata is enabled but does not contain an image.")
        chain = _append_semantic(
            chain=chain,
            image=latent_image,
            clip=clip,
            instruction=str(latent_semantic.get("instruction", "")),
            grounding_px=int(latent_semantic.get("grounding_px", 768)),
            alias="target image",
        )

    return chain


def _conditioning_extras(conditioning):
    if not conditioning:
        return {}
    first = conditioning[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and isinstance(first[1], dict):
        return first[1]
    return {}


def _resolve_reference_target_masks(visual_entries, latent):
    """Read masks from their owning Visual References, never from a parallel chain."""
    samples = latent["samples"]
    target_h, target_w = (int(samples.shape[-2]) * 8, int(samples.shape[-1]) * 8)
    batch = int(samples.shape[0])
    masks, scopes = [], []
    for index, entry in enumerate(visual_entries, start=1):
        scope = entry.attention_scope
        mask = entry.target_attention_mask
        if scope != "global" and mask is None:
            raise ValueError(f"[Krea2 CcC Edit] Visual Reference {index} requires a target_attention_mask.")
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3 or tuple(mask.shape[-2:]) != (target_h, target_w):
                raise ValueError(
                    f"[Krea2 CcC Edit] Visual Reference {index} target_attention_mask must match "
                    f"the prepared target {target_w}x{target_h}; received {tuple(mask.shape)}."
                )
            if mask.shape[0] not in (1, batch):
                raise ValueError(f"[Krea2 CcC Edit] Visual Reference {index} target mask batch must be 1 or {batch}.")
        masks.append(mask if scope != "global" else None)
        scopes.append(scope)
    return masks, scopes


def _format_conditioning_runtime(conditioning, reference_count: int) -> str:
    extras = _conditioning_extras(conditioning)
    refs = list(extras.get("reference_latents") or [])
    shapes = [tuple(int(v) for v in ref.shape) for ref in refs if torch.is_tensor(ref)]
    rope = list(extras.get("reference_rope_positions") or [])
    boosts = extras.get("reference_boosts")
    if boosts is None:
        boosts = [1.0] * reference_count
    else:
        boosts = [float(v) for v in boosts]
    return f"refs={len(refs)} shapes={shapes}, boosts={boosts}, rope={rope}"


class CcCKrea2Edit:
    """Krea2 CcC Edit orchestrator consuming Krea2 CcC Latent and reference node outputs."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING", "IMAGE")
    RETURN_NAMES = ("patched_model", "positive", "negative", "latent", "edit_info", "rope_preview")
    FUNCTION = "process"
    DESCRIPTION = (
        "Krea2 CcC Edit orchestrator. Krea2 CcC Edit resolves target geometry and reference preparation while "
        "appearance refs, per-reference masks, boosts and RoPE placement travel with CONDITIONING. "
        "The grounded negative branch always uses an empty text prompt and neutral reference boosts. "
        "Krea2 CcC Edit owns a single edit runtime for all visual references."
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
                "references": ("KREA2_REFERENCE_CHAIN",),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
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
        references=None,
        prompt_augmentation=None,
    ):
        base_positive_prompt = positive_prompt
        positive_prompt, _ = apply_prompt_augmentation(
            positive_prompt, KREA2_EDIT_NEGATIVE_PROMPT, prompt_augmentation,
            positive_separator=" ",
        )
        all_entries = (references or EditReferenceChain()).entries
        visual_entries = tuple(e for e in all_entries if isinstance(e, VisualReferenceEntry))
        vision_entries = tuple(e for e in all_entries if isinstance(e, VisionReferenceEntry))
        latent_semantic = latent.get("ccc_krea2_latent_semantic") or {}
        runtime_latent = _runtime_latent(latent)

        chain = _combine_reference_chains(
            clip=clip,
            references=references,
            latent=latent,
        )

        # Krea2 CcC Edit prepares geometry, Qwen images and VAE reference latents.
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
        target_masks, attention_scopes = _resolve_reference_target_masks(visual_entries, latent)

        if reference_count:
            positive = attach_reference_runtime_to_conditioning(
                positive,
                reference_count=reference_count,
                rope_positions=rope_positions,
                reference_boosts=positive_boosts,
                reference_target_masks=target_masks,
                reference_attention_scopes=attention_scopes,
            )
            negative = attach_reference_runtime_to_conditioning(
                negative,
                reference_count=reference_count,
                rope_positions=rope_positions,
                reference_boosts=None,
                reference_target_masks=target_masks,
                reference_attention_scopes=attention_scopes,
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
            f"Vision References: {len(vision_entries)}",
            f"Latent Semantic: {'enabled' if latent_semantic.get('enabled') else 'disabled'}",
            "Conditioning Path: orchestrated Krea2 Edit preparation",
            (
                "Appearance Transport: CONDITIONING metadata"
                if reference_count
                else "Appearance Transport: none"
            ),
            "Qwen Contract: positional visual blocks with optional Image N prompt annotations",
            "Negative Prompt: fixed empty string",
            f"Prompt Augmentation: {'connected' if prompt_augmentation is not None else 'not connected'}",
            f"Base Positive Prompt: {base_positive_prompt!r}",
            f"Effective Positive Prompt: {positive_prompt!r}",
            (
                "Edit Runtime: Krea2 CcC Edit"
                if runtime_mode == "ccc"
                else "Edit Runtime: patch disabled"
            ),
            f"Krea2 CcC Edit Forward Override: {'yes' if runtime_mode == 'ccc' else 'no'}",
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
                f"Visual Reference {index}: "
                f"boost={entry.boost}, fit={entry.reference_fit}, "
                f"rope={entry.rope_position}, semantic={entry.semantic}, "
                f"attention_scope={entry.attention_scope}, reference_boost_mask={entry.reference_boost_mask is not None}, target_attention_mask={entry.target_attention_mask is not None}, "
                f"semantic_resize={entry.semantic_resize}, semantic_grounding_px={entry.semantic_grounding_px}, "
                f"semantic_resize_method=bicubic, prompt_annotation={annotation}"
            )

        latent_info = latent.get("ccc_krea2_latent_info")
        if latent_info:
            lines.extend(["", str(latent_info)])

        lines.extend(["", pipeline_info])
        preview = render_rope_preview(visual_entries, latent)
        return patched_model, positive, negative, latent_out, "\n".join(lines), preview
