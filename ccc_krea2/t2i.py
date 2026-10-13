"""Native Text-to-Image helper node for Krea2 CcC Edit."""

import re
from typing import Tuple, Dict, Any, Optional
import torch

from .constants import NODE_CATEGORY, LOGGER_PREFIX
from .krea_geometry import (
    DEFAULT_KREA_PRESET_SIZE,
    GEOMETRY_POLICIES,
    KREA_PRESET_SIZES,
    preset_geometry,
    resolve_krea_geometry,
)
from .conditioning import encode_krea2_qwen_context
from .identity_contract import build_grounded_positive_user_content
from .modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry
from .qwen_guidance import guidance_detail_instruction, guidance_generation_kwargs
from .reference_slots import resolve_reference_slots_and_aliases
from .reference_specs import ReferenceChain
from .style_processing import expand_style_reference_spans, slice_style_image
from .vision_reference_runtime import append_vision_references
from .prompt_augmentation import (
    CCC_KREA2_PROMPT_AUGMENTATION,
    apply_prompt_augmentation,
)


def create_empty_sd3_latent(width: int, height: int, batch_size: int = 1) -> Dict[str, Any]:
    """Generate empty SD3-compatible LATENT dictionary matching EmptySD3LatentImage contract."""
    try:
        import nodes

        if hasattr(nodes, "EmptySD3LatentImage"):
            res = nodes.EmptySD3LatentImage().generate(width=width, height=height, batch_size=batch_size)
            if isinstance(res, tuple) and len(res) > 0:
                return res[0]
            elif isinstance(res, dict):
                return res
    except Exception:
        pass

    device = "cpu"
    try:
        import comfy.model_management

        device = comfy.model_management.intermediate_device()
    except Exception:
        pass

    samples = torch.zeros((batch_size, 16, height // 8, width // 8), dtype=torch.float32, device=device)
    return {"samples": samples}


T2I_VISION_SYSTEM_PROMPTS = {
    "semantic": (
        "Analyze the supplied reference image for text-to-image guidance. Extract only semantic/content information "
        "that can be applied to the SAME subject or subjects already requested by the target prompt: subject category, "
        "non-style attributes, pose, action, objects, composition, and spatial relationships. The target prompt is "
        "authoritative. Do not introduce an additional person or duplicate a reference subject. Ignore visual medium, "
        "rendering style, line work, lighting treatment, texture, and palette. Return one concise English guidance "
        "sentence only, with no preamble and no mention of a reference image."
    ),
}


def _public_vision_mode(runtime_mode: str) -> str:
    return {
        "semantic_only": "semantic",
        "style_direct": "visual",
        "style_indirect": "style",
        "context_indirect": "context",
    }.get(runtime_mode, runtime_mode)


def _t2i_analysis_images(entry: VisionReferenceEntry):
    image = entry.image
    if entry.mode != "style_indirect" or entry.processing == "full":
        return [image]
    return slice_style_image(image, mode=entry.processing)


def _clean_generated_guidance(clip: Any, generated_ids: Any) -> str:
    text = str(clip.decode(generated_ids)).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = re.sub(r"(?is)^<think>.*?</think>\s*", "", text).strip()

    if text.startswith("~~~") and text.endswith("~~~"):
        text = text[3:-3].strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()

    text = re.sub(
        r"(?is)^\s*(?:here is|here's|guidance|description|style guidance|context guidance)\s*:\s*",
        "",
        text,
        count=1,
    )
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()

    if not text:
        raise RuntimeError("[Krea2 CcC T2I] Qwen returned empty Vision guidance.")

    if text.strip().lower() in {"user", "assistant", "system", "user:", "assistant:", "system:"}:
        raise RuntimeError(
            "[Krea2 CcC T2I] Qwen Vision analysis returned a chat-role token instead of usable guidance. "
            "The Vision reference was not applied to final conditioning."
        )

    return text


def _generate_t2i_vision_guidance(
    clip: Any,
    entry: VisionReferenceEntry,
    target_prompt: str,
):
    public_mode = _public_vision_mode(entry.mode)
    if public_mode != "semantic":
        raise ValueError(
            f"{LOGGER_PREFIX} Text guidance generation is only used for Semantic Vision References; got {public_mode!r}."
        )
    system_prompt = T2I_VISION_SYSTEM_PROMPTS["semantic"]
    analysis_images = _t2i_analysis_images(entry)

    user_instruction = str(entry.instruction or "").strip()
    detail_instruction = guidance_detail_instruction(entry.guidance_detail)
    preserve_semantic_balanced = public_mode == "semantic" and entry.guidance_detail == "balanced"
    if preserve_semantic_balanced:
        scope_instruction = (
            "Apply only information that belongs to this Vision mode. "
            "If the additional guidance asks for information outside this mode, ignore that part."
        )
    else:
        scope_instruction = (
            "MODE SCOPE IS STRICT: apply only information that belongs to this Vision mode. "
            "If any requested or observed detail belongs to another mode, omit it instead of transferring, substituting, "
            "or inventing it. The Guidance Detail preset changes only how much information is retained inside this mode; "
            "it never expands the mode's scope."
        )

    analysis_prompt = (
        "TARGET TEXT-TO-IMAGE PROMPT:\n"
        f"{target_prompt or '<empty>'}\n\n"
        f"VISION MODE: {public_mode}\n"
        f"{scope_instruction}\n"
        f"ADDITIONAL GUIDANCE: {user_instruction or '<none>'}\n"
        + (f"{detail_instruction}\n" if detail_instruction else "")
        + "\nReturn the extracted guidance now."
    )

    image_batch = torch.cat(analysis_images, dim=0)
    generation_kwargs = guidance_generation_kwargs(
        guidance_seed=entry.guidance_seed,
        guidance_detail=entry.guidance_detail,
        vision_guidance=entry.vision_guidance,
    )

    # Krea2's official ComfyUI blueprint feeds prompt-engineering instructions as
    # ordinary user text rather than through the optional system_prompt input.
    # The abliterated Qwen3-VL checkpoint used by this project can otherwise emit
    # a bare chat-role token ("user") as the first/only generated token.
    blueprint_prompt = f"{system_prompt}\n\n{analysis_prompt}".strip()
    tokens = clip.tokenize(
        blueprint_prompt,
        image=image_batch,
        skip_template=False,
        min_length=1,
        thinking=False,
        system_prompt="",
    )
    generated_ids = clip.generate(tokens, **generation_kwargs)

    try:
        guidance = _clean_generated_guidance(clip, generated_ids)
        generation_strategy = "blueprint-user-prompt"
    except RuntimeError:
        # Retry 1: ComfyUI's explicit system-prompt path. This covers checkpoints
        # that terminate immediately or emit a bare chat-role token with the
        # blueprint-style user prompt.
        tokens = clip.tokenize(
            analysis_prompt,
            image=image_batch,
            skip_template=False,
            min_length=1,
            thinking=False,
            system_prompt=system_prompt,
        )
        generated_ids = clip.generate(tokens, **generation_kwargs)

        try:
            guidance = _clean_generated_guidance(clip, generated_ids)
            generation_strategy = "system-prompt-fallback"
        except RuntimeError:
            # Retry 2: exact multimodal call shape already proven by this
            # project's Prompt Creator. Some Qwen3-VL checkpoints behave more
            # reliably with the raw image list passed through images=.
            tokens = clip.tokenize(
                analysis_prompt,
                images=analysis_images,
                min_length=1,
                thinking=False,
                system_prompt=system_prompt,
            )
            generated_ids = clip.generate(tokens, **generation_kwargs)
            guidance = _clean_generated_guidance(clip, generated_ids)
            generation_strategy = "prompt-creator-fallback"

    return guidance, len(analysis_images), generation_strategy


def _encode_t2i_direct_vision(
    clip: Any,
    prompt: str,
    entries: Tuple[VisionReferenceEntry, ...],
):
    """Encode Visual/Style/Context references as real Qwen vision conditioning.

    This reuses the same Vision Reference runtime contract as Edit:
    Visual keeps raw Qwen vision rows, Style uses the shared style extraction,
    and Context uses the shared compact context representation.
    """
    if not entries:
        tokens = clip.tokenize(prompt or "")
        return clip.encode_from_tokens_scheduled(tokens), [], prompt or "", 0

    direct_chain = VisionReferenceChain(entries=entries)
    reference_chain = append_vision_references(ReferenceChain(), direct_chain)
    resolved_refs, _ = resolve_reference_slots_and_aliases(reference_chain)

    physical_images = []
    physical_image_map = []
    direct_items = []

    for ref_item in resolved_refs:
        spec = ref_item["spec"]
        slot = ref_item["resolved_slot"]
        ref_path = getattr(spec, "reference_path", spec.role.lower())
        vision_mode = str(getattr(spec, "vision_reference_mode", "") or "")

        if ref_path == "style":
            prep_crops, _, _ = expand_style_reference_spans(spec, start_slot=slot, clip=clip)
            for crop_idx, crop_prep in enumerate(prep_crops):
                physical_index = len(physical_images) + 1
                physical_images.append(crop_prep.vision_image)
                physical_image_map.append(
                    {
                        "role": "style",
                        "slot": slot,
                        "logical_reference_id": slot,
                        "logical_role": "style",
                        "logical_vision_slot": slot,
                        "physical_qwen_image_index": physical_index,
                        "style_group_id": slot,
                        "crop_tile_index": crop_idx,
                        "image": crop_prep.vision_image,
                        "spec": spec,
                    }
                )
        else:
            if spec.prepared_image is None:
                raise ValueError(f"{LOGGER_PREFIX} Direct Vision Reference is missing its prepared Qwen image.")
            vis_img = spec.prepared_image.vision_image
            physical_index = len(physical_images) + 1
            physical_images.append(vis_img)
            physical_image_map.append(
                {
                    "role": spec.role.lower(),
                    "slot": slot,
                    "logical_reference_id": slot,
                    "logical_role": spec.role.lower(),
                    "logical_vision_slot": slot,
                    "physical_qwen_image_index": physical_index,
                    "style_group_id": None,
                    "crop_tile_index": None,
                    "image": vis_img,
                    "spec": spec,
                }
            )

        direct_items.append(
            {
                "slot": slot,
                "mode": vision_mode,
                "processing": getattr(spec, "style_processing", "full"),
                "transport": "direct-qwen-vision",
            }
        )

    user_content = build_grounded_positive_user_content(
        resolved_references=resolved_refs,
        user_prompt=prompt or "",
    )
    qwen_context = encode_krea2_qwen_context(
        clip=clip,
        prompt=user_content,
        physical_images=physical_images,
        physical_image_map=physical_image_map,
        is_positive=True,
    )
    return (
        qwen_context.conditioning,
        direct_items,
        user_content,
        len(physical_images),
    )


def _encode_t2i_vision(
    clip: Any,
    prompt: str,
    vision_references: VisionReferenceChain,
):
    """Build mixed T2I Vision conditioning.

    Semantic remains image -> generated text because that path is proven for T2I.
    Visual, Style, and Context stay as real Qwen vision conditioning, reusing
    the same shared runtime/extraction contract as Edit.
    """
    semantic_entries = []
    direct_entries = []

    for entry in vision_references.entries:
        if _public_vision_mode(entry.mode) == "semantic":
            semantic_entries.append(entry)
        else:
            direct_entries.append(entry)

    semantic_items = []
    total_analysis_images = 0
    semantic_lines = []
    for index, entry in enumerate(semantic_entries, start=1):
        guidance, analysis_image_count, generation_strategy = _generate_t2i_vision_guidance(
            clip=clip,
            entry=entry,
            target_prompt=prompt,
        )
        total_analysis_images += analysis_image_count
        semantic_items.append(
            {
                "index": index,
                "mode": "semantic",
                "runtime_mode": entry.mode,
                "processing": entry.processing,
                "instruction": entry.instruction,
                "guidance": guidance,
                "analysis_images": analysis_image_count,
                "generation_strategy": generation_strategy,
                "transport": "generated-text",
            }
        )
        semantic_lines.append(
            f"Preserve this semantic content for the requested result: {guidance}"
        )

    conditioning_prompt = prompt or ""
    if semantic_lines:
        conditioning_prompt = (
            f"{conditioning_prompt}\n\n"
            "Vision-derived semantic guidance for the same requested result:\n"
            + "\n".join(semantic_lines)
        ).strip()

    if direct_entries:
        conditioning, direct_items, effective_prompt, direct_image_count = _encode_t2i_direct_vision(
            clip=clip,
            prompt=conditioning_prompt,
            entries=tuple(direct_entries),
        )
        total_analysis_images += direct_image_count
        guidance_items = semantic_items + direct_items
        conditioning_prompt = effective_prompt
    else:
        tokens = clip.tokenize(conditioning_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        guidance_items = semantic_items

    return conditioning, guidance_items, conditioning_prompt, total_analysis_images


class CcCKrea2TextToImage:
    """CcC Krea2 - Text to Image native helper node."""

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "t2i_info")
    FUNCTION = "process"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "dimensions": (
                    ["preset", "fixed"],
                    {"default": "preset"},
                ),
                "preset_size": (
                    KREA_PRESET_SIZES,
                    {"default": DEFAULT_KREA_PRESET_SIZE},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 16384, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 16384, "step": 16},
                ),
                "geometry_policy": (
                    GEOMETRY_POLICIES,
                    {"default": "preserve_aspect_krea_bounds"},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "vision_references": ("KREA2_VISION_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        model: Any,
        clip: Any,
        prompt: str,
        dimensions: str = "preset",
        preset_size: str = DEFAULT_KREA_PRESET_SIZE,
        width: int = 1024,
        height: int = 1024,
        geometry_policy: str = "preserve_aspect_krea_bounds",
        batch_size: int = 1,
        negative_prompt: str = "",
        prompt_augmentation: Optional[Any] = None,
        vision_references: Optional[VisionReferenceChain] = None,
    ) -> Tuple[Any, Any, Any, Dict[str, Any], str]:
        if clip is None:
            raise ValueError(f"{LOGGER_PREFIX} CLIP text encoder input cannot be None.")

        # Apply immutable prompt augmentation
        eff_pos, eff_neg = apply_prompt_augmentation(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            augmentation=prompt_augmentation,
        )

        # T2I target geometry uses the same shared Krea geometry library as Latent.
        if dimensions == "preset":
            target_width, target_height = preset_geometry(preset_size)
            dimensions_label = f"preset {preset_size}"
        elif dimensions == "fixed":
            target_width, target_height = resolve_krea_geometry(
                int(width),
                int(height),
                geometry_policy,
            )
            dimensions_label = (
                f"fixed {int(width)} x {int(height)} -> {geometry_policy}"
            )
        else:
            raise ValueError(
                f"{LOGGER_PREFIX} Unsupported T2I dimensions mode: {dimensions!r}."
            )

        # Native empty SD3 latent creation
        latent = create_empty_sd3_latent(
            width=target_width,
            height=target_height,
            batch_size=batch_size,
        )

        vision_entries = tuple((vision_references or VisionReferenceChain()).entries)
        vision_guidance_items = []
        conditioning_prompt = eff_pos
        vision_analysis_images = 0
        if vision_entries:
            positive, vision_guidance_items, conditioning_prompt, vision_analysis_images = _encode_t2i_vision(
                clip=clip,
                prompt=eff_pos,
                vision_references=vision_references,
            )
        else:
            pos_tokens = clip.tokenize(eff_pos)
            positive = clip.encode_from_tokens_scheduled(pos_tokens)

        # T2I keeps the explicit negative prompt text-only.
        neg_tokens = clip.tokenize(eff_neg)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        runtime_to_public = {
            "semantic_only": "semantic",
            "style_direct": "visual",
            "style_indirect": "style",
            "context_indirect": "context",
        }
        vision_modes = [
            runtime_to_public.get(entry.mode, entry.mode)
            for entry in vision_entries
        ]

        info_lines = [
            "=== Krea2 CcC Text to Image ===",
            f"Prompt Augmentation: {'yes' if prompt_augmentation is not None else 'no'}",
            f"Vision References: {len(vision_entries)}",
            f"Vision Modes: {', '.join(vision_modes) if vision_modes else '<none>'}",
            f"Positive Conditioning: {'text-only with Qwen Vision-derived guidance' if vision_entries else 'text-only'}",
            "Negative Conditioning: text-only",
            f"Effective Positive Prompt: {eff_pos!r}",
            f"Effective Negative Prompt: {eff_neg!r}",
            f"Dimensions Mode: {dimensions}",
            f"Dimensions Source: {dimensions_label}",
            f"Resolved Resolution: {target_width} x {target_height}",
            f"Target Latent Geometry: {target_width // 8} x {target_height // 8}",
            f"Geometry Policy: {geometry_policy if dimensions == 'fixed' else '<preset>'}",
            f"Batch Size: {batch_size}",
        ]
        for index, entry in enumerate(vision_entries, start=1):
            public_mode = runtime_to_public.get(entry.mode, entry.mode)
            info_lines.append(
                f"Vision Reference {index}: mode={public_mode}, "
                f"runtime={entry.mode}, processing={entry.processing}, "
                f"instruction={entry.instruction!r}, "
                f"guidance_seed={entry.guidance_seed}, "
                f"guidance_detail={entry.guidance_detail}, "
                f"vision_guidance={entry.vision_guidance}"
            )

        if vision_entries:
            info_lines.extend(
                [
                    "",
                    "T2I Vision Pipeline:",
                    "  Vision Analysis: multimodal Qwen generation",
                    "  Final Conditioning: text-only Krea2 encode",
                    "  Direct Vision Rows in Final Conditioning: 0",
                    f"  Vision Analysis Images: {vision_analysis_images}",
                    f"  Conditioning Prompt: {conditioning_prompt!r}",
                ]
            )
            for item in vision_guidance_items:
                info_lines.extend(
                    [
                        f"  Vision Reference {item['index']}:",
                        f"    Mode: {item['mode']}",
                        f"    Analysis Images: {item['analysis_images']}",
                        f"    Generation Strategy: {item['generation_strategy']}",
                        f"    Guidance: {item['guidance']!r}",
                    ]
                )

        # Return unchanged input MODEL along with conditioning, latent, and diagnostics.
        return model, positive, negative, latent, "\n".join(info_lines)
