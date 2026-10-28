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
from .modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry
from .qwen_guidance import guidance_detail_instruction, guidance_generation_kwargs
from .vision_presets import VISION_DIRECTIVES
from .style_processing import slice_style_image
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


LEGACY_VISUAL_SYSTEM_PROMPT = (
    "Analyze the supplied reference image for text-to-image guidance. Describe transferable visible appearance "
    "for the SAME subject/result already requested by the target prompt: face and hair traits, clothing, accessories, "
    "materials, colors, pose, and other useful visible details. The target prompt is authoritative. Do not introduce "
    "the reference subject as an additional person and do not describe this as a separate image. Return one concise "
    "English guidance sentence only, with no preamble."
)


T2I_VISION_SYSTEM_PROMPTS = {
    "semantic": (
        "Analyze the supplied reference image for text-to-image guidance. Extract only semantic/content information "
        "that can be applied to the SAME subject or subjects already requested by the target prompt: subject category, "
        "non-style attributes, pose, action, objects, composition, and spatial relationships. The target prompt is "
        "authoritative. Do not introduce an additional person or duplicate a reference subject. Ignore visual medium, "
        "rendering style, line work, lighting treatment, texture, and palette. Return one concise English guidance "
        "sentence only, with no preamble and no mention of a reference image."
    ),
    "visual": (
        "Analyze the supplied reference image as the authoritative FULL VISUAL BLUEPRINT for the desired result. "
        "The goal of Visual mode is to reproduce the reference as closely as possible, except for changes explicitly "
        "required by the target prompt. Extract every important visible fact needed for a near-copy: subject appearance, "
        "face and hair traits, visible body traits, exact clothing and layers, accessories, footwear, materials, exact "
        "colors and patterns, pose, body/head/hand positions, gaze, expression, objects, environment and background, "
        "camera/framing/viewpoint, composition, spatial relationships, lighting, palette, texture, rendering medium, and "
        "visual style. Preserve concrete specifics; do not generalize distinctive details into vague equivalents. "
        "The target prompt's subject count is authoritative. Never duplicate, clone, repeat, mirror, tile, or create "
        "alternate versions of a subject. Produce one coherent image, not a contact sheet, storyboard, multi-panel layout, "
        "or set of variants. Do not introduce extra subjects. Return direct English generation guidance with no preamble "
        "and no mention of a reference image."
    ),
    "style": LEGACY_VISUAL_SYSTEM_PROMPT,
    "native": (
        "Describe the supplied image directly and accurately, without applying semantic, "
        "style or context extraction rules. Return a concise English description "
        "of the visible content only."
    ),
    "context": (
        "Analyze the supplied reference image only for scene CONTEXT and staging. Reproduce the situation, not the "
        "subject's visual identity. Capture setting and environment, camera/framing/viewpoint, subject count and scene "
        "roles, full-body pose and body orientation, head orientation, gaze direction, relevant facial expression, "
        "arm/hand placement, gesture, action, interaction, object relationships, composition, foreground/midground/"
        "background layout, scale, depth, and spatial relationships. Treat every visible staging cue as a constraint, not "
        "a suggestion: pose, gaze, head angle, hands, gesture, camera, crop, framing, interaction, object placement, and "
        "relative positions must be retained unless the target prompt explicitly conflicts with them. Do not summarize "
        "them away. The target prompt remains authoritative for identity, appearance, and subject count. Never duplicate "
        "or add subjects just because they appear in the context reference. Refer to people only generically as the target "
        "subject or scene role. Do not transfer face identity, hair design, clothing design or color, accessories, rendering "
        "medium, visual style, palette treatment, or texture. Return direct English context guidance with no preamble and "
        "no mention of a reference image."
    ),
}


def _public_vision_mode(runtime_mode: str) -> str:
    return {
        "semantic_only": "semantic",
        "style_direct": "visual",
        "style_indirect": "style",
        "context_indirect": "context",
        "native": "native",
        "guided": "native",
    }.get(runtime_mode, runtime_mode)


def _t2i_analysis_images(entry: VisionReferenceEntry):
    image = entry.image
    if entry.mode != "style_indirect" or entry.processing == "full":
        return [image]
    crops = slice_style_image(image, mode=entry.processing)
    if entry.tile_resize:
        from .geometry import resize_tensor
        if entry.tile_max_edge < 1:
            raise ValueError("Vision Style tile_max_edge must be positive.")
        resized = []
        for crop in crops:
            h, w = int(crop.shape[1]), int(crop.shape[2])
            if max(h, w) > entry.tile_max_edge:
                scale = entry.tile_max_edge / max(h, w)
                crop = resize_tensor(
                    crop, max(1, round(h * scale)),
                    max(1, round(w * scale)), "bicubic",
                )
            resized.append(crop)
        return resized
    return crops


def _mode_guidance_policy(
    public_mode: str,
    guidance_detail: str,
    vision_guidance: str,
) -> str:
    """Return mode-specific extraction/anchoring policy without changing Semantic."""

    if public_mode == "visual":
        if vision_guidance == "flexible":
            variation = (
                "VISUAL VARIATION POLICY: flexible. Preserve the reference's subject appearance, clothing, colors, "
                "materials, accessories, rendering, and overall visual identity, but allow pose, gaze, camera/framing, "
                "and composition to change when that produces a better result for the target prompt. Subject count is "
                "never flexible: do not clone, repeat, tile, mirror, or create multiple variants of the same subject."
            )
            detail = {
                "compact": (
                    "VISUAL DETAIL POLICY: compact. Preserve the dominant appearance, outfit, palette, rendering, and "
                    "overall silhouette; fine pose geometry may be simplified."
                ),
                "balanced": (
                    "VISUAL DETAIL POLICY: balanced. Preserve all major appearance and drawing details faithfully while "
                    "allowing the flexible variation policy to adapt pose and composition."
                ),
                "detailed": (
                    "VISUAL DETAIL POLICY: detailed. Preserve fine appearance, clothing, color, material, line/rendering, "
                    "lighting, and background details exhaustively while the flexible policy may still adapt pose, gaze, "
                    "camera, and composition."
                ),
            }[guidance_detail]
            return f"{variation}\n{detail}"

        variation = (
            "VISUAL VARIATION POLICY: locked. Treat pose, limb placement, hand position, head angle, gaze, expression, "
            "camera/viewpoint, crop, framing, composition, spatial layout, and subject count as anchors. Do not change "
            "them unless the target prompt explicitly requests that specific change. Never clone or repeat a subject."
            if vision_guidance == "conservative"
            else
            "VISUAL VARIATION POLICY: stable. Preserve pose, limb placement, head angle, gaze, expression, camera/framing, "
            "composition, spatial layout, and exact target subject count. Adapt pose/composition only when the target prompt "
            "explicitly conflicts with the reference. Never clone, repeat, mirror, or multiply a subject."
        )
        detail = {
            "compact": (
                "VISUAL DETAIL POLICY: compact. Keep the same overall drawing, silhouette, pose, gaze, framing, outfit, "
                "palette, and rendering; omit only minor surface details."
            ),
            "balanced": (
                "VISUAL DETAIL POLICY: balanced. Strongly anchor the same drawing: preserve exact body pose, limb and hand "
                "placement, head orientation, gaze, expression, framing, crop, camera angle, composition, subject placement, "
                "outfit, colors, and rendering characteristics. Describe each subject once as one coherent figure; do not "
                "turn different observed details into separate copies or variants."
            ),
            "detailed": (
                "VISUAL DETAIL POLICY: detailed. Use the reference as a strict geometry and appearance blueprint. Preserve "
                "exact body/limb/hand geometry, head angle, gaze target, expression, crop, camera/viewpoint, composition, "
                "relative positions, outfit construction, colors, materials, background layout, lighting, texture, and "
                "rendering details. More detail means MORE anchoring, not more invention. Keep each subject as one single "
                "instance in one coherent image; never expand detailed observations into clones, variants, or panels."
            ),
        }[guidance_detail]
        return f"{variation}\n{detail}"

    if public_mode == "context":
        variation = {
            "conservative": (
                "CONTEXT VARIATION POLICY: locked. Reproduce the staging literally: pose, head angle, gaze direction, "
                "gesture, interaction, camera, crop, composition, object placement, and spatial relationships are anchors."
            ),
            "balanced": (
                "CONTEXT VARIATION POLICY: stable. Treat the observed staging, pose, head angle, gaze, gesture, limb/hand "
                "placement, framing, camera, interaction, object placement, and spatial layout as required constraints. "
                "Change them only when the target prompt explicitly conflicts with them."
            ),
            "flexible": (
                "CONTEXT VARIATION POLICY: flexible. Preserve the situation, roles, environment, and key relationships, "
                "but allow pose, gaze, gesture, camera, crop, and composition to adapt to the target prompt."
            ),
        }[vision_guidance]
        detail = {
            "compact": (
                "CONTEXT DETAIL POLICY: compact. Even when compact, always retain the main pose, head direction, gaze, "
                "primary gesture/action, camera/framing, and key spatial relationship."
            ),
            "balanced": (
                "CONTEXT DETAIL POLICY: balanced. Explicitly capture every visible staging cue: body orientation and pose, "
                "head angle, gaze target, expression when relevant, shoulders, arms, hands, leg stance, gesture, interaction, "
                "camera angle, viewpoint, crop, framing, foreground/midground/background placement, object positions, scale, "
                "depth, and relative distances. State these explicitly and preserve them as mandatory staging constraints; "
                "do not collapse them into a generic scene summary or normalize them into a generic pose."
            ),
            "detailed": (
                "CONTEXT DETAIL POLICY: detailed. Reconstruct the scene geometry exhaustively: exact pose and limb placement, "
                "left/right relationships, head and eye direction, hand contact/gesture, weight distribution, action phase, "
                "camera height and angle, crop boundaries, subject position in frame, overlap/occlusion, object placement, "
                "foreground/midground/background layers, depth, distances, and all meaningful spatial relationships. "
                "Treat every recovered staging detail as mandatory unless the target prompt explicitly changes it."
            ),
        }[guidance_detail]
        return f"{variation}\n{detail}"

    return ""


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
    system_prompt = VISION_DIRECTIVES[public_mode] + "\n\n" + T2I_VISION_SYSTEM_PROMPTS[public_mode]
    analysis_images = _t2i_analysis_images(entry)

    user_instruction = str(entry.instruction or "").strip()

    # Semantic is already known-good and keeps its current behavior.
    # Style intentionally reuses the exact legacy Visual preset/analysis behavior.
    if public_mode in {"semantic", "style"}:
        detail_instruction = (
            ""
            if (public_mode == "semantic" and entry.guidance_detail == "balanced")
            or public_mode == "style"
            else guidance_detail_instruction(entry.guidance_detail)
        )
        mode_policy = ""
    else:
        detail_instruction = guidance_detail_instruction(entry.guidance_detail)
        mode_policy = _mode_guidance_policy(
            public_mode,
            entry.guidance_detail,
            entry.vision_guidance,
        )

    analysis_mode_label = "visual" if public_mode == "style" else public_mode
    analysis_prompt = (
        "TARGET TEXT-TO-IMAGE PROMPT:\n"
        f"{target_prompt or '<empty>'}\n\n"
        f"VISION MODE: {analysis_mode_label}\n"
        "Apply only information that belongs to this Vision mode. "
        "If the additional guidance asks for information outside this mode, ignore that part.\n"
        f"ADDITIONAL GUIDANCE: {user_instruction or '<none>'}\n"
        + (f"{detail_instruction}\n" if detail_instruction else "")
        + (f"{mode_policy}\n" if mode_policy else "")
        + "\nReturn the extracted guidance now."
    )

    image_batch = torch.cat(analysis_images, dim=0)
    generation_kwargs = guidance_generation_kwargs(
        guidance_seed=entry.guidance_seed,
        guidance_detail=entry.guidance_detail,
        vision_guidance=entry.vision_guidance,
    )

    # Keep the original first-pass path that produced the known-good Semantic results.
    # IMPORTANT: exactly one clip.generate() call per Vision Reference. The old fallback
    # chain could call Qwen two more times after an empty/invalid result; on this runtime
    # those consecutive generations can trigger a CUDA ScatterGather device-side assert.
    blueprint_prompt = f"{system_prompt}\n\n{analysis_prompt}".strip()
    print(
        f"{LOGGER_PREFIX} T2I Qwen single-pass start: "
        f"mode={public_mode} images={len(analysis_images)} "
        f"shape={tuple(image_batch.shape)} seed={entry.guidance_seed} "
        f"detail={entry.guidance_detail} vision_guidance={entry.vision_guidance} "
        f"thinking=krea2-default",
        flush=True,
    )
    tokens = clip.tokenize(
        blueprint_prompt,
        image=image_batch,
        skip_template=False,
        min_length=1,
        system_prompt="",
    )
    generated_ids = clip.generate(tokens, **generation_kwargs)
    raw_text = str(clip.decode(generated_ids))
    print(
        f"{LOGGER_PREFIX} T2I Qwen single-pass result: "
        f"mode={public_mode} decoded={raw_text!r}",
        flush=True,
    )
    guidance = _clean_generated_guidance(clip, generated_ids)
    generation_strategy = "blueprint-user-prompt-single-pass"

    return guidance, len(analysis_images), generation_strategy


def _encode_t2i_vision(
    clip: Any,
    prompt: str,
    references: VisionReferenceChain,
):
    """Convert Vision References into mode-scoped text guidance, then encode text-only conditioning.

    This is intentionally different from Edit. The image is used only during Qwen analysis;
    no direct image rows are passed into the final Krea2 conditioning. That prevents a reference
    person or composition from competing with the user's T2I prompt while still letting the four
    Vision modes contribute distinct information.
    """
    guidance_items = []
    total_analysis_images = 0

    for index, entry in enumerate((e for e in references.entries if isinstance(e, VisionReferenceEntry)), start=1):
        public_mode = _public_vision_mode(entry.mode)
        guidance, analysis_image_count, generation_strategy = _generate_t2i_vision_guidance(
            clip=clip,
            entry=entry,
            target_prompt=prompt,
        )
        total_analysis_images += analysis_image_count
        guidance_items.append(
            {
                "index": index,
                "mode": public_mode,
                "runtime_mode": entry.mode,
                "processing": entry.processing,
                "instruction": entry.instruction,
                "guidance": guidance,
                "analysis_images": analysis_image_count,
                "generation_strategy": generation_strategy,
            }
        )

    guidance_prefix = {
        "semantic": "Preserve this semantic content for the requested result",
        "visual": (
            "MANDATORY VISUAL BLUEPRINT: reproduce the same drawing as faithfully as possible. Preserve pose, gaze, head "
            "angle, limb/hand placement, framing, crop, composition, appearance, outfit, colors, background layout, and "
            "rendering. Preserve the target prompt's exact subject count and NEVER clone, repeat, mirror, tile, panelize, "
            "or create alternate copies of a subject unless explicitly requested"
        ),
        "style": (
            "STRONG STYLE REFERENCE: apply the following reference-derived visible treatment as the dominant visual "
            "direction across the entire result. Prioritize its rendering, line/shading treatment, palette, lighting, "
            "texture, finish, and overall look over generic style cues while keeping the target prompt's subject and scene"
        ),
        "context": (
            "MANDATORY STAGING CONSTRAINTS: reproduce the pose, body orientation, head direction, gaze target, expression "
            "when relevant, limb/hand placement, gesture, action, framing, camera/viewpoint, crop, interaction, object "
            "placement, composition, depth, and spatial arrangement. Do not simplify these into a generic pose or scene. "
            "Keep the target prompt's identity, appearance, and exact subject count"
        ),
    }
    guidance_lines = [
        f"{guidance_prefix[item['mode']]}: {item['guidance']}"
        for item in guidance_items
    ]
    conditioning_prompt = prompt or ""
    if guidance_lines:
        conditioning_prompt = (
            f"{conditioning_prompt}\n\n"
            "Vision-derived guidance for the same requested result:\n"
            + "\n".join(guidance_lines)
        ).strip()

    tokens = clip.tokenize(conditioning_prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return conditioning, guidance_items, conditioning_prompt, total_analysis_images


class CcCKrea2TextToImage:
    """Krea2 CcC Text To Image native helper node."""

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
                "references": ("KREA2_REFERENCE_CHAIN",),
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
        references: Optional[VisionReferenceChain] = None,
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

        vision_entries = tuple(e for e in (references or VisionReferenceChain()).entries if isinstance(e, VisionReferenceEntry))
        vision_guidance_items = []
        conditioning_prompt = eff_pos
        vision_analysis_images = 0
        if vision_entries:
            positive, vision_guidance_items, conditioning_prompt, vision_analysis_images = _encode_t2i_vision(
                clip=clip,
                prompt=eff_pos,
                references=references,
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
