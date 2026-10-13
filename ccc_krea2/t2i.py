"""Native Text-to-Image helper node for Krea2 CcC Edit."""

import math
import random
import re
from typing import Tuple, Dict, Any, Optional
import torch

from .constants import NODE_CATEGORY, LOGGER_PREFIX
from .modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry
from .style_processing import slice_style_image
from .prompt_augmentation import (
    CCC_KREA2_PROMPT_AUGMENTATION,
    apply_prompt_augmentation,
)

MAX_GUIDANCE_SEED = 0xFFFFFFFFFFFFFFFF

GUIDANCE_DETAIL_PRESETS = {
    "compact": {
        "max_length": 256,
        "instruction": (
            "Keep the guidance compact. Prioritize only the most important information "
            "that belongs to this Vision mode."
        ),
    },
    "balanced": {
        "max_length": 512,
        "instruction": "",
    },
    "detailed": {
        "max_length": 1024,
        "instruction": (
            "Include all useful information that belongs to this Vision mode. Preserve relevant "
            "details rather than compressing them unnecessarily, while avoiding unrelated content."
        ),
    },
}

VISION_GUIDANCE_PRESETS = {
    "conservative": {"temperature": 0.15, "top_p": 0.85},
    "balanced": {"temperature": 0.25, "top_p": 0.90},
    "flexible": {"temperature": 0.45, "top_p": 0.95},
}

ASPECT_RATIO_MAP: Dict[str, Tuple[float, float]] = {
    "1:1": (1.0, 1.0),
    "4:3": (4.0, 3.0),
    "3:4": (3.0, 4.0),
    "16:9": (16.0, 9.0),
    "9:16": (9.0, 16.0),
    "3:2": (3.0, 2.0),
    "2:3": (2.0, 3.0),
}


def calculate_t2i_resolution(
    aspect_ratio: str,
    megapixels: float,
    custom_aspect_width: int = 1,
    custom_aspect_height: int = 1,
) -> Tuple[int, int]:
    """Calculate 16-pixel aligned dimensions for T2I based on aspect ratio and megapixels.

    Guarantees output width and height are at least 128 and multiples of 16.
    """
    if aspect_ratio in ASPECT_RATIO_MAP:
        rw, rh = ASPECT_RATIO_MAP[aspect_ratio]
    else:
        rw = max(1.0, float(custom_aspect_width))
        rh = max(1.0, float(custom_aspect_height))

    target_pixels = megapixels * 1_000_000.0
    ratio = rw / rh
    w_raw = math.sqrt(target_pixels * ratio)
    h_raw = math.sqrt(target_pixels / ratio)

    out_w = max(128, int(round(w_raw / 16.0) * 16))
    out_h = max(128, int(round(h_raw / 16.0) * 16))

    return out_w, out_h


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
    "visual": (
        "Analyze the supplied reference image for text-to-image guidance. Describe transferable visible appearance "
        "for the SAME subject/result already requested by the target prompt: face and hair traits, clothing, accessories, "
        "materials, colors, pose, and other useful visible details. The target prompt is authoritative. Do not introduce "
        "the reference subject as an additional person and do not describe this as a separate image. Return one concise "
        "English guidance sentence only, with no preamble."
    ),
    "style": (
        "Analyze the supplied reference image or tiles only for visual STYLE. Faithfully extract the source image's "
        "medium, rendering, line work, shading, palette, lighting treatment, texture, tone, and mood. Do not let style "
        "or medium words in the target prompt bias this analysis: describe the actual source style. Do not mention or "
        "transfer people, identities, faces, hair, clothing, objects, pose, action, layout, composition, or scene content. "
        "Return one concise English style guidance sentence only, with no preamble and no mention of a reference image."
    ),
    "context": (
        "Analyze the supplied reference image only for scene CONTEXT that can be applied to the target prompt: setting, "
        "environment, camera/framing, action, interaction, composition, and spatial relationships. Faithfully preserve "
        "those scene relationships from the source while the target prompt remains authoritative for subject identity and "
        "subject count. Refer to existing people only generically as the target subject or role; do not introduce an "
        "additional person just because one appears in the reference. Ignore identity, face, hair, clothing, exact "
        "appearance, medium, rendering style, palette, and texture. Return one concise English context guidance sentence "
        "only, with no preamble."
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
    guidance_seed: int = 0,
    guidance_detail: str = "balanced",
    vision_guidance: str = "balanced",
):
    public_mode = _public_vision_mode(entry.mode)
    system_prompt = T2I_VISION_SYSTEM_PROMPTS[public_mode]
    analysis_images = _t2i_analysis_images(entry)

    try:
        detail_settings = GUIDANCE_DETAIL_PRESETS[guidance_detail]
    except KeyError as exc:
        raise ValueError(
            f"{LOGGER_PREFIX} Unsupported Guidance Detail preset: {guidance_detail!r}."
        ) from exc

    try:
        sampling_settings = VISION_GUIDANCE_PRESETS[vision_guidance]
    except KeyError as exc:
        raise ValueError(
            f"{LOGGER_PREFIX} Unsupported Vision Guidance preset: {vision_guidance!r}."
        ) from exc

    user_instruction = str(entry.instruction or "").strip()
    detail_instruction = str(detail_settings["instruction"]).strip()
    analysis_prompt = (
        "TARGET TEXT-TO-IMAGE PROMPT:\n"
        f"{target_prompt or '<empty>'}\n\n"
        f"VISION MODE: {public_mode}\n"
        "Apply only information that belongs to this Vision mode. "
        "If the additional guidance asks for information outside this mode, ignore that part.\n"
        f"ADDITIONAL GUIDANCE: {user_instruction or '<none>'}\n\n"
        f"{detail_instruction + chr(10) if detail_instruction else ''}"
        "Return the extracted guidance now."
    )

    image_batch = torch.cat(analysis_images, dim=0)
    generation_kwargs = {
        "do_sample": True,
        "max_length": int(detail_settings["max_length"]),
        "temperature": float(sampling_settings["temperature"]),
        "top_k": 64,
        "top_p": float(sampling_settings["top_p"]),
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "seed": guidance_seed,
        "mtp": True,
    }

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


def _encode_t2i_vision(
    clip: Any,
    prompt: str,
    vision_references: VisionReferenceChain,
    guidance_seed: int = 0,
    guidance_detail: str = "balanced",
    vision_guidance: str = "balanced",
):
    """Convert Vision References into mode-scoped text guidance, then encode text-only conditioning.

    This is intentionally different from Edit. The image is used only during Qwen analysis;
    no direct image rows are passed into the final Krea2 conditioning. That prevents a reference
    person or composition from competing with the user's T2I prompt while still letting the four
    Vision modes contribute distinct information.
    """
    guidance_items = []
    total_analysis_images = 0

    for index, entry in enumerate(vision_references.entries, start=1):
        public_mode = _public_vision_mode(entry.mode)
        guidance, analysis_image_count, generation_strategy = _generate_t2i_vision_guidance(
            clip=clip,
            entry=entry,
            target_prompt=prompt,
            guidance_seed=guidance_seed,
            guidance_detail=guidance_detail,
            vision_guidance=vision_guidance,
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
        "visual": "Use these visible attributes for the same requested subject",
        "style": "Render the requested result in this visual style",
        "context": "Place the requested subject in this scene/context and composition",
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
                "aspect_ratio": (
                    ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "custom"],
                    {"default": "1:1"},
                ),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "guidance_seed_mode": (
                    ["fixed", "incremental", "random"],
                    {"default": "fixed"},
                ),
                "guidance_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_GUIDANCE_SEED, "step": 1},
                ),
                "guidance_detail": (
                    ["compact", "balanced", "detailed"],
                    {"default": "balanced"},
                ),
                "vision_guidance": (
                    ["conservative", "balanced", "flexible"],
                    {"default": "balanced"},
                ),
                "custom_aspect_width": ("INT", {"default": 1, "min": 1, "max": 32}),
                "custom_aspect_height": ("INT", {"default": 1, "min": 1, "max": 32}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "prompt_augmentation": (CCC_KREA2_PROMPT_AUGMENTATION,),
                "vision_references": ("KREA2_VISION_REFERENCE_CHAIN",),
            },
        }

    @classmethod
    def IS_CHANGED(
        cls,
        guidance_seed_mode: str = "fixed",
        guidance_seed: int = 0,
        **kwargs,
    ):
        """Force Qwen guidance regeneration for non-fixed seed modes."""
        if guidance_seed_mode in {"incremental", "random"}:
            return float("nan")
        return f"guidance-seed:{int(guidance_seed)}"

    def _resolve_guidance_seed(self, mode: str, configured_seed: int) -> int:
        """Resolve the Qwen-only seed once per T2I execution."""
        seed = int(configured_seed)
        if not 0 <= seed <= MAX_GUIDANCE_SEED:
            raise ValueError(
                f"{LOGGER_PREFIX} Guidance seed must be between 0 and {MAX_GUIDANCE_SEED}."
            )

        if mode == "fixed":
            self._guidance_seed_last_mode = mode
            return seed

        if mode == "incremental":
            if getattr(self, "_guidance_seed_last_mode", None) != "incremental":
                self._guidance_seed_counter = seed
            resolved = int(self._guidance_seed_counter)
            self._guidance_seed_counter = (resolved + 1) & MAX_GUIDANCE_SEED
            self._guidance_seed_last_mode = mode
            return resolved

        if mode == "random":
            self._guidance_seed_last_mode = mode
            return random.SystemRandom().randint(0, MAX_GUIDANCE_SEED)

        raise ValueError(
            f"{LOGGER_PREFIX} Unsupported guidance seed mode: {mode!r}."
        )

    def process(
        self,
        model: Any,
        clip: Any,
        prompt: str,
        aspect_ratio: str = "1:1",
        megapixels: float = 1.0,
        batch_size: int = 1,
        guidance_seed_mode: str = "fixed",
        guidance_seed: int = 0,
        guidance_detail: str = "balanced",
        vision_guidance: str = "balanced",
        custom_aspect_width: int = 1,
        custom_aspect_height: int = 1,
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

        # T2I resolution calculation
        width, height = calculate_t2i_resolution(
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            custom_aspect_width=custom_aspect_width,
            custom_aspect_height=custom_aspect_height,
        )

        # Native empty SD3 latent creation
        latent = create_empty_sd3_latent(width=width, height=height, batch_size=batch_size)

        vision_entries = tuple((vision_references or VisionReferenceChain()).entries)
        vision_guidance = []
        conditioning_prompt = eff_pos
        vision_analysis_images = 0
        resolved_guidance_seed = None
        if vision_entries:
            resolved_guidance_seed = self._resolve_guidance_seed(
                guidance_seed_mode,
                guidance_seed,
            )
            positive, vision_guidance, conditioning_prompt, vision_analysis_images = _encode_t2i_vision(
                clip=clip,
                prompt=eff_pos,
                vision_references=vision_references,
                guidance_seed=resolved_guidance_seed,
                guidance_detail=guidance_detail,
                vision_guidance=vision_guidance,
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
            f"Aspect Ratio: {aspect_ratio}",
            f"Resolved Resolution: {width} x {height}",
            f"Batch Size: {batch_size}",
            f"Guidance Seed Mode: {guidance_seed_mode}",
            f"Guidance Seed Configured: {int(guidance_seed)}",
            f"Guidance Seed Used: {resolved_guidance_seed if resolved_guidance_seed is not None else '<not used>'}",
            f"Guidance Detail: {guidance_detail}",
            f"Guidance Max Tokens: {GUIDANCE_DETAIL_PRESETS[guidance_detail]['max_length']}",
            f"Vision Guidance: {vision_guidance}",
            f"Vision Temperature: {VISION_GUIDANCE_PRESETS[vision_guidance]['temperature']}",
            f"Vision Top P: {VISION_GUIDANCE_PRESETS[vision_guidance]['top_p']}",
            "Vision Top K: 64",
            "Vision Min P: 0.05",
            "Vision Repetition Penalty: 1.05",
        ]
        for index, entry in enumerate(vision_entries, start=1):
            public_mode = runtime_to_public.get(entry.mode, entry.mode)
            info_lines.append(
                f"Vision Reference {index}: mode={public_mode}, "
                f"runtime={entry.mode}, processing={entry.processing}, "
                f"instruction={entry.instruction!r}"
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
            for item in vision_guidance:
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
