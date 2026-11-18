"""Native Text-to-Image helper node for Krea2 CcC Edit."""

from typing import Tuple, Dict, Any, Optional
import torch

from .constants import NODE_CATEGORY, LOGGER_PREFIX, VISION_PAD_TOKEN
from .krea_geometry import (
    DEFAULT_KREA_PRESET_SIZE,
    GEOMETRY_POLICIES,
    KREA_PRESET_SIZES,
    preset_geometry,
    resolve_krea_geometry,
)
from .modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry
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


def _encode_t2i_vision(clip, prompt: str, references: VisionReferenceChain):
    """Encode both presets with Qwen, excluding Semantic image rows afterwards."""
    from .conditioning import extract_vision_spans_from_tokens, apply_vision_weights_to_conditioning
    images, image_map, annotations, items = [], [], [], []
    for index, entry in enumerate(
        (entry for entry in references.entries if isinstance(entry, VisionReferenceEntry)), start=1
    ):
        if entry.mode not in ("visual", "semantic"):
            raise ValueError(f"[Krea2 CcC T2I] Unsupported Vision mode: {entry.mode!r}")
        images.append(entry.image)
        image_map.append({"spec": entry, "physical_qwen_image_index": index})
        instruction = entry.instruction.strip()
        if instruction:
            annotations.append(f"Image {index} guidance: {instruction}")
        items.append({
            "index": index, "mode": entry.mode, "runtime_mode": entry.mode,
            "instruction": instruction, "guidance": "<none>",
            "analysis_images": 0, "generation_strategy": "multimodal-encode",
        })
    user_text = prompt or ""
    if annotations:
        user_text += "\n\n" + "\n".join(annotations)
    conditioning_prompt = (VISION_PAD_TOKEN * len(images)) + user_text
    tokens = clip.tokenize(conditioning_prompt, images=images)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    spans, _, _, _ = extract_vision_spans_from_tokens(
        tokens=tokens, physical_image_map=image_map, clip=clip,
        test_mode=getattr(clip, "is_test_dummy", False),
    )
    removed = sorted({
        row
        for (start, end), item in zip(spans, image_map)
        if item["spec"].mode == "semantic"
        for row in range(start, end)
    })
    if removed:
        updated = []
        for item in conditioning:
            if not isinstance(item, (tuple, list)) or not isinstance(item[0], torch.Tensor):
                updated.append(item)
                continue
            tensor, extras = item[0], dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}
            keep = torch.ones(tensor.shape[1], dtype=torch.bool, device=tensor.device)
            keep[removed] = False
            mask = extras.get("attention_mask")
            if isinstance(mask, torch.Tensor) and mask.shape[-1] == tensor.shape[1]:
                extras["attention_mask"] = mask[..., keep.to(mask.device)]
            else:
                extras.pop("attention_mask", None)
            updated.append([tensor[:, keep, :], extras])
        conditioning = updated
    conditioning = apply_vision_weights_to_conditioning(
        conditioning, image_map, spans, removed_row_indices=removed,
    )
    return conditioning, items, conditioning_prompt, 0


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

        runtime_to_public = {"visual": "visual", "semantic": "semantic"}
        vision_modes = [
            runtime_to_public.get(entry.mode, entry.mode)
            for entry in vision_entries
        ]

        info_lines = [
            "=== Krea2 CcC Text to Image ===",
            f"Prompt Augmentation: {'yes' if prompt_augmentation is not None else 'no'}",
            f"Vision References: {len(vision_entries)}",
            f"Vision Modes: {', '.join(vision_modes) if vision_modes else '<none>'}",
            f"Positive Conditioning: {'text and/or Qwen visual tokens according to Vision mode' if vision_entries else 'text-only'}",
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
                f"runtime={entry.mode}, instruction={entry.instruction!r}"
            )

        if vision_entries:
            info_lines.extend(
                [
                    "",
                    "T2I Vision Pipeline:",
                    "  Vision Analysis: no text generation; Semantic removes Qwen image rows",
                    "  Final Conditioning: text and/or multimodal Qwen encoding",
                    f"  Direct Vision References: {sum(item['generation_strategy'] == 'direct-qwen-image-conditioning' for item in vision_guidance_items)}",
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
