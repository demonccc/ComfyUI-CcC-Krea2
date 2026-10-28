"""Unified public Krea2 reference: independent Visual (VAE) and Vision (Qwen) paths.

The two downstream entry types are deliberately retained: Edit's spatial reference
patch consumes VisualReferenceEntry while Qwen guidance consumes VisionReferenceEntry.
They are emitted by a single user-facing node, in the original reference order.
"""
from typing import Optional
from dataclasses import replace
import torch

from ..constants import NODE_CATEGORY
from ..geometry import resize_tensor
from .edit_reference_types import EditReferenceChain, ImageReferenceEntry
from .visual_reference_node import CcCKrea2VisualReference
from .mask_node import MaskChain
from .vision_reference_node import CcCKrea2VisionReference


class CcCKrea2Reference:
    CATEGORY = NODE_CATEGORY
    FUNCTION = "process"
    RETURN_TYPES = ("KREA2_REFERENCE_CHAIN", "STRING")
    RETURN_NAMES = ("references", "reference_info")
    DESCRIPTION = (
        "One source image, independently usable as a Krea2 Edit spatial/VAE Visual "
        "reference, Qwen Vision guidance, or both. Visual controls come first."
    )

    def __init__(self):
        self._visual = CcCKrea2VisualReference()
        self._vision = CcCKrea2VisionReference()

    @classmethod
    def INPUT_TYPES(cls):
        visual = CcCKrea2VisualReference.INPUT_TYPES()
        vision = CcCKrea2VisionReference.INPUT_TYPES()
        required = {"image": ("IMAGE",), "vae_rope_enable": ("BOOLEAN", {"default": True})}
        for name, spec in visual["required"].items():
            if name != "image" and name not in ("semantic", "semantic_resize", "semantic_grounding_px", "prompt_annotation"):
                required["vae_rope_" + name] = spec
        required["vision_enable"] = ("BOOLEAN", {"default": True})
        required["vision_resize"] = ("BOOLEAN", {"default": True})
        required["vision_max_edge"] = ("INT", {"default": 768, "min": 384, "max": 2048, "step": 64})
        for name, spec in vision["required"].items():
            if name != "image" and name not in ("guidance_detail", "vision_guidance"):
                required["vision_" + name] = spec
        optional = {
            "previous_references": ("KREA2_REFERENCE_CHAIN",),
            "mask_attention": ("KREA2_MASK_CHAIN",),
            "target_attention_mask": visual["optional"]["target_attention_mask"],
        }
        return {"required": required, "optional": optional}

    @classmethod
    def IS_CHANGED(cls, vision_enable=True, vision_guidance_seed_mode="fixed",
                   vision_guidance_seed=0, vision_guidance_detail="balanced",
                   vision_vision_guidance="balanced", **kwargs):
        if not vision_enable:
            return "vision-disabled"
        return CcCKrea2VisionReference.IS_CHANGED(
            guidance_seed_mode=vision_guidance_seed_mode,
            guidance_seed=vision_guidance_seed,
            guidance_detail=vision_guidance_detail,
            vision_guidance=vision_vision_guidance,
        )

    def process(self, image: torch.Tensor, vae_rope_enable=True, vision_enable=True,
                previous_references: Optional[EditReferenceChain] = None,
                reference_boost_mask=None, mask_attention=None, target_attention_mask=None, **kwargs):
        # Accept saved workflows with the previous visual control spelling.
        if "visual_enable" in kwargs:
            vae_rope_enable = bool(kwargs.pop("visual_enable"))
        for key in tuple(kwargs):
            if key.startswith("visual_"):
                kwargs.setdefault("vae_rope_" + key[len("visual_"):], kwargs.pop(key))
        chain = previous_references if previous_references is not None else EditReferenceChain()
        info = ["=== Krea2 CcC Reference ===",
                f"VAE/RoPE: {'enabled' if vae_rope_enable else 'disabled'}",
                f"Vision: {'enabled' if vision_enable else 'disabled'}"]
        attention_weight_map = None
        if mask_attention is not None:
            if not isinstance(mask_attention, MaskChain):
                raise TypeError("Reference attention must come from Krea2 CcC Mask.")
            height, width = int(image.shape[1]), int(image.shape[2])
            if mask_attention.geometry != (int(image.shape[0]), height, width):
                raise ValueError("Reference attention masks must match reference image geometry.")
            # Region weights override the global reference boost; last region wins.
            global_boost = float(kwargs.get("vae_rope_boost", 1.0))
            attention_weight_map = torch.full(
                (int(image.shape[0]), height, width),
                global_boost, dtype=torch.float32, device=image.device
            )
            for region in mask_attention.regions:
                selection = region.mask.to(image.device) > 0.5
                attention_weight_map = torch.where(
                    selection, torch.full_like(attention_weight_map, float(region.boost)),
                    attention_weight_map
                )
        if vae_rope_enable:
            visual_kwargs = {
                name: kwargs["vae_rope_" + name]
                for name in CcCKrea2VisualReference.INPUT_TYPES()["required"]
                if name not in ("image", "semantic", "semantic_resize", "semantic_grounding_px", "prompt_annotation")
                and "vae_rope_" + name in kwargs
            }
            chain, = self._visual.process(
                image=image, previous_references=chain,
                semantic=bool(vision_enable),
                semantic_resize=bool(kwargs.get("vision_resize", True)),
                semantic_grounding_px=int(kwargs.get("vision_max_edge", 768)),
                reference_boost_mask=reference_boost_mask,
                target_attention_mask=target_attention_mask, **visual_kwargs)
            if attention_weight_map is not None:
                chain = EditReferenceChain(chain.entries[:-1] + (
                    replace(chain.entries[-1], attention_weight_map=attention_weight_map),
                ))
        if vision_enable:
            vision_kwargs = {
                name: kwargs["vision_" + name]
                for name in CcCKrea2VisionReference.INPUT_TYPES()["required"]
                if name != "image" and "vision_" + name in kwargs
            }
            vision_image = image
            resize = bool(kwargs.get("vision_resize", True))
            max_edge = int(kwargs.get("vision_max_edge", 768))
            tiled_style = (
                kwargs.get("vision_mode", "semantic") == "style"
                and kwargs.get("vision_style_processing", "full") != "full"
            )
            if resize and not tiled_style:
                if max_edge < 1:
                    raise ValueError("[Krea2 CcC Reference] vision_max_edge must be positive.")
                height, width = (int(image.shape[-3]), int(image.shape[-2]))
                if max(height, width) > max_edge:
                    scale = max_edge / float(max(height, width))
                    vision_image = resize_tensor(
                        image, max(1, round(height * scale)), max(1, round(width * scale)), "bicubic"
                    )
            chain, vision_info = self._vision.process(
                image=vision_image, previous_references=chain, **vision_kwargs)
            if vae_rope_enable:
                # One source image and one logical reference: the Vision preset
                # augments the immediately preceding Visual entry in Edit.
                paired_entry = replace(chain.entries[-1], paired_visual=True, source_image=image,
                                       tile_resize=resize and tiled_style,
                                       tile_max_edge=max_edge)
                chain = EditReferenceChain(chain.entries[:-1] + (paired_entry,))
            elif tiled_style:
                tiled_entry = replace(chain.entries[-1], source_image=image, tile_resize=resize,
                                      tile_max_edge=max_edge)
                chain = EditReferenceChain(chain.entries[:-1] + (tiled_entry,))
            if not vae_rope_enable and not tiled_style:
                chain = EditReferenceChain(chain.entries[:-1] + (
                    replace(chain.entries[-1], source_image=image),
                ))
            info.append(vision_info)
        if not vae_rope_enable and not vision_enable:
            chain = chain.append(ImageReferenceEntry(image=image))
        info.append(f"Chain Entries: {len(chain.entries)}")
        return chain, "\n".join(info)
