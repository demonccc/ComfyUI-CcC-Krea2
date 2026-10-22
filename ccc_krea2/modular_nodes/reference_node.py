"""Unified public Krea2 reference: independent Visual (VAE) and Vision (Qwen) paths.

The two downstream entry types are deliberately retained: Edit's spatial reference
patch consumes VisualReferenceEntry while Qwen guidance consumes VisionReferenceEntry.
They are emitted by a single user-facing node, in the original reference order.
"""
from typing import Optional
import torch

from ..constants import NODE_CATEGORY
from .edit_reference_types import EditReferenceChain
from .visual_reference_node import CcCKrea2VisualReference
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
        required = {"image": ("IMAGE",), "visual_enable": ("BOOLEAN", {"default": True})}
        for name, spec in visual["required"].items():
            if name != "image" and name not in ("semantic", "semantic_resize", "semantic_grounding_px", "prompt_annotation"):
                required["visual_" + name] = spec
        required["vision_enable"] = ("BOOLEAN", {"default": False})
        for name, spec in vision["required"].items():
            if name != "image":
                required["vision_" + name] = spec
        optional = {
            "previous_references": ("KREA2_REFERENCE_CHAIN",),
            "reference_boost_mask": visual["optional"]["reference_boost_mask"],
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

    def process(self, image: torch.Tensor, visual_enable=True, vision_enable=False,
                previous_references: Optional[EditReferenceChain] = None,
                reference_boost_mask=None, target_attention_mask=None, **kwargs):
        if not visual_enable and not vision_enable:
            raise ValueError("[Krea2 CcC Reference] Enable Visual, Vision, or both.")
        chain = previous_references if previous_references is not None else EditReferenceChain()
        info = ["=== Krea2 CcC Reference ===",
                f"Visual: {'enabled' if visual_enable else 'disabled'}",
                f"Vision: {'enabled' if vision_enable else 'disabled'}"]
        if visual_enable:
            visual_kwargs = {
                name: kwargs["visual_" + name]
                for name in CcCKrea2VisualReference.INPUT_TYPES()["required"]
                if name not in ("image", "semantic", "semantic_resize", "semantic_grounding_px", "prompt_annotation")
                and "visual_" + name in kwargs
            }
            chain, = self._visual.process(
                image=image, previous_references=chain,
                semantic=False, reference_boost_mask=reference_boost_mask,
                target_attention_mask=target_attention_mask, **visual_kwargs)
        if vision_enable:
            vision_kwargs = {
                name: kwargs["vision_" + name]
                for name in CcCKrea2VisionReference.INPUT_TYPES()["required"]
                if name != "image" and "vision_" + name in kwargs
            }
            chain, vision_info = self._vision.process(
                image=image, previous_references=chain, **vision_kwargs)
            info.append(vision_info)
        info.append(f"Chain Entries: {len(chain.entries)}")
        return chain, "\n".join(info)
