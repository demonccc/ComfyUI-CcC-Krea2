"""Unified CcC reference with independent Qwen Vision and VAE appearance paths."""
from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain, VisionReferenceChain, VisionReferenceEntry, VisualReferenceEntry
from .visual_reference_node import CcCKrea2VisualReference, REFERENCE_FIT
from .vision_reference_node import CcCKrea2VisionReference, VISION_MODES

class CcCKrea2Reference:
    CATEGORY = NODE_CATEGORY
    FUNCTION = "process"
    RETURN_TYPES = ("KREA2_VISUAL_REFERENCE_CHAIN", "KREA2_VISION_REFERENCE_CHAIN", "STRING")
    RETURN_NAMES = ("visual_references", "vision_references", "reference_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "tag": ("STRING", {"default": "subject"}),
            "semantic_enabled": ("BOOLEAN", {"default": True}),
            "visual_enabled": ("BOOLEAN", {"default": True}),
            "vision_mode": (VISION_MODES, {"default": "semantic"}),
            "vision_instruction": ("STRING", {"multiline": True, "default": ""}),
            "vision_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "visual_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            "reference_fit": (REFERENCE_FIT, {"default": "native"}),
            "semantic_grounding_px": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
        }, "optional": {
            "previous_visual_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
            "previous_vision_references": ("KREA2_VISION_REFERENCE_CHAIN",),
        }}

    def process(self, image, tag="subject", semantic_enabled=True, visual_enabled=True,
                vision_mode="semantic", vision_instruction="", vision_strength=1.0,
                visual_boost=1.0, reference_fit="native", semantic_grounding_px=768,
                previous_visual_references=None, previous_vision_references=None):
        tag = tag.strip()
        if not tag:
            raise ValueError("Reference tag cannot be empty")
        visuals = previous_visual_references or VisualReferenceChain()
        visions = previous_vision_references or VisionReferenceChain()
        if visual_enabled:
            (visuals,) = CcCKrea2VisualReference().process(
                image=image, boost=visual_boost, reference_fit=reference_fit,
                semantic=False, previous_references=visuals)
            last = visuals.entries[-1]
            visuals = VisualReferenceChain(visuals.entries[:-1] + (VisualReferenceEntry(**{**last.__dict__, "reference_tag":tag, "vision_strength":float(vision_strength)}),))
        if semantic_enabled:
            visions, _ = CcCKrea2VisionReference().process(
                image=image, mode=vision_mode,
                instruction=vision_instruction.strip(),
                previous_references=visions)
            last = visions.entries[-1]
            visions = VisionReferenceChain(visions.entries[:-1] + (VisionReferenceEntry(**{**last.__dict__, "reference_tag":tag, "vision_strength":float(vision_strength)}),))
        report = (f"@{tag}: semantic={semantic_enabled} mode={vision_mode}; "
                  f"visual={visual_enabled} visual_boost={visual_boost}; "
                  f"vision_strength={vision_strength} (pending token-weight runtime)")
        return visuals, visions, report
