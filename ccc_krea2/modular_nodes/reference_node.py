"""Unified CcC reference with independent Qwen Vision and VAE appearance paths."""
from ..constants import NODE_CATEGORY
from ..qwen_guidance import GUIDANCE_DETAIL_MODES, GUIDANCE_SEED_MODES, MAX_GUIDANCE_SEED, VISION_GUIDANCE_MODES
from .edit_reference_types import VisualReferenceChain, VisionReferenceChain, VisionReferenceEntry, VisualReferenceEntry
from .visual_reference_node import CcCKrea2VisualReference, REFERENCE_FIT, PLACEMENT_GRIDS, GRID_HORIZONTAL, GRID_VERTICAL, RESIZE_METHODS
from .vision_reference_node import CcCKrea2VisionReference, VISION_MODES, STYLE_PROCESSING

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
            "placement_grid": (PLACEMENT_GRIDS, {"default": "inside"}),
            "grid_horizontal_position": (GRID_HORIZONTAL, {"default": "center"}),
            "grid_vertical_position": (GRID_VERTICAL, {"default": "center"}),
            "resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
            "semantic_resize": ("BOOLEAN", {"default": True}),
            "semantic_resize_method": (RESIZE_METHODS, {"default": "lanczos"}),
            "style_processing": (STYLE_PROCESSING, {"default": "full"}),
            "guidance_seed_mode": (GUIDANCE_SEED_MODES, {"default": "fixed"}),
            "guidance_seed": ("INT", {"default": 0, "min": 0, "max": MAX_GUIDANCE_SEED, "step": 1}),
            "guidance_detail": (GUIDANCE_DETAIL_MODES, {"default": "balanced"}),
            "vision_guidance": (VISION_GUIDANCE_MODES, {"default": "balanced"}),
            "previous_visual_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
            "previous_vision_references": ("KREA2_VISION_REFERENCE_CHAIN",),
        }}

    def process(self, image, tag="subject", semantic_enabled=True, visual_enabled=True,
                vision_mode="semantic", vision_instruction="", vision_strength=1.0,
                visual_boost=1.0, reference_fit="native", semantic_grounding_px=768,
                previous_visual_references=None, previous_vision_references=None,
        placement_grid="inside", grid_horizontal_position="center", grid_vertical_position="center",
        resize_method="lanczos", semantic_resize=True, semantic_resize_method="lanczos",
        style_processing="full", guidance_seed_mode="fixed", guidance_seed=0,
        guidance_detail="balanced", vision_guidance="balanced"):
        tag = tag.strip()
        if not tag:
            raise ValueError("Reference tag cannot be empty")
        visuals = previous_visual_references or VisualReferenceChain()
        visions = previous_vision_references or VisionReferenceChain()
        if any(getattr(item, "reference_tag", "") == tag for item in visuals.entries):
            raise ValueError(f"Duplicate reference tag '@{tag}' in visual chain")
        if any(getattr(item, "reference_tag", "") == tag for item in visions.entries):
            raise ValueError(f"Duplicate reference tag '@{tag}' in Qwen chain")
        if visual_enabled:
            (visuals,) = CcCKrea2VisualReference().process(
                image=image, boost=visual_boost, reference_fit=reference_fit,
                placement_grid=placement_grid, grid_horizontal_position=grid_horizontal_position,
                grid_vertical_position=grid_vertical_position, resize_method=resize_method,
                semantic_resize=semantic_resize, semantic_grounding_px=semantic_grounding_px,
                semantic_resize_method=semantic_resize_method,
                semantic=bool(semantic_enabled), prompt_annotation=vision_instruction.strip() if semantic_enabled else "", previous_references=visuals)
            last = visuals.entries[-1]
            visuals = VisualReferenceChain(visuals.entries[:-1] + (VisualReferenceEntry(**{**last.__dict__, "reference_tag":tag, "vision_strength":float(vision_strength)}),))
        if semantic_enabled:
            visions, _ = CcCKrea2VisionReference().process(
                image=image, mode=vision_mode,
                instruction=vision_instruction.strip(), style_processing=style_processing,
                guidance_seed_mode=guidance_seed_mode, guidance_seed=guidance_seed,
                guidance_detail=guidance_detail, vision_guidance=vision_guidance,
                previous_references=visions)
            last = visions.entries[-1]
            visions = VisionReferenceChain(visions.entries[:-1] + (VisionReferenceEntry(**{**last.__dict__, "reference_tag":tag, "vision_strength":float(vision_strength)}),))
        report = (f"@{tag}: semantic={semantic_enabled} mode={vision_mode}; "
                  f"visual={visual_enabled} visual_boost={visual_boost}; "
                  f"vision_strength={vision_strength} (applies only when Edit.vision_attention_patch is ON)")
        return visuals, visions, report
