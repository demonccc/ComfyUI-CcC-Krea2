"""Krea2 CcC Visual Reference node."""

from typing import Optional, Tuple

import torch

from ..attention_regions import REFERENCE_ATTENTION_SCOPES
from ..constants import NODE_CATEGORY
from .edit_reference_types import VisualReferenceChain, VisualReferenceEntry


REFERENCE_FIT = ("crop", "resize", "contain", "native")
PLACEMENT_GRIDS = ("inside", "outside")
GRID_HORIZONTAL = ("center", "left", "right")
GRID_VERTICAL = ("center", "up", "down")
RESIZE_METHODS = ("lanczos", "bicubic", "bilinear", "area")


class CcCKrea2VisualReference:
    """Declare one ordered visual reference for Krea2 Edit."""

    CATEGORY = NODE_CATEGORY
    RETURN_TYPES = ("KREA2_VISUAL_REFERENCE_CHAIN",)
    RETURN_NAMES = ("visual_references",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Adds one ordered Krea2 Edit appearance reference. reference_fit controls only the pixel path sent to VAE: "
        "crop uses the target grid as an inside crop window positioned by the grid controls; resize maps the reference "
        "long edge to the corresponding target edge; contain scales the reference to fit inside the target; and "
        "native keeps a 1:1 source scale. resize, contain and native use minimal centered crop-down alignment to /16 "
        "instead of padding or distorting. semantic controls whether the same image is also shown to Qwen. prompt_annotation optionally "
        "adds 'Image N: ...' text after the vision prefix. attention_scope can keep the reference global, "
        "apply its boost only inside a tagged target region, or block it outside that region."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "reference_fit": (REFERENCE_FIT, {"default": "native"}),
                "placement_grid": (
                    PLACEMENT_GRIDS,
                    {
                        "default": "inside",
                        "tooltip": "RoPE placement grid. crop always forces inside.",
                    },
                ),
                "grid_horizontal_position": (GRID_HORIZONTAL, {"default": "center"}),
                "grid_vertical_position": (GRID_VERTICAL, {"default": "center"}),
                "resize_method": (
                    RESIZE_METHODS,
                    {
                        "default": "lanczos",
                        "tooltip": "Interpolation used when reference_fit is resize or contain.",
                    },
                ),
                "semantic": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, this appearance reference is also sent to Qwen.",
                    },
                ),
                "semantic_resize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, Qwen input is downscaled only when it exceeds semantic_grounding_px. Never upscales.",
                    },
                ),
                "semantic_grounding_px": (
                    "INT",
                    {"default": 768, "min": 32, "max": 4096, "step": 32},
                ),
                "semantic_resize_method": (
                    RESIZE_METHODS,
                    {
                        "default": "lanczos",
                        "tooltip": "Interpolation used only for semantic downscaling.",
                    },
                ),
                "prompt_annotation": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional text appended as 'Image N: <annotation>' for this Qwen image.",
                    },
                ),
                "attention_scope": (
                    REFERENCE_ATTENTION_SCOPES,
                    {
                        "default": "global",
                        "tooltip": "global = current behavior; boost in region = boost only inside the tagged target box; only in region = block this reference outside the tagged target box.",
                    },
                ),
                "region_tag": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Attention Region tag used when attention_scope is not global.",
                    },
                ),
            },
            "optional": {
                "previous_references": ("KREA2_VISUAL_REFERENCE_CHAIN",),
            },
        }

    def process(
        self,
        image: torch.Tensor,
        boost: float = 1.0,
        reference_fit: str = "native",
        placement_grid: str = "inside",
        grid_horizontal_position: str = "center",
        grid_vertical_position: str = "center",
        resize_method: str = "lanczos",
        semantic: bool = True,
        semantic_resize: bool = True,
        semantic_grounding_px: int = 768,
        semantic_resize_method: str = "lanczos",
        prompt_annotation: str = "",
        attention_scope: str = "global",
        region_tag: str = "",
        previous_references: Optional[VisualReferenceChain] = None,
    ) -> Tuple[VisualReferenceChain]:
        chain = previous_references if previous_references is not None else VisualReferenceChain()

        semantic_enabled = bool(semantic)
        attention_scope = str(attention_scope)
        if attention_scope not in REFERENCE_ATTENTION_SCOPES:
            raise ValueError(f"[Krea2 CcC Visual Reference] Invalid attention_scope '{attention_scope}'.")
        resolved_region_tag = str(region_tag or "").strip()
        if attention_scope != "global" and not resolved_region_tag:
            raise ValueError(
                "[Krea2 CcC Visual Reference] regional attention requires a non-empty region_tag."
            )

        entry = VisualReferenceEntry(
            image=image,
            boost=float(boost),
            reference_fit=reference_fit,
            placement_grid="inside" if reference_fit == "crop" else placement_grid,
            grid_horizontal_position=grid_horizontal_position,
            grid_vertical_position=grid_vertical_position,
            resize_method=resize_method,
            semantic=semantic_enabled,
            semantic_resize=bool(semantic_resize) if semantic_enabled else False,
            semantic_grounding_px=int(semantic_grounding_px),
            semantic_resize_method=semantic_resize_method,
            prompt_annotation=prompt_annotation.strip() if semantic_enabled else "",
            attention_scope=attention_scope,
            region_tag=resolved_region_tag if attention_scope != "global" else "",
        )
        return (chain.append(entry),)
