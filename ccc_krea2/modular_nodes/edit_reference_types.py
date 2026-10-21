"""Typed declarative inputs for the split Krea2 Edit nodes."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class VisualReferenceEntry:
    """One visual edit reference before Qwen/VAE preparation."""

    image: Optional[torch.Tensor] = None
    boost: float = 1.0
    reference_fit: str = "native"
    rope_mode: str = "spatial"
    reference_stack: str = "overlay"
    rope_geometry: str = "latent"
    placement_grid: str = "inside"
    grid_horizontal_position: str = "center"
    grid_vertical_position: str = "center"
    resize_method: str = "bicubic"
    semantic: bool = True
    semantic_resize: bool = True
    semantic_grounding_px: int = 768
    semantic_resize_method: str = "bicubic"
    prompt_annotation: str = ""
    attention_scope: str = "global"
    region_tag: str = ""

    @property
    def rope_position(self) -> str:
        """Compact transport representation consumed by the Krea2 patch."""
        grid = "inside" if self.reference_fit == "crop" else self.placement_grid
        geometry = "source" if self.reference_fit == "native" else self.rope_geometry
        height = self.image.shape[1] // 16 if self.image is not None else 0
        width = self.image.shape[2] // 16 if self.image is not None else 0
        return f"{grid}:{self.grid_horizontal_position}:{self.grid_vertical_position}:{self.reference_stack}:{self.rope_mode}:{geometry}:{height}:{width}"

    @property
    def resolved_fit_mode(self) -> str:
        """Return the public geometry mode consumed by the edit engine."""
        return self.reference_fit


@dataclass(frozen=True)
class VisionReferenceEntry:
    """One Qwen-only Vision Reference before vision preparation."""

    image: torch.Tensor
    mode: str = "semantic_only"
    instruction: str = ""
    processing: str = "full"
    guidance_seed_mode: str = "fixed"
    guidance_seed_configured: int = 0
    guidance_seed: int = 0
    guidance_detail: str = "balanced"
    vision_guidance: str = "balanced"


@dataclass(frozen=True)
class EditReferenceChain:
    """One ordered chain containing both appearance and Qwen-only references."""

    entries: Tuple[VisualReferenceEntry | VisionReferenceEntry, ...] = ()

    def append(self, entry: VisualReferenceEntry | VisionReferenceEntry) -> "EditReferenceChain":
        return EditReferenceChain(self.entries + (entry,))

    def __len__(self) -> int:
        return len(self.entries)


# Python-side compatibility for callers, not separate ComfyUI socket types.
VisualReferenceChain = EditReferenceChain
VisionReferenceChain = EditReferenceChain
