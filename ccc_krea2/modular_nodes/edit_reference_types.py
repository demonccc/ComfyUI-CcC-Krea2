"""Typed declarative inputs for the split Krea2 Edit nodes."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class ImageReferenceEntry:
    """Always-present image, without active conditioning."""
    image: torch.Tensor


@dataclass(frozen=True)
class VisualReferenceEntry:
    """One visual edit reference before Qwen/VAE preparation."""

    image: Optional[torch.Tensor] = None
    boost: float = 1.0
    reference_boost_mask: Optional[torch.Tensor] = None  # legacy internal contract
    attention_weight_map: Optional[torch.Tensor] = None
    reference_fit: str = "native"
    rope_mode: str = "spatial"
    reference_stack: str = "overlay"
    rope_geometry: str = "latent"
    placement_grid: str = "inside"
    grid_horizontal_position: str = "center"
    grid_vertical_position: str = "center"
    semantic: bool = True
    semantic_resize: bool = True
    semantic_grounding_px: int = 768
    prompt_annotation: str = ""

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
    """Qwen image and optional user instruction; mode controls token retention only."""

    image: torch.Tensor
    mode: str = "visual"
    instruction: str = ""
    vision_weight: float = 1.0
    # Paired with the preceding VAE/RoPE appearance reference in unified Reference.
    paired_visual: bool = False
    # Original source image (before the optional Qwen downscale).
    source_image: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class EditReferenceChain:
    """One ordered chain containing both appearance and Qwen-only references."""

    entries: Tuple[ImageReferenceEntry | VisualReferenceEntry | VisionReferenceEntry, ...] = ()

    def append(self, entry: ImageReferenceEntry | VisualReferenceEntry | VisionReferenceEntry) -> "EditReferenceChain":
        return EditReferenceChain(self.entries + (entry,))

    def __len__(self) -> int:
        return len(self.entries)


# Python-side compatibility for callers, not separate ComfyUI socket types.
VisualReferenceChain = EditReferenceChain
VisionReferenceChain = EditReferenceChain
