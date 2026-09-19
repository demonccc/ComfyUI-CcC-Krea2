"""Typed declarative inputs for the split Krea2 Edit nodes."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass(frozen=True)
class VisualReferenceEntry:
    """One visual edit reference before Qwen/VAE preparation."""

    image: Optional[torch.Tensor] = None
    cache: Optional[Any] = None
    boost: float = 1.0
    reference_fit: str = "native"
    placement_grid: str = "inside"
    grid_horizontal_position: str = "center"
    grid_vertical_position: str = "center"
    resize_method: str = "lanczos"
    semantic: bool = True
    semantic_resize: bool = True
    semantic_grounding_px: int = 768
    semantic_resize_method: str = "lanczos"
    prompt_annotation: str = ""

    @property
    def rope_position(self) -> str:
        """Compact transport representation consumed by the Krea2 patch."""
        grid = "inside" if self.reference_fit == "crop" else self.placement_grid
        return f"{grid}:{self.grid_horizontal_position}:{self.grid_vertical_position}"

    @property
    def resolved_fit_mode(self) -> str:
        """Return the public geometry mode consumed by the edit engine."""
        return self.reference_fit


@dataclass(frozen=True)
class VisualReferenceChain:
    """Ordered visual-reference chain. Order is the Krea2 Edit reference order."""

    entries: Tuple[VisualReferenceEntry, ...] = ()

    def append(self, entry: VisualReferenceEntry) -> "VisualReferenceChain":
        return VisualReferenceChain(self.entries + (entry,))

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class SemanticReferenceEntry:
    """One Qwen-only semantic/style reference before vision preparation."""

    image: torch.Tensor
    mode: str = "semantic_only"
    instruction: str = ""
    grounding_px: int = 768
    processing: str = "2x2"
    fidelity: float = 1.0


@dataclass(frozen=True)
class SemanticReferenceChain:
    """Ordered semantic/style reference chain."""

    entries: Tuple[SemanticReferenceEntry, ...] = ()

    def append(self, entry: SemanticReferenceEntry) -> "SemanticReferenceChain":
        return SemanticReferenceChain(self.entries + (entry,))

    def __len__(self) -> int:
        return len(self.entries)
