"""Typed declarative inputs for the split Krea2 Edit nodes."""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class VisualReferenceEntry:
    """One visual edit reference before Qwen/VAE preparation."""

    image: torch.Tensor
    boost: float = 1.0
    fit_to_latent: bool = True
    rope_grid: str = "inside"
    rope_horizontal: str = "center"
    rope_vertical: str = "center"
    semantic: bool = True
    semantic_role: str = ""
    instruction: str = ""
    grounding_px: int = 768

    @property
    def rope_position(self) -> str:
        """Compact transport representation consumed by the Krea2 patch."""
        return f"{self.rope_grid}:{self.rope_horizontal}:{self.rope_vertical}"


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
