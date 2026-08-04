"""Typed data structures for Qwen Vision preparation, reference specifications, and reference chains."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import torch


@dataclass(frozen=True)
class VisionPrepSpec:
    """Specification describing how a vision image derivative was prepared for Qwen Vision."""
    mode: str  # "native", "adaptive", "fixed"
    semantic_min_mp: float
    semantic_max_mp: float
    semantic_fixed_mp: float
    downscale_method_requested: str
    upscale_method_requested: str
    encoder_signature: Optional[str] = None
    resolved_alignment: Optional[int] = None
    resolved_native_limits: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PreparedVisionImage:
    """Wrapper storing untouched original image and derived Qwen Vision image."""
    original_image: torch.Tensor
    vision_image: torch.Tensor
    prep_spec: VisionPrepSpec
    debug_metadata: Dict[str, Any]


@dataclass(frozen=True)
class BaseReferenceSpec:
    """Base class for declarative reference specifications."""
    role: str
    prepared_image: PreparedVisionImage
    requested_vision_slot: Optional[int]
    aliases_template: str
    parsed_aliases: Tuple[str, ...]
    extra_vision_directive: str


@dataclass(frozen=True)
class SubjectReferenceSpec(BaseReferenceSpec):
    """Declarative specification for a Subject reference."""
    attention_boost: float = 1.0
    pose_anchor: float = 0.0
    outfit_anchor: float = 0.0
    attention_mask: Optional[torch.Tensor] = None
    masked_attention_boost: float = 1.0
    masked_identity_anchor: float = 1.0
    visual_fit_mode: str = "auto"


@dataclass(frozen=True)
class SceneReferenceSpec(BaseReferenceSpec):
    """Declarative specification for a Scene reference."""
    attention_boost: float = 1.0
    scene_anchor: float = 0.0
    attention_mask: Optional[torch.Tensor] = None
    masked_attention_boost: float = 1.0
    masked_region_anchor: float = 1.0
    visual_fit_mode: str = "auto"


@dataclass(frozen=True)
class OutfitReferenceSpec(BaseReferenceSpec):
    """Declarative specification for an Outfit reference."""
    attention_boost: float = 1.0
    outfit_anchor: float = 0.0
    attention_mask: Optional[torch.Tensor] = None
    masked_attention_boost: float = 1.0
    visual_fit_mode: str = "auto"


@dataclass(frozen=True)
class StyleReferenceSpec(BaseReferenceSpec):
    """Declarative specification for a Style reference."""
    style_fidelity: float = 0.5
    style_processing: str = "2x2"  # "full", "2x2", "4x4"
    indirect_style_transfer: bool = True
    style_directive: bool = True


@dataclass(frozen=True)
class ReferenceChain:
    """Immutable sequence of reference specifications."""
    references: Tuple[BaseReferenceSpec, ...] = ()

    def append(self, spec: BaseReferenceSpec) -> "ReferenceChain":
        """Return a new ReferenceChain with spec appended."""
        return ReferenceChain(references=self.references + (spec,))
