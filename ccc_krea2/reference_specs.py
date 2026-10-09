"""Typed data structures for Qwen Vision preparation and Krea2 CcC Edit references."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch


@dataclass(frozen=True)
class VisionPrepSpec:
    """Describe how a Qwen Vision image derivative was prepared."""

    mode: str
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
    """Store the untouched source image and the derived Qwen Vision image."""

    original_image: torch.Tensor
    vision_image: torch.Tensor
    prep_spec: VisionPrepSpec
    debug_metadata: Dict[str, Any]


@dataclass(frozen=True)
class ReferenceSpec:
    """Declarative reference used by the Krea2 CcC Edit pipeline."""

    reference_path: str = "edit"
    prepared_image: Optional[PreparedVisionImage] = None
    requested_vision_slot: Optional[int] = None
    alias: str = ""
    parsed_aliases: Tuple[str, ...] = ()
    vision_instruction: str = ""
    include_in_vision: bool = True

    # Visual/edit path
    appearance_reference: bool = True
    attention_boost: float = 1.0
    masked_attention_boost: float = 1.0
    attention_mask: Optional[torch.Tensor] = None
    visual_reference_fit: str = "native"
    visual_resize_method: str = "lanczos"
    rope_position: str = "inside:center:center"
    cached_appearance_latent: Optional[torch.Tensor] = None
    cached_geometry: Optional[Any] = None
    cached_qwen_visual: Optional[Any] = None

    # Qwen-only semantic processing
    semantic_extract: str = "none"
    semantic_strength: float = 1.0

    # Style path
    style_fidelity: float = 0.5
    style_processing: str = "2x2"
    indirect_style_transfer: bool = True

    @property
    def role(self) -> str:
        return self.reference_path


@dataclass(frozen=True)
class StyleReferenceSpec(ReferenceSpec):
    """Reference that contributes style information through Qwen Vision."""

    reference_path: str = "style"
    appearance_reference: bool = False


@dataclass(frozen=True)
class ReferenceChain:
    """Immutable ordered sequence of references."""

    specs: Tuple[ReferenceSpec, ...] = ()

    @property
    def references(self) -> Tuple[ReferenceSpec, ...]:
        return self.specs

    def append(self, spec: ReferenceSpec) -> "ReferenceChain":
        return ReferenceChain(specs=self.specs + (spec,))

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]
