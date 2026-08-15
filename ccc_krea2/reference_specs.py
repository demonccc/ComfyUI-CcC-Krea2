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
class ReferenceSpec:
    """Generic declarative specification for an edit or style reference image."""

    reference_path: str = "edit"  # "edit" or "style"
    prepared_image: Optional[PreparedVisionImage] = None
    requested_vision_slot: Optional[int] = None
    alias: str = ""
    parsed_aliases: Tuple[str, ...] = ()
    vision_instruction: str = ""

    # Edit-path parameters
    appearance_reference: bool = True
    attention_boost: float = 1.0
    masked_attention_boost: float = 1.0
    attention_mask: Optional[torch.Tensor] = None
    visual_reference_fit: str = "auto"

    # Style-path parameters
    style_fidelity: float = 0.5
    style_processing: str = "2x2"  # "full", "2x2", "4x4"
    indirect_style_transfer: bool = True
    style_directive: bool = True

    # Legacy & directive anchor fields for backward compatibility
    role: str = ""
    aliases_template: str = ""
    extra_vision_directive: str = ""
    visual_fit_mode: str = ""
    pose_anchor: float = 0.0
    outfit_anchor: float = 0.0
    masked_identity_anchor: float = 0.0
    scene_anchor: float = 0.0
    masked_region_anchor: float = 0.0

    # Internal legacy role storage
    _legacy_role: str = ""

    def __post_init__(self):
        if self.role and not self._legacy_role:
            object.__setattr__(self, "_legacy_role", self.role)
        if self._legacy_role and (not self.reference_path or self.reference_path == "edit"):
            ref_p = "style" if self._legacy_role.lower() == "style" else "edit"
            object.__setattr__(self, "reference_path", ref_p)
        if not self.role:
            object.__setattr__(self, "role", self._legacy_role if self._legacy_role else self.reference_path)
        if self.aliases_template and not self.alias:
            object.__setattr__(self, "alias", self.aliases_template)
        if self.extra_vision_directive and not self.vision_instruction:
            object.__setattr__(self, "vision_instruction", self.extra_vision_directive)
        if self.visual_fit_mode and self.visual_reference_fit == "auto":
            object.__setattr__(self, "visual_reference_fit", self.visual_fit_mode)

    @property
    def get_role(self) -> str:
        if self._legacy_role:
            return self._legacy_role
        if self.role:
            return self.role
        return self.reference_path


# Compatibility base class and role specs for existing workflows and tests
BaseReferenceSpec = ReferenceSpec


@dataclass(frozen=True)
class SubjectReferenceSpec(ReferenceSpec):
    """Deprecated compatibility wrapper for Subject reference."""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "reference_path", "edit")
        if not self._legacy_role:
            object.__setattr__(self, "_legacy_role", "subject")


@dataclass(frozen=True)
class SceneReferenceSpec(ReferenceSpec):
    """Deprecated compatibility wrapper for Scene reference."""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "reference_path", "edit")
        if not self._legacy_role:
            object.__setattr__(self, "_legacy_role", "scene")


@dataclass(frozen=True)
class OutfitReferenceSpec(ReferenceSpec):
    """Deprecated compatibility wrapper for Outfit reference."""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "reference_path", "edit")
        if not self._legacy_role:
            object.__setattr__(self, "_legacy_role", "outfit")


@dataclass(frozen=True)
class StyleReferenceSpec(ReferenceSpec):
    """Deprecated compatibility wrapper for Style reference."""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "reference_path", "style")
        if not self._legacy_role:
            object.__setattr__(self, "_legacy_role", "style")


@dataclass(frozen=True)
class ReferenceChain:
    """Immutable sequence of ReferenceSpec items."""

    specs: Tuple[ReferenceSpec, ...] = ()

    @property
    def references(self) -> Tuple[ReferenceSpec, ...]:
        return self.specs

    def append(self, spec: ReferenceSpec) -> "ReferenceChain":
        """Append a specification returning a new ReferenceChain instance."""
        return ReferenceChain(specs=self.specs + (spec,))

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx]
