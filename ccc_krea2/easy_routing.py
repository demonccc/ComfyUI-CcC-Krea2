"""Easy Edit 3-Phase Routing Engine for opinionated Krea2 Edit and Ostris Edit workflows."""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any


# Centralized boost constants for Easy presets
BALANCED_SUBJECT_BOOST = 2.5
MAX_IDENTITY_SUBJECT_BOOST = 4.0
PRESERVE_IDENTITY_SUBJECT_BOOST = 4.0
PRESERVE_SCENE_BOOST = 2.5
OUTFIT_EMPHASIS_BOOST = 2.5
OUTFIT_TRANSFER_BOOST = 4.0
NORMAL_BOOST = 1.0


# Centralized explicit instruction constants for Easy Edit
EASY_SUBJECT_INSTRUCTION = "Use this reference for the subject identity, facial features, hair, anatomy, body shape, and body proportions."
EASY_SCENE_INSTRUCTION = "Use this reference for scene composition, environment, spatial relationships, camera framing, and lighting."
EASY_OUTFIT_INSTRUCTION = "Use this reference for the clothing, garments, and accessories. Do not use the wearer's identity as the subject identity."
EASY_TARGET_SCENE_INSTRUCTION = "Use this image as the target scene/context."
EASY_SCENE_AND_OUTFIT_INSTRUCTION = (
    "Use this reference for the scene composition, environment, spatial relationships, "
    "camera framing and lighting, and also for the clothing, garments and accessories. "
    "Do not use the clothing wearer's identity as the subject identity."
)

EASY_ROLE_INSTRUCTIONS = {
    "subject": EASY_SUBJECT_INSTRUCTION,
    "scene": EASY_SCENE_INSTRUCTION,
    "outfit": EASY_OUTFIT_INSTRUCTION,
    "scene+outfit": EASY_SCENE_AND_OUTFIT_INSTRUCTION,
    "target_scene": EASY_TARGET_SCENE_INSTRUCTION,
    "target": EASY_TARGET_SCENE_INSTRUCTION,
}


def get_easy_instruction_for_role(role: str) -> str:
    return EASY_ROLE_INSTRUCTIONS.get(str(role).lower(), "")



@dataclass(frozen=True)
class EasyResolvedSources:
    """Phase 1: Resolved input sources with strict no-fallback rules."""
    subject: Optional[Any]
    scene: Optional[Any]
    outfit: Optional[Any]
    style: Optional[Any]
    effective_subject: Optional[Any]
    effective_scene: Optional[Any]
    effective_outfit: Optional[Any]
    effective_style: Optional[Any]
    outfit_source: str
    outfit_source_kind: str   # "outfit" | "scene" | "style"
    style_source: str
    style_source_kind: str    # "style" | "scene" | "subject"
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class EasyStyleConfig:
    """Config for Easy Edit style reference parameters using valid range values."""
    style_fidelity: float = 1.0
    style_processing: str = "2x2"
    indirect_style_transfer: bool = False
    vision_instruction: str = ""

    def __post_init__(self):
        if not (0.0 <= self.style_fidelity <= 1.0):
            raise ValueError(f"style_fidelity must be between 0.0 and 1.0, got {self.style_fidelity}")
        if self.style_processing not in ("full", "2x2", "4x4"):
            raise ValueError(f"Invalid style_processing '{self.style_processing}'. Must be 'full', '2x2', or '4x4'.")


DEFAULT_EASY_STYLE_CONFIG = EasyStyleConfig(
    style_fidelity=1.0,
    style_processing="2x2",
    indirect_style_transfer=False,
    vision_instruction=""
)

STRONG_EASY_STYLE_CONFIG = EasyStyleConfig(
    style_fidelity=1.0,
    style_processing="4x4",
    indirect_style_transfer=False,
    vision_instruction="Adopt the artistic style, color palette, texture, and visual mood of this style reference."
)


@dataclass(frozen=True)
class EasyPresetRoute:
    """Phase 2 & 3: Preset routing decision containing target content, target geometry, references, and style config."""
    preset: str
    target_content_mode: str  # "empty" or "image"
    target_content_source: Optional[Any]
    target_geometry_mode: str  # "fixed" or "favor_image"
    target_geometry_source: Optional[Any]
    target_content_role: str
    # 4-tuple: (image, boost, alias_role, instruction)
    edit_references: Tuple[Tuple[Any, float, str, str], ...]
    # 2-tuple: (image, alias_role)
    semantic_only_references: Tuple[Tuple[Any, str], ...]
    style_active: bool
    style_source: Optional[Any]
    style_config: EasyStyleConfig
    warnings: Tuple[str, ...]


def resolve_easy_sources(
    subject: Optional[Any] = None,
    scene: Optional[Any] = None,
    outfit: Optional[Any] = None,
    style: Optional[Any] = None,
    outfit_source: str = "outfit image",
    style_source: str = "style image",
) -> EasyResolvedSources:
    """Phase 1: Resolve effective sources according to selectors with strict NO-FALLBACK policy."""
    warnings: List[str] = []

    effective_subject = subject
    effective_scene = scene

    # Resolve outfit_source
    outfit_source_kind = "outfit"
    if outfit_source == "outfit image":
        effective_outfit = outfit
        outfit_source_kind = "outfit"
    elif outfit_source == "scene image":
        outfit_source_kind = "scene"
        if scene is not None:
            effective_outfit = scene
        else:
            effective_outfit = None
            warnings.append("outfit_source set to 'scene image' but scene image is disconnected.")
    elif outfit_source == "style image":
        outfit_source_kind = "style"
        if style is not None:
            effective_outfit = style
        else:
            effective_outfit = None
            warnings.append("outfit_source set to 'style image' but style image is disconnected.")
    else:
        effective_outfit = outfit

    # Resolve style_source
    style_source_kind = "style"
    if style_source == "style image":
        effective_style = style
        style_source_kind = "style"
    elif style_source == "scene image":
        style_source_kind = "scene"
        if scene is not None:
            effective_style = scene
        else:
            effective_style = None
            warnings.append("style_source set to 'scene image' but scene image is disconnected.")
    elif style_source == "subject image":
        style_source_kind = "subject"
        if subject is not None:
            effective_style = subject
        else:
            effective_style = None
            warnings.append("style_source set to 'subject image' but subject image is disconnected.")
    else:
        effective_style = style

    return EasyResolvedSources(
        subject=subject,
        scene=scene,
        outfit=outfit,
        style=style,
        effective_subject=effective_subject,
        effective_scene=effective_scene,
        effective_outfit=effective_outfit,
        effective_style=effective_style,
        outfit_source=outfit_source,
        outfit_source_kind=outfit_source_kind,
        style_source=style_source,
        style_source_kind=style_source_kind,
        warnings=tuple(warnings),
    )


def _resolve_default_geometry_source(
    scene: Optional[Any],
    subject: Optional[Any],
    outfit: Optional[Any]
) -> Tuple[str, Optional[Any]]:
    """Determine default geometry source based on hierarchy: Scene -> Subject -> Outfit."""
    if scene is not None:
        return ("favor_image", scene)
    if subject is not None:
        return ("favor_image", subject)
    if outfit is not None:
        return ("favor_image", outfit)
    return ("fixed", None)


def _combined_scene_outfit_ref(
    image: Any,
    boost: float,
) -> Tuple[Any, float, str, str]:
    """Return a 4-tuple for a ref that serves as both Scene and Outfit logically."""
    return (image, boost, "scene+outfit", EASY_SCENE_AND_OUTFIT_INSTRUCTION)


def _ref(image: Any, boost: float, alias: str) -> Tuple[Any, float, str, str]:
    """Return a 4-tuple for a standard single-role ref."""
    return (image, boost, alias, get_easy_instruction_for_role(alias))


def route_easy_preset(
    sources: EasyResolvedSources,
    preset: str = "balanced",
) -> EasyPresetRoute:
    """Phase 2: Evaluate preset routing matrix across all 6 presets.

    Exhaustive 8-combination coverage per preset:
        none, S, Sc, Ou, S+Sc, S+Ou, Sc+Ou, S+Sc+Ou
    """
    S = sources.effective_subject
    Sc = sources.effective_scene
    Ou = sources.effective_outfit
    St = sources.effective_style

    has_s = S is not None
    has_sc = Sc is not None
    has_o = Ou is not None
    has_st = St is not None

    # Detect when outfit and scene are the SAME physical object
    # This occurs when outfit_source_kind == "scene" (selector redirect) OR Ou is Sc
    outfit_is_scene_physically = has_o and has_sc and (Ou is Sc)
    # Distinct outfit = outfit exists and is not the physical scene image
    outfit_is_distinct = has_o and (not has_sc or Ou is not Sc)

    target_content_mode = "empty"
    target_content_source: Optional[Any] = None
    target_content_role = ""
    target_geometry_mode = "fixed"
    target_geometry_source: Optional[Any] = None
    refs: List[Tuple[Any, float, str, str]] = []
    semantic_only_refs: List[Tuple[Any, str]] = []
    preset_warnings: List[str] = list(sources.warnings)

    if preset in ("balanced", "style_transfer"):
        # --- balanced / style_transfer: exact 8-combination matrix ---
        if not has_s and not has_sc and not has_o:
            # none: no refs, default geometry
            pass
        elif has_s and not has_sc and not has_o:
            # S only
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, BALANCED_SUBJECT_BOOST, "subject"))
        elif not has_s and has_sc and not has_o:
            # Sc only
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
        elif not has_s and not has_sc and has_o:
            # Ou only
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Ou
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and not has_o:
            # S + Sc
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
            refs.append(_ref(S, BALANCED_SUBJECT_BOOST, "subject"))
        elif has_s and not has_sc and has_o:
            # S + Ou
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, BALANCED_SUBJECT_BOOST, "subject"))
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif not has_s and has_sc and has_o:
            # Sc + Ou
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, NORMAL_BOOST))
            else:
                refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        else:
            # S + Sc + Ou
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(S, BALANCED_SUBJECT_BOOST, "subject"))
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, NORMAL_BOOST))
            else:
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))

    elif preset == "preserve_identity":
        if not has_s:
            preset_warnings.append(
                "preset 'preserve_identity' selected but Subject source is missing; "
                "falling back to balanced appearance routing."
            )

        if not has_s and not has_sc and not has_o:
            pass
        elif has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
        elif not has_s and has_sc and not has_o:
            # Balanced fallback
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
        elif not has_s and not has_sc and has_o:
            # Balanced fallback
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Ou
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
            refs.append(_ref(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and not has_sc and has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif not has_s and has_sc and has_o:
            # Balanced fallback: Sc+Ou without S
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, NORMAL_BOOST))
            else:
                refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        else:
            # S + Sc + Ou
            if outfit_is_scene_physically:
                target_content_mode = "empty"
                target_geometry_mode, target_geometry_source = "favor_image", Sc
                refs.append(_combined_scene_outfit_ref(Sc, OUTFIT_EMPHASIS_BOOST))
                refs.append(_ref(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
            else:
                target_content_mode = "image"
                target_content_source = Sc
                target_content_role = "scene"
                target_geometry_mode, target_geometry_source = "favor_image", Sc
                refs.append(_ref(S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
                refs.append(_ref(Ou, OUTFIT_EMPHASIS_BOOST, "outfit"))

    elif preset == "max_identity":
        if not has_s:
            preset_warnings.append(
                "preset 'max_identity' selected but Subject source is missing; "
                "falling back to balanced appearance routing."
            )

        if not has_s and not has_sc and not has_o:
            pass
        elif has_s and not has_sc and not has_o:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif not has_s and has_sc and not has_o:
            # Balanced fallback
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
        elif not has_s and not has_sc and has_o:
            # Balanced fallback
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Ou
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and not has_o:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
            refs.append(_ref(S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and not has_sc and has_o:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif not has_s and has_sc and has_o:
            # Balanced fallback
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, NORMAL_BOOST))
            else:
                refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        else:
            # S + Sc + Ou
            target_content_mode = "image"
            if outfit_is_scene_physically:
                target_content_source = S
                target_content_role = "subject"
                target_geometry_mode, target_geometry_source = "favor_image", S
                refs.append(_combined_scene_outfit_ref(Sc, NORMAL_BOOST))
                refs.append(_ref(S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            elif outfit_is_distinct:
                target_content_source = Sc
                target_content_role = "scene"
                target_geometry_mode, target_geometry_source = "favor_image", Sc
                refs.append(_ref(S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))

    elif preset == "preserve_scene":
        if not has_sc:
            preset_warnings.append("preset 'preserve_scene' selected but Scene source is missing.")

        if not has_s and not has_sc and not has_o:
            pass
        elif has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
        elif not has_s and has_sc and not has_o:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, PRESERVE_SCENE_BOOST, "scene"))
        elif not has_s and not has_sc and has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", Ou
        elif has_s and has_sc and not has_o:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(Sc, PRESERVE_SCENE_BOOST, "scene"))
            refs.append(_ref(S, NORMAL_BOOST, "subject"))
        elif has_s and not has_sc and has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, NORMAL_BOOST, "subject"))
            refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        elif not has_s and has_sc and has_o:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, PRESERVE_SCENE_BOOST))
            else:
                refs.append(_ref(Sc, PRESERVE_SCENE_BOOST, "scene"))
                # Outfit becomes semantic-only when scene fills 2 appearance slots... but we only have scene here
                # With no Subject, add Outfit as second appearance ref (we have room)
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))
        else:
            # S + Sc + Ou
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, PRESERVE_SCENE_BOOST))
            else:
                refs.append(_ref(Sc, PRESERVE_SCENE_BOOST, "scene"))

            refs.append(_ref(S, NORMAL_BOOST, "subject"))

            if outfit_is_distinct:
                semantic_only_refs.append((Ou, "outfit"))

    elif preset == "outfit_transfer":
        if not has_o:
            preset_warnings.append("preset 'outfit_transfer' selected but Outfit source is missing.")

        if not has_s and not has_sc and not has_o:
            pass
        elif has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = ""
            target_geometry_mode, target_geometry_source = "favor_image", S
        elif not has_s and has_sc and not has_o:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = ""
            target_geometry_mode, target_geometry_source = "favor_image", Sc
        elif not has_s and not has_sc and has_o:
            target_content_mode = "image"
            target_content_source = Ou
            target_content_role = "outfit"
            target_geometry_mode, target_geometry_source = "favor_image", Ou
            refs.append(_ref(Ou, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif has_s and has_sc and not has_o:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = ""
            target_geometry_mode, target_geometry_source = "favor_image", Sc
        elif has_s and not has_sc and has_o:
            target_content_mode = "image"
            target_content_source = Ou
            target_content_role = "outfit"
            target_geometry_mode, target_geometry_source = "favor_image", Ou
            refs.append(_ref(S, NORMAL_BOOST, "subject"))
            refs.append(_ref(Ou, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif not has_s and has_sc and has_o:
            # Sc + Ou without Subject
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, OUTFIT_TRANSFER_BOOST))
            else:
                refs.append(_ref(Sc, NORMAL_BOOST, "scene"))
                refs.append(_ref(Ou, OUTFIT_TRANSFER_BOOST, "outfit"))
        else:
            # S + Sc + Ou
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc
            refs.append(_ref(S, NORMAL_BOOST, "subject"))
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, OUTFIT_TRANSFER_BOOST))
            elif outfit_is_distinct:
                refs.append(_ref(Ou, OUTFIT_TRANSFER_BOOST, "outfit"))

    # Default geometry fallback if geometry source was not explicitly assigned
    if target_geometry_source is None:
        target_geometry_mode, target_geometry_source = _resolve_default_geometry_source(Sc, S, Ou)

    # Style configuration: active for ALL presets whenever effective_style is present
    style_active = has_st
    style_config = STRONG_EASY_STYLE_CONFIG if preset == "style_transfer" else DEFAULT_EASY_STYLE_CONFIG

    return EasyPresetRoute(
        preset=preset,
        target_content_mode=target_content_mode,
        target_content_source=target_content_source,
        target_geometry_mode=target_geometry_mode,
        target_geometry_source=target_geometry_source,
        target_content_role=target_content_role,
        edit_references=tuple(refs),
        semantic_only_references=tuple(semantic_only_refs),
        style_active=style_active,
        style_source=St,
        style_config=style_config,
        warnings=tuple(preset_warnings),
    )
