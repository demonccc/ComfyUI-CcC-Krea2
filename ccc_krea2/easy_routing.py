"""Easy Edit 3-Phase Routing Engine for opinionated Krea2 Edit and Ostris Edit workflows."""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union


# Centralized boost constants for Easy presets
BALANCED_SUBJECT_BOOST = 2.5
MAX_IDENTITY_SUBJECT_BOOST = 4.0
PRESERVE_IDENTITY_SUBJECT_BOOST = 2.5
PRESERVE_SCENE_BOOST = 2.5
OUTFIT_EMPHASIS_BOOST = 2.5
OUTFIT_TRANSFER_BOOST = 4.0
NORMAL_BOOST = 1.0


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
    style_source: str
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class EasyPresetRoute:
    """Phase 2 & 3: Preset routing decision containing target content, target geometry, references, and style config."""
    preset: str
    target_content_mode: str  # "empty" or "image"
    target_content_source: Optional[Any]
    target_geometry_mode: str  # "fixed" or "favor_image"
    target_geometry_source: Optional[Any]
    edit_references: Tuple[Tuple[Any, float, str], ...]  # Tuple of (image, boost, alias_role)
    style_active: bool
    style_source: Optional[Any]
    style_strength: float  # 1.0 for normal, 2.0 (high) for style_transfer
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
    if outfit_source == "outfit image":
        effective_outfit = outfit
    elif outfit_source == "scene image":
        if scene is not None:
            effective_outfit = scene
        else:
            effective_outfit = None
            warnings.append("outfit_source set to 'scene image' but scene image is disconnected.")
    elif outfit_source == "style image":
        if style is not None:
            effective_outfit = style
        else:
            effective_outfit = None
            warnings.append("outfit_source set to 'style image' but style image is disconnected.")
    else:
        effective_outfit = outfit

    # Resolve style_source
    if style_source == "style image":
        effective_style = style
    elif style_source == "scene image":
        if scene is not None:
            effective_style = scene
        else:
            effective_style = None
            warnings.append("style_source set to 'scene image' but scene image is disconnected.")
    elif style_source == "subject image":
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
        style_source=style_source,
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


def route_easy_preset(
    sources: EasyResolvedSources,
    preset: str = "balanced",
) -> EasyPresetRoute:
    """Phase 2: Evaluate preset routing matrix across all 6 presets."""
    S = sources.effective_subject
    Sc = sources.effective_scene
    O = sources.effective_outfit
    St = sources.effective_style

    has_s = S is not None
    has_sc = Sc is not None
    has_o = O is not None
    has_st = St is not None

    is_o_same_sc = has_o and has_sc and (O is Sc)
    is_o_distinct_sc = has_o and (not has_sc or O is not Sc)

    target_content_mode = "empty"
    target_content_source: Optional[Any] = None
    target_geometry_mode = "fixed"
    target_geometry_source: Optional[Any] = None
    refs: List[Tuple[Any, float, str]] = []
    preset_warnings: List[str] = list(sources.warnings)

    if preset in ("balanced", "style_transfer"):
        if has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
        elif has_s and has_sc and not is_o_distinct_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((Sc, NORMAL_BOOST, "scene"))
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
        elif has_s and has_o and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_sc and not has_s and not is_o_distinct_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((Sc, NORMAL_BOOST, "scene"))
        elif has_o and not has_s and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", O)
            refs.append((O, NORMAL_BOOST, "outfit"))

    elif preset == "preserve_identity":
        if not has_s:
            preset_warnings.append("preset 'preserve_identity' selected but Subject source is missing.")

        if has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_sc and not is_o_distinct_sc:
            if is_o_same_sc:
                target_content_mode = "empty"
                target_geometry_mode, target_geometry_source = ("favor_image", Sc)
                refs.append((Sc, OUTFIT_EMPHASIS_BOOST, "scene"))
                refs.append((S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
            else:
                target_content_mode = "empty"
                target_geometry_mode, target_geometry_source = ("favor_image", Sc)
                refs.append((Sc, NORMAL_BOOST, "scene"))
                refs.append((S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_o and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((S, PRESERVE_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))
        elif has_sc and not has_s and not is_o_distinct_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((Sc, NORMAL_BOOST, "scene"))
        elif has_o and not has_s and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", O)
            refs.append((O, NORMAL_BOOST, "outfit"))

    elif preset == "max_identity":
        if not has_s:
            preset_warnings.append("preset 'max_identity' selected but Subject source is missing.")

        if has_s and not has_sc and not has_o:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_sc and not is_o_distinct_sc:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((Sc, NORMAL_BOOST, "scene"))
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_o and not has_sc:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_sc and not has_s and not is_o_distinct_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((Sc, NORMAL_BOOST, "scene"))
        elif has_o and not has_s and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", O)
            refs.append((O, NORMAL_BOOST, "outfit"))

    elif preset == "preserve_scene":
        if not has_sc:
            preset_warnings.append("preset 'preserve_scene' selected but Scene source is missing.")

        if has_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            if has_s:
                refs.append((Sc, PRESERVE_SCENE_BOOST, "scene"))
                refs.append((S, NORMAL_BOOST, "subject"))
            else:
                refs.append((Sc, PRESERVE_SCENE_BOOST, "scene"))
        elif has_s and has_o and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", S)
            refs.append((S, NORMAL_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", S)
        elif has_o and not has_s and not has_sc:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = ("favor_image", O)

    elif preset == "outfit_transfer":
        if not has_o:
            preset_warnings.append("preset 'outfit_transfer' selected but Outfit source is missing.")

        if has_o and not has_s and not has_sc:
            target_content_mode = "image"
            target_content_source = O
            target_geometry_mode, target_geometry_source = ("favor_image", O)
            refs.append((O, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif has_s and has_o and not has_sc:
            target_content_mode = "image"
            target_content_source = O
            target_geometry_mode, target_geometry_source = ("favor_image", O)
            refs.append((S, NORMAL_BOOST, "subject"))
            refs.append((O, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            refs.append((S, NORMAL_BOOST, "subject"))
            refs.append((O, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif is_o_same_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_geometry_mode, target_geometry_source = ("favor_image", Sc)
            if has_s:
                refs.append((Sc, OUTFIT_TRANSFER_BOOST, "scene"))
                refs.append((S, NORMAL_BOOST, "subject"))
            else:
                refs.append((Sc, OUTFIT_TRANSFER_BOOST, "scene"))

    # Default geometry fallback if geometry source was not explicitly assigned
    if target_geometry_source is None:
        target_geometry_mode, target_geometry_source = _resolve_default_geometry_source(Sc, S, O)

    # Style configuration: active for ALL presets whenever effective_style is present
    style_active = has_st
    style_strength = 2.0 if preset == "style_transfer" else 1.0

    return EasyPresetRoute(
        preset=preset,
        target_content_mode=target_content_mode,
        target_content_source=target_content_source,
        target_geometry_mode=target_geometry_mode,
        target_geometry_source=target_geometry_source,
        edit_references=tuple(refs),
        style_active=style_active,
        style_source=St,
        style_strength=style_strength,
        warnings=tuple(preset_warnings),
    )
