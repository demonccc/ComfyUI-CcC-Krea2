"""Easy Edit 3-Phase Routing Engine for opinionated Krea2 Edit and Ostris Edit workflows."""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Union


# Boost constants for Easy presets
BALANCED_SUBJECT_BOOST = 2.5
MAX_IDENTITY_SUBJECT_BOOST = 4.0
PRESERVE_SCENE_BOOST = 4.0
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
    """Phase 2: Preset routing decision containing target and reference choices."""
    preset: str
    target_source: Optional[Any]
    target_content_mode: str  # "empty" or "image"
    geometry_mode: str  # "fixed" or "favor_image"
    edit_references: Tuple[Tuple[Any, float, str], ...]  # Tuple of (image, boost, alias_role)


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


def route_easy_preset(
    sources: EasyResolvedSources,
    preset: str = "balanced",
) -> EasyPresetRoute:
    """Phase 2: Evaluate preset routing matrix across all 6 presets."""
    S = sources.effective_subject
    Sc = sources.effective_scene
    O = sources.effective_outfit

    has_s = S is not None
    has_sc = Sc is not None
    has_o = O is not None

    is_o_same_sc = has_o and has_sc and (O is Sc)
    is_o_distinct_sc = has_o and (not has_sc or O is not Sc)

    target_source = None
    target_content_mode = "empty"
    geometry_mode = "fixed"
    refs: List[Tuple[Any, float, str]] = []

    if preset in ("balanced", "style_transfer"):
        if has_s and not has_sc and not has_o:
            target_source = None
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
        elif has_s and has_sc and not is_o_distinct_sc:
            target_source = None
            refs.append((Sc, NORMAL_BOOST, "scene"))
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
        elif has_s and has_o and not has_sc:
            target_source = None
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_source = Sc
            geometry_mode = "favor_image"
            target_content_mode = "image"
            refs.append((S, BALANCED_SUBJECT_BOOST, "subject"))
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))
        elif has_sc and not has_s and not is_o_distinct_sc:
            target_source = None
            refs.append((Sc, NORMAL_BOOST, "scene"))
        elif has_o and not has_s and not has_sc:
            target_source = None
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))

    elif preset == "preserve_identity":
        if has_s and not has_sc and not has_o:
            target_source = None
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_sc and not is_o_distinct_sc:
            if is_o_same_sc:
                # Special case: outfit is scene -> scene boost 2.5, subject 4.0
                target_source = None
                refs.append((Sc, OUTFIT_EMPHASIS_BOOST, "scene"))
                refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            else:
                target_source = None
                refs.append((Sc, NORMAL_BOOST, "scene"))
                refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_o and not has_sc:
            target_source = None
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_source = Sc
            geometry_mode = "favor_image"
            target_content_mode = "image"
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))
        elif has_sc and not has_s and not is_o_distinct_sc:
            target_source = None
            refs.append((Sc, NORMAL_BOOST, "scene"))
        elif has_o and not has_s and not has_sc:
            target_source = None
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))

    elif preset == "max_identity":
        if has_s and not has_sc and not has_o:
            target_source = S
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_sc and not is_o_distinct_sc:
            target_source = S
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((Sc, NORMAL_BOOST, "scene"))
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
        elif has_s and has_o and not has_sc:
            target_source = S
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_source = Sc
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((S, MAX_IDENTITY_SUBJECT_BOOST, "subject"))
            refs.append((O, NORMAL_BOOST, "outfit"))
        elif has_sc and not has_s and not is_o_distinct_sc:
            target_source = None
            refs.append((Sc, NORMAL_BOOST, "scene"))
        elif has_o and not has_s and not has_sc:
            target_source = None
            refs.append((O, NORMAL_BOOST, "outfit"))

    elif preset == "preserve_scene":
        if has_sc:
            target_source = Sc
            target_content_mode = "image"
            geometry_mode = "favor_image"
            if has_s:
                refs.append((Sc, PRESERVE_SCENE_BOOST, "scene"))
                refs.append((S, NORMAL_BOOST, "subject"))
            else:
                refs.append((Sc, PRESERVE_SCENE_BOOST, "scene"))
        elif has_s and has_o and not has_sc:
            target_source = None
            refs.append((S, NORMAL_BOOST, "subject"))
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))
        elif has_o and not has_s and not has_sc:
            target_source = None
            refs.append((O, OUTFIT_EMPHASIS_BOOST, "outfit"))

    elif preset == "outfit_transfer":
        if has_o and not has_s and not has_sc:
            target_source = O
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((O, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif has_s and has_o and not has_sc:
            target_source = O
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((S, NORMAL_BOOST, "subject"))
            refs.append((O, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif has_s and has_sc and is_o_distinct_sc:
            target_source = Sc
            target_content_mode = "image"
            geometry_mode = "favor_image"
            refs.append((S, NORMAL_BOOST, "subject"))
            refs.append((O, OUTFIT_TRANSFER_BOOST, "outfit"))
        elif is_o_same_sc:
            target_source = Sc
            target_content_mode = "image"
            geometry_mode = "favor_image"
            if has_s:
                refs.append((Sc, OUTFIT_TRANSFER_BOOST, "scene"))
                refs.append((S, NORMAL_BOOST, "subject"))
            else:
                refs.append((Sc, OUTFIT_TRANSFER_BOOST, "scene"))

    return EasyPresetRoute(
        preset=preset,
        target_source=target_source,
        target_content_mode=target_content_mode,
        geometry_mode=geometry_mode,
        edit_references=tuple(refs),
    )
