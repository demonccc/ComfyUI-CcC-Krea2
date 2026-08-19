"""Easy Edit 3-Phase Routing Engine for opinionated Krea2 Edit and Ostris Edit workflows."""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Dict


# Centralized boost constants for Easy presets
NORMAL_BOOST = 1.0

FLEXIBLE_SUBJECT_BOOST = 1.0
BALANCED_SUBJECT_BOOST = 2.5
CONSISTENT_SUBJECT_BOOST = 4.0
PRESERVE_IDENTITY_SUBJECT_BOOST = 6.0
MAX_IDENTITY_SUBJECT_BOOST = 10.0

IDENTITY_TRANSFER_SCENE_BOOST = 2.5
IDENTITY_TRANSFER_SUBJECT_BOOST = 7.0

SUBJECT_TRANSFER_SCENE_BOOST = 2.5
SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST = 7.0
SUBJECT_TRANSFER_SUBJECT_BOOST = 8.0
SUBJECT_TRANSFER_OUTFIT_BOOST = 4.0

PRESERVE_SCENE_BOOST = 2.5
OUTFIT_EMPHASIS_BOOST = 2.5
OUTFIT_TRANSFER_SUBJECT_BOOST = 8.0
OUTFIT_TRANSFER_BOOST = 6.0


# Centralized explicit instruction constants for Easy Edit
EASY_SUBJECT_INSTRUCTION = (
    "Use this reference for the subject identity, facial features, hair, anatomy, body shape, and body proportions."
)
EASY_SCENE_INSTRUCTION = (
    "Use this reference for scene composition, environment, spatial relationships, camera framing, and lighting."
)
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


# Centralized default positive prompts for Easy Edit
EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER = (
    "Transfer only the outfit and accessories from the outfit reference to the subject. "
    "Preserve the subject identity, body, pose, framing, and composition. "
    "Do not preserve the subject clothing. "
    "Fit the transferred outfit and accessories naturally to the subject. "
    "Keep accessories physically attached to the subject in a natural way and never floating. "
    "Do not duplicate accessories."
)

EASY_DEFAULT_PROMPT_SUBJECT_SCENE = (
    "Place the subject from the subject reference naturally into the scene reference. "
    "Preserve the subject identity, body shape, and body proportions. "
    "Preserve the scene composition, environment, framing, perspective, and spatial layout. "
    "Adapt the subject naturally to the scene lighting and environment."
)

EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT = (
    "Place the subject from the subject reference naturally into the scene reference wearing the outfit and accessories from the outfit reference. "
    "Preserve the subject identity, body shape, and body proportions. "
    "Preserve the scene composition, environment, framing, perspective, and spatial layout. "
    "Do not preserve the subject clothing. "
    "Fit the transferred outfit and accessories naturally to the subject and the scene. "
    "Keep accessories physically attached to the subject in a natural way and never floating. "
    "Do not duplicate accessories."
)

EASY_DEFAULT_PROMPT_STYLE = (
    "Apply the visual style from the style reference while preserving the subject identity, content, geometry, framing, and composition. "
    "Transfer only the visual style, including its color palette, texture, lighting character, and overall visual mood. "
    "Do not copy subjects, objects, or scene content from the style reference."
)


OUTFIT_POLICY_USER = "user"
OUTFIT_POLICY_DISABLED = "disabled"

STYLE_POLICY_USER = "user"
STYLE_POLICY_SCENE_AUTO = "scene_auto"
STYLE_POLICY_DISABLED = "disabled"

SCENE_REINTERPRETATION_SUBJECT_BOOST = 4.0

EASY_SCENE_REINTERPRETATION_SCENE_INSTRUCTION = (
    "Use this reference for the main subject's action, activity, pose, body dynamics, environment, spatial context, "
    "camera framing, perspective, lighting, and broad outfit concept. Reinterpret these elements creatively for the "
    "subject reference. Do not use the scene subject's identity as the generated subject identity."
)

EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_BASE = (
    "Replace only the target subject in the scene reference with the subject from the subject reference.\n\n"
    "Preserve the identity of the subject from the subject reference.\n\n"
    "Keep every other person and the rest of the scene unchanged."
)

EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER = (
    "Replace only the identity of the target subject in the scene reference with the identity of the subject from the subject reference.\n\n"
    "Preserve the facial identity, facial features, hair, body identity, anatomy, body shape, and body proportions of the subject reference.\n\n"
    "Preserve the target subject's scene role, position, action, pose, clothing, interaction, and surrounding scene.\n\n"
    "Keep every other person and the rest of the scene unchanged."
)

EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION = (
    "Create a new image of the subject from the subject reference performing the main action or activity shown by the main subject in the scene reference.\n\n"
    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the subject reference.\n\n"
    "Use the scene reference as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n"
    "Adapt the subject naturally to the referenced action and environment.\n\n"
    "Creatively reinterpret the clothing and accessories worn by the main subject in the scene so they are appropriate for the subject and the newly generated image. Do not copy the original scene outfit literally.\n\n"
    "Generate a coherent new image rather than recreating the source scene exactly."
)


def get_easy_instruction_for_role(role: str) -> str:
    return EASY_ROLE_INSTRUCTIONS.get(str(role).lower(), "")


def resolve_easy_visual_reference_fit(preset: str = "", role: str = "", common_geometry_active: bool = True) -> str:
    """Resolve visual reference fit mode.

    INVARIANT: EVERY visual appearance reference in Easy Edit MUST fit completely inside
    the target latent geometry with 'contain_no_upscale'. No Easy appearance reference
    may ever crop away source image content.
    """
    return "contain_no_upscale"


@dataclass(frozen=True)
class EasyPresetCapabilities:
    """Capability contract for Easy Edit presets."""

    outfit_policy: str  # "user" | "disabled"
    style_policy: str  # "user" | "scene_auto" | "disabled"

    @property
    def uses_outfit(self) -> bool:
        return self.outfit_policy != OUTFIT_POLICY_DISABLED

    @property
    def uses_style(self) -> bool:
        return self.style_policy != STYLE_POLICY_DISABLED


EASY_PRESET_CAPABILITIES: Dict[str, EasyPresetCapabilities] = {
    "flexible": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "balanced": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "consistent": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "preserve_identity": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "max_identity": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "identity_transfer": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "subject_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_SCENE_AUTO),
    "preserve_scene": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_USER),
    "outfit_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "style_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_USER),
    "scene_reinterpretation": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
}


def get_easy_preset_capabilities(preset: str) -> EasyPresetCapabilities:
    """Return capability contract for given preset."""
    return EASY_PRESET_CAPABILITIES.get(
        preset, EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER)
    )


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
    outfit_source_kind: str  # "outfit" | "scene" | "style"
    style_source: str
    style_source_kind: str  # "style" | "scene" | "subject"
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
    style_fidelity=1.0, style_processing="2x2", indirect_style_transfer=False, vision_instruction=""
)

INDIRECT_EASY_STYLE_CONFIG = EasyStyleConfig(
    style_fidelity=1.0, style_processing="2x2", indirect_style_transfer=True, vision_instruction=""
)

STRONG_EASY_STYLE_CONFIG = EasyStyleConfig(
    style_fidelity=1.0,
    style_processing="4x4",
    indirect_style_transfer=False,
    vision_instruction="Adopt the artistic style, color palette, texture, and visual mood of this style reference.",
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
    target_content_fit: str
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
    preset: Optional[str] = None,
) -> EasyResolvedSources:
    """Phase 1: Resolve effective sources according to selectors with strict NO-FALLBACK policy and preset gating."""
    warnings: List[str] = []

    caps = (
        get_easy_preset_capabilities(preset)
        if preset
        else EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER)
    )

    effective_subject = subject
    effective_scene = scene

    # Resolve outfit_source
    outfit_source_kind = "outfit"
    if caps.outfit_policy == OUTFIT_POLICY_DISABLED:
        effective_outfit = None
        outfit_source_kind = "disabled"
    else:
        if outfit_source == "none":
            effective_outfit = None
            outfit_source_kind = "none"
        elif outfit_source == "outfit image":
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
    if caps.style_policy == STYLE_POLICY_DISABLED:
        effective_style = None
        style_source_kind = "disabled"
    elif caps.style_policy == STYLE_POLICY_SCENE_AUTO:
        # Automatic Scene style policy (Subject Transfer & Scene Reinterpretation)
        style_source_kind = "scene"
        if scene is not None:
            effective_style = scene
        else:
            effective_style = None
    else:
        # User style policy
        if style_source == "none":
            effective_style = None
            style_source_kind = "none"
        elif style_source == "style image":
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
    scene: Optional[Any], subject: Optional[Any], outfit: Optional[Any]
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
    """Phase 2: Evaluate preset routing matrix across all 10 presets.

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
    target_content_fit = "crop"
    target_geometry_mode = "fixed"
    target_geometry_source: Optional[Any] = None
    refs: List[Tuple[Any, float, str, str]] = []
    semantic_only_refs: List[Tuple[Any, str]] = []
    preset_warnings: List[str] = list(sources.warnings)

    if preset in ("flexible", "balanced", "style_transfer"):
        # --- flexible / balanced / style_transfer: exact 8-combination matrix ---
        subj_boost = FLEXIBLE_SUBJECT_BOOST if preset == "flexible" else BALANCED_SUBJECT_BOOST
        if not has_s and not has_sc and not has_o:
            # none: no refs, default geometry
            pass
        elif has_s and not has_sc and not has_o:
            # S only
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, subj_boost, "subject"))
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
            refs.append(_ref(S, subj_boost, "subject"))
        elif has_s and not has_sc and has_o:
            # S + Ou
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, subj_boost, "subject"))
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
            refs.append(_ref(S, subj_boost, "subject"))
            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, NORMAL_BOOST))
            else:
                refs.append(_ref(Ou, NORMAL_BOOST, "outfit"))

    elif preset == "consistent":
        # --- consistent: strong Subject reference influence (4.0) with target_content_mode = "empty" for S-only ---
        if not has_s:
            preset_warnings.append(
                "preset 'consistent' selected but Subject source is missing; "
                "falling back to balanced appearance routing."
            )

        if not has_s and not has_sc and not has_o:
            pass
        elif has_s and not has_sc and not has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, CONSISTENT_SUBJECT_BOOST, "subject"))
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
            refs.append(_ref(S, CONSISTENT_SUBJECT_BOOST, "subject"))
        elif has_s and not has_sc and has_o:
            target_content_mode = "empty"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, CONSISTENT_SUBJECT_BOOST, "subject"))
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
                refs.append(_ref(S, CONSISTENT_SUBJECT_BOOST, "subject"))
            else:
                target_content_mode = "image"
                target_content_source = Sc
                target_content_role = "scene"
                target_geometry_mode, target_geometry_source = "favor_image", Sc
                refs.append(_ref(S, CONSISTENT_SUBJECT_BOOST, "subject"))
                refs.append(_ref(Ou, OUTFIT_EMPHASIS_BOOST, "outfit"))

    elif preset in ("preserve_identity", "max_identity"):
        # --- preserve_identity (6.0) / max_identity (10.0): target_content_mode = "image" for S-only ---
        subj_boost = PRESERVE_IDENTITY_SUBJECT_BOOST if preset == "preserve_identity" else MAX_IDENTITY_SUBJECT_BOOST
        if not has_s:
            preset_warnings.append(
                f"preset '{preset}' selected but Subject source is missing; "
                "falling back to balanced appearance routing."
            )

        if not has_s and not has_sc and not has_o:
            pass
        elif has_s and not has_sc and not has_o:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, subj_boost, "subject"))
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
            refs.append(_ref(S, subj_boost, "subject"))
        elif has_s and not has_sc and has_o:
            target_content_mode = "image"
            target_content_source = S
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, subj_boost, "subject"))
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
                refs.append(_ref(S, subj_boost, "subject"))
            elif outfit_is_distinct:
                target_content_source = Sc
                target_content_role = "scene"
                target_geometry_mode, target_geometry_source = "favor_image", Sc
                refs.append(_ref(S, subj_boost, "subject"))
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
            target_content_source = S
            target_content_role = "subject"
            target_geometry_mode, target_geometry_source = "favor_image", S
            refs.append(_ref(S, OUTFIT_TRANSFER_SUBJECT_BOOST, "subject"))
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

    elif preset == "identity_transfer":
        if not has_s:
            preset_warnings.append(
                "preset 'identity_transfer' selected but Subject source is missing; no identity can be transferred."
            )
        if not has_sc:
            preset_warnings.append(
                "preset 'identity_transfer' selected but Scene source is missing; target Scene is missing."
            )

        if has_sc and has_s:
            target_content_mode = "image"
            target_content_source = S
            target_content_role = "subject"
            target_content_fit = "contain_no_upscale"
            target_geometry_mode, target_geometry_source = "favor_image", Sc

            refs.append(_ref(Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"))
            refs.append(_ref(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"))
        else:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = "none"
            target_geometry_mode, target_geometry_source = "favor_image", (Sc if has_sc else (S if has_s else None))
            if has_s:
                refs.append(_ref(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"))
            elif has_sc:
                refs.append(_ref(Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"))

    elif preset == "subject_transfer":
        if not has_s:
            preset_warnings.append(
                "preset 'subject_transfer' selected but Subject source is missing; no subject can be transferred."
            )

        if has_sc:
            target_content_mode = "image"
            target_content_source = Sc
            target_content_role = "scene+outfit" if outfit_is_scene_physically else "scene"
            target_geometry_mode, target_geometry_source = "favor_image", Sc

            if outfit_is_scene_physically:
                refs.append(_combined_scene_outfit_ref(Sc, SUBJECT_TRANSFER_OUTFIT_BOOST))
                if has_s:
                    refs.append(_ref(S, SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST, "subject"))
            else:
                refs.append(_ref(Sc, SUBJECT_TRANSFER_SCENE_BOOST, "scene"))
                if has_s:
                    refs.append(_ref(S, SUBJECT_TRANSFER_WITH_SCENE_SUBJECT_BOOST, "subject"))
                if has_o:
                    refs.append(_ref(Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit"))
        else:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = "none"
            if has_s:
                target_geometry_mode, target_geometry_source = "favor_image", S
            elif has_o:
                target_geometry_mode, target_geometry_source = "favor_image", Ou
            else:
                target_geometry_mode, target_geometry_source = "favor_image", None

            if has_s:
                refs.append(_ref(S, SUBJECT_TRANSFER_SUBJECT_BOOST, "subject"))

            if has_o:
                refs.append(_ref(Ou, SUBJECT_TRANSFER_OUTFIT_BOOST, "outfit"))

    elif preset == "scene_reinterpretation":
        target_content_mode = "empty"
        target_content_source = None
        target_content_role = "none"

        if has_sc:
            target_geometry_mode, target_geometry_source = "favor_image", Sc
        elif has_s:
            target_geometry_mode, target_geometry_source = "favor_image", S
        else:
            target_geometry_mode, target_geometry_source = "fixed", None

        if has_sc:
            refs.append((Sc, NORMAL_BOOST, "scene", EASY_SCENE_REINTERPRETATION_SCENE_INSTRUCTION))
        if has_s:
            refs.append(_ref(S, SCENE_REINTERPRETATION_SUBJECT_BOOST, "subject"))

        if not has_s:
            preset_warnings.append("Scene Reinterpretation selected but Subject source is missing.")
        if not has_sc:
            preset_warnings.append("Scene Reinterpretation selected but Scene source is missing.")

    # Default geometry fallback if geometry source was not explicitly assigned
    if target_geometry_source is None:
        target_geometry_mode, target_geometry_source = _resolve_default_geometry_source(Sc, S, Ou)

    # Style configuration: active for ALL presets whenever effective_style is present
    style_active = has_st
    if preset == "style_transfer":
        style_config = STRONG_EASY_STYLE_CONFIG
    else:
        style_config = DEFAULT_EASY_STYLE_CONFIG

    return EasyPresetRoute(
        preset=preset,
        target_content_mode=target_content_mode,
        target_content_source=target_content_source,
        target_geometry_mode=target_geometry_mode,
        target_geometry_source=target_geometry_source,
        target_content_role=target_content_role,
        target_content_fit=target_content_fit,
        edit_references=tuple(refs),
        semantic_only_references=tuple(semantic_only_refs),
        style_active=style_active,
        style_source=St,
        style_config=style_config,
        warnings=tuple(preset_warnings),
    )


def resolve_default_positive_prompt(
    preset: str,
    has_s: bool,
    has_sc: bool,
    has_o: bool,
    has_st: bool,
    outfit_source: str = "outfit image",
    style_source: str = "style image",
) -> Tuple[bool, str, str]:
    """Resolve default positive prompt text and internal key for Easy Edit based on preset and connected inputs.

    Returns:
        (has_default: bool, prompt_text: str, prompt_key: str)
    """
    # 1. Subject-only: if only Subject is connected (no Scene, Outfit, or Style), NO default prompt exists.
    if has_s and not has_sc and not has_o and not has_st:
        return False, "", "none"

    # 2. No inputs at all: NO default prompt exists.
    if not has_s and not has_sc and not has_o and not has_st:
        return False, "", "none"

    base_prompt = ""
    base_key = ""

    if preset == "identity_transfer":
        if not (has_s and has_sc):
            return False, "", "none"
        base_prompt = EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER
        base_key = "identity_transfer"
        return True, base_prompt, base_key

    elif preset == "subject_transfer":
        if not (has_s and has_sc):
            return False, "", "none"

        base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_BASE
        if not has_o:
            outfit_clause = "Keep the clothing and accessories of the subject reference."
            base_key = "subject_transfer"
        elif outfit_source == "outfit image":
            outfit_clause = "Use the clothing and accessories from the outfit reference."
            base_key = "subject_transfer_outfit"
        elif outfit_source == "scene image":
            outfit_clause = "Use the clothing and accessories of the target subject from the scene reference."
            base_key = "subject_transfer_scene_outfit"
        elif outfit_source == "style image":
            outfit_clause = "Use the clothing and accessories from the outfit reference."
            base_key = "subject_transfer_style_outfit"
        else:
            outfit_clause = "Keep the clothing and accessories of the subject reference."
            base_key = "subject_transfer"

        base_prompt = f"{base_prompt}\n\n{outfit_clause}"
        return True, base_prompt, base_key

    elif preset == "scene_reinterpretation":
        if not (has_s and has_sc):
            return False, "", "none"
        base_prompt = EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION
        base_key = "scene_reinterpretation"
        return True, base_prompt, base_key

    elif preset == "outfit_transfer":
        if has_o:
            base_prompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
            base_key = "outfit_transfer"
        elif has_sc:
            base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
            base_key = "subject_scene"
    elif preset == "preserve_scene":
        if has_sc:
            if has_o:
                base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT
                base_key = "subject_scene_outfit"
            else:
                base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
                base_key = "subject_scene"
        elif has_o:
            base_prompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
            base_key = "outfit_transfer"
    elif preset == "style_transfer":
        if has_st:
            base_prompt = EASY_DEFAULT_PROMPT_STYLE
            base_key = "style"
        else:
            if has_sc and has_o:
                base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT
                base_key = "subject_scene_outfit"
            elif has_sc:
                base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
                base_key = "subject_scene"
            elif has_o:
                base_prompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
                base_key = "outfit_transfer"
    else:
        # Identity presets: flexible, balanced, consistent, preserve_identity, max_identity
        if has_sc and has_o:
            base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT
            base_key = "subject_scene_outfit"
        elif has_sc:
            base_prompt = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
            base_key = "subject_scene"
        elif has_o:
            base_prompt = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
            base_key = "outfit_transfer"

    # Style clause handling
    if has_st and preset != "style_transfer":
        if base_prompt:
            return True, f"{base_prompt}\n\n{EASY_DEFAULT_PROMPT_STYLE}", f"{base_key}_style"
        else:
            return True, EASY_DEFAULT_PROMPT_STYLE, "style"

    if base_prompt:
        return True, base_prompt, base_key

    return False, "", "none"
