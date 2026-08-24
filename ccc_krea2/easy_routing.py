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

IDENTITY_TRANSFER_SCENE_BOOST = 2.0
IDENTITY_TRANSFER_SUBJECT_BOOST = 2.0
SUBJECT_TRANSFER_SCENE_BOOST = 1.0
FLEXIBLE_SUBJECT_TRANSFER_SCENE_BOOST = 2.5
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
EASY_SUBJECT_TRANSFER_SCENE_STYLE_INSTRUCTION = (
    "Use the scene image to reinforce the scene composition, framing, environment, objects, lighting, "
    "color palette, and overall visual treatment.\n\n"
    "Preserve the position, pose, action, role, and interactions of the person being replaced.\n\n"
    "Do not transfer the identity, facial features, hair, anatomy, body shape, body proportions, clothing, "
    "or accessories of that person."
)
EASY_DEFAULT_STYLE_INSTRUCTION = (
    "Use this image only as a visual style reference.\n\n"
    "Apply its color palette, lighting character, contrast, texture, rendering treatment, photographic treatment, "
    "and overall visual mood.\n\n"
    "Do not transfer subjects, identities, facial features, hair, anatomy, body shapes, clothing, accessories, "
    "poses, objects, environment, layout, framing, or scene composition from this image."
)
EASY_SEMANTIC_OUTFIT_INSTRUCTION = (
    "Use this image only as the outfit reference.\n\n"
    "Transfer the clothing, garments, footwear, and accessories shown in this image to the transferred subject.\n\n"
    "Do not transfer the wearer's identity, facial features, hair, anatomy, body shape, body proportions, pose, "
    "background, environment, objects, composition, lighting, or photographic style."
)
EASY_SCENE_REINTERPRETATION_STYLE_INSTRUCTION = (
    "Use the scene image as a direct visual reference for recreating the scene composition, framing, environment, "
    "objects, lighting, color treatment, every other person, and all spatial relationships.\n\n"
    "Preserve the position, pose, action, role, and interactions of the {reference_subject}.\n\n"
    "Do not transfer the identity, facial features, hair, anatomy, body shape, body proportions, clothing, or "
    "accessories of the {reference_subject}."
)

EASY_ROLE_INSTRUCTIONS = {
    "subject": EASY_SUBJECT_INSTRUCTION,
    "scene": EASY_SCENE_INSTRUCTION,
    "outfit": EASY_OUTFIT_INSTRUCTION,
    "scene+outfit": EASY_SCENE_AND_OUTFIT_INSTRUCTION,
    "target_scene": EASY_TARGET_SCENE_INSTRUCTION,
    "target": EASY_TARGET_SCENE_INSTRUCTION,
}


# Centralized default positive prompts for Easy Edit (Placeholder-driven templates)

EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER = (
    "Replace only the identity of the {reference_subject} of the scene image with the identity of the {subject_description} from the subject image.\n\n"
    "Transfer the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the {subject_description} from the subject image to the {reference_subject} of the scene image.\n\n"
    "Preserve the position, action, pose, role, interaction, clothing, and accessories of the {reference_subject} from the scene image."
)

EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_NO_OUTFIT = (
    "Replace only the {reference_subject} of the {scene_source} with the {subject} of the {subject_source}.\n\n"
    "Transfer the complete {subject} from the {subject_source}, including the exact facial identity, facial features, hair, anatomy, body shape, body proportions, clothing, and accessories.\n\n"
    "Preserve the face, body shape, body proportions, clothing, and accessories of the {subject} from the {subject_source}.\n\n"
    "Adapt the transferred {subject} naturally to the target scene while preserving the rest of the scene.\n\n"
    "Keep every other person and the rest of the scene unchanged."
)

EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER = (
    "Replace only the {reference_subject} of the {scene_source} with the {subject} from the {subject_source}.\n\n"
    "Transfer the exact facial identity, facial features, hair, anatomy, body shape, body proportions, clothing, and accessories of the {subject} from the {subject_source}.\n\n"
    "Place the transferred {subject} in the same position and pose as the {reference_subject}. Make the transferred {subject} perform the same action, fulfill the same role, and interact with every person and object in the same way as the {reference_subject}.\n\n"
    "Keep every other person and the rest of the scene unchanged."
)

EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT = (
    "Replace only the {reference_subject} of the {scene_source} with the {subject} of the {subject_source}.\n\n"
    "Preserve the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n"
    "Dress the transferred {subject} using {outfit_reference}.\n\n"
    "Do not preserve the clothing or accessories of the {subject} from the {subject_source} when an explicit outfit source is selected. Use {outfit_reference} instead.\n\n"
    "Keep every other person and the rest of the scene unchanged."
)

EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER = (
    "Transfer only {outfit_reference} to the {subject}.\n\n"
    "Preserve the {subject} identity, body, pose, framing, and composition.\n\n"
    "Do not preserve the {subject} clothing.\n\n"
    "Fit the transferred outfit and accessories naturally to the {subject}.\n\n"
    "Keep accessories physically attached to the {subject} in a natural way and never floating.\n\n"
    "Do not duplicate accessories."
)

EASY_DEFAULT_PROMPT_SUBJECT_SCENE = (
    "Place the {subject} from the {subject_source} naturally into the {scene_source}.\n\n"
    "Preserve the {subject} identity, body shape, and body proportions.\n\n"
    "Preserve the scene composition, environment, framing, perspective, and spatial layout.\n\n"
    "Adapt the {subject} naturally to the {scene_source} lighting and environment."
)

EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT = (
    "Place the {subject} from the {subject_source} naturally into the {scene_source} wearing {outfit_reference}.\n\n"
    "Preserve the {subject} identity, body shape, and body proportions.\n\n"
    "Preserve the scene composition, environment, framing, perspective, and spatial layout.\n\n"
    "Do not preserve the {subject} clothing.\n\n"
    "Fit the transferred outfit and accessories naturally to the {subject} and the scene.\n\n"
    "Keep accessories physically attached to the {subject} in a natural way and never floating.\n\n"
    "Do not duplicate accessories."
)

EASY_DEFAULT_PROMPT_STYLE = (
    "Use the {style_source} only as a visual style reference.\n\n"
    "Apply its color palette, lighting character, contrast, texture, rendering treatment, photographic treatment, and overall visual mood.\n\n"
    "Do not transfer subjects, identities, facial features, hair, anatomy, body shapes, clothing, accessories, poses, objects, environment, layout, framing, or scene composition from the {style_source}."
)


OUTFIT_POLICY_USER = "user"
OUTFIT_POLICY_DISABLED = "disabled"

STYLE_POLICY_USER = "user"
STYLE_POLICY_SCENE_AUTO = "scene_auto"
STYLE_POLICY_DISABLED = "disabled"

SCENE_REINTERPRETATION_SUBJECT_BOOST = 7.0
SCENE_REINTERPRETATION_OUTFIT_BOOST = 4.0

EASY_PRESET_DISPLAY_LABELS = {
    "flexible": "Flexible",
    "balanced": "Balanced",
    "consistent": "Consistent",
    "preserve_identity": "Preserve Identity",
    "max_identity": "Max Identity",
    "identity_transfer": "Identity Transfer",
    "subject_transfer_1": "Subject Transfer 1",
    "subject_transfer_2": "Subject Transfer 2",
    "flexible_subject_transfer_1": "Flexible Subject Transfer 1",
    "flexible_subject_transfer_2": "Flexible Subject Transfer 2",
    "preserve_scene": "Preserve Scene",
    "outfit_transfer": "Outfit Transfer",
    "style_transfer": "Style Transfer",
    "scene_reinterpretation": "Scene Reinterpretation",
}

IDENTITY_TEST_PRESETS = (
    "identity_transfer",
)

EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION = (
    "Create a new image of the {subject} from the {subject_source} performing the main action or activity shown by the {reference_subject} in the {scene_source}.\n\n"
    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n"
    "Use the {scene_source} as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n"
    "Adapt the {subject} naturally to the referenced action and environment.\n\n"
    "Creatively reinterpret the clothing and accessories worn by the {reference_subject} in the {scene_source} so they are appropriate for the {subject} and the newly generated image. Do not copy the original scene outfit literally.\n\n"
    "Generate a coherent new image rather than recreating the source scene exactly."
)

EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION_WITH_OUTFIT = (
    "Create a new image of the {subject} from the {subject_source} performing the main action or activity shown by the {reference_subject} in the {scene_source}.\n\n"
    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n"
    "Use the {scene_source} as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n"
    "Dress the {subject} using {outfit_reference}.\n\n"
    "Adapt the {subject} and the selected outfit naturally to the referenced action and environment.\n\n"
    "Generate a coherent new image rather than recreating the source scene exactly."
)


def get_easy_instruction_for_role(role: str) -> str:
    return EASY_ROLE_INSTRUCTIONS.get(str(role).lower(), "")


def resolve_easy_visual_reference_fit(preset: str = "", role: str = "", common_geometry_active: bool = True) -> str:
    """Resolve visual reference fit mode.

    INVARIANT: EVERY visual appearance reference in Easy Edit MUST fit completely inside
    the target latent geometry with 'contain'. No Easy appearance reference
    may crop away face, hair, clothing, or subject context.
    """
    return "contain"


def render_easy_prompt(template: str, context: Dict[str, str]) -> str:
    """Safely format prompt template with supported context placeholders."""
    if not template:
        return ""
    ref_subj = context.get("reference_subject", "main subject")
    subj = context.get("subject", "main subject")
    subj_desc = context.get("subject_description", context.get("subject", "main subject"))
    scene_src = context.get("scene_source", "scene image")
    subj_src = context.get("subject_source", "subject image")
    outfit_src = context.get("outfit_source", "outfit image")
    style_src = context.get("style_source", "style image")
    outfit_reference = context.get("outfit_reference") or resolve_easy_outfit_reference_text(
        outfit_source=outfit_src,
        reference_subject=ref_subj,
        subject_description=subj_desc,
    )

    fmt_context = {
        "reference_subject": ref_subj,
        "subject": subj,
        "subject_description": subj_desc,
        "scene_source": scene_src,
        "subject_source": subj_src,
        "outfit_source": outfit_src,
        "outfit_reference": outfit_reference,
        "style_source": style_src,
    }
    return template.format(**fmt_context)


def resolve_easy_outfit_reference_text(
    outfit_source: str,
    reference_subject: str = "main subject",
    subject_description: str = "main subject",
) -> str:
    """Resolve the source-aware Outfit phrase used by automatic prompts."""
    ref_subj = reference_subject.strip() if reference_subject and reference_subject.strip() else "main subject"
    subj = subject_description.strip() if subject_description and subject_description.strip() else "main subject"
    if outfit_source == "scene image":
        return f"the clothing, footwear, and accessories worn by the {ref_subj} in the scene image"
    if outfit_source == "subject image":
        return f"the clothing, footwear, and accessories worn by the {subj} in the subject image"
    if outfit_source == "style image":
        return "the relevant clothing, footwear, and accessories interpreted from the style image"
    return "the principal outfit identified in the outfit image"


def resolve_easy_outfit_vision_instruction(
    outfit_source_kind: str,
    use_default_prompt: bool,
    reference_subject: str = "main subject",
    subject_description: str = "main subject",
) -> str:
    """Resolve Outfit vision guidance for appearance, semantic-only, and Style-path references."""
    if not use_default_prompt:
        return (
            "Use this image as the outfit reference.\n\n"
            "Identify and transfer the relevant clothing, footwear, and accessories visible in the image.\n\n"
            "Do not transfer identity, facial features, hair, anatomy, body shape, body proportions, pose, "
            "background, environment, objects, composition, lighting, or photographic style."
        )

    ref_subj = reference_subject.strip() if reference_subject and reference_subject.strip() else "main subject"
    subj = subject_description.strip() if subject_description and subject_description.strip() else "main subject"
    if outfit_source_kind == "scene":
        selection = (
            f"Use only the clothing, footwear, and accessories worn by the {ref_subj} in the scene image.\n\n"
            "Do not use clothing or accessories from any other person in the image."
        )
    elif outfit_source_kind == "subject":
        selection = (
            f"Use only the clothing, footwear, and accessories worn by the {subj} in the subject image.\n\n"
            "Do not use clothing or accessories from any other person in the image."
        )
    elif outfit_source_kind == "style":
        selection = "Interpret and use the relevant clothing, footwear, and accessories visible in the style image."
    else:
        selection = "Identify and use the principal outfit, footwear, and accessories visible in the outfit image."

    return (
        f"{selection}\n\n"
        "Do not transfer identity, facial features, hair, anatomy, body shape, body proportions, pose, background, "
        "environment, objects, composition, lighting, or photographic style."
    )


def resolve_default_positive_prompt(
    preset: str,
    has_s: bool,
    has_sc: bool,
    has_o: bool,
    has_st: bool,
    outfit_source: str = "outfit image",
    style_source: str = "style image",
    reference_subject: str = "main subject",
    subject_description: str = "main subject",
) -> Tuple[bool, str, str]:
    """Resolve default positive prompt text and internal key for Easy Edit based on preset and connected inputs.

    Helper LoRA Workflow Alignment:
    - Conrad Identity Edit LoRA (krea2_identity_edit_v1_2.safetensors) is naturally aligned with identity_transfer
      (identity replacement while preserving Scene clothing).
    - BFS Body Swap LoRA (bfs_body_swap_v1_krea2.safetensors) is naturally aligned with Subject Transfer presets
      (full-person replacement including Subject clothing).
    - Both LoRAs can be used with either preset and are not exclusive.

    Returns:
        (has_default: bool, prompt_text: str, prompt_key: str)
    """
    # 1. Subject-only: if only Subject is connected (no Scene, Outfit, or Style), NO default prompt exists.
    if has_s and not has_sc and not has_o and not has_st:
        return False, "", "none"

    # 2. No inputs at all: NO default prompt exists.
    if not has_s and not has_sc and not has_o and not has_st:
        return False, "", "none"

    # Clean & fallback subject descriptions
    ref_subj = reference_subject.strip() if reference_subject and reference_subject.strip() else "main subject"
    subj_desc = subject_description.strip() if subject_description and subject_description.strip() else "main subject"

    # Resolve effective outfit_source placeholder string
    if outfit_source in {"subject image", "outfit image", "scene image", "style image"}:
        eff_outfit_source = outfit_source
    else:
        eff_outfit_source = "outfit image"

    # Resolve effective style_source placeholder string
    if style_source in {"style image", "scene image", "subject image"}:
        eff_style_source = style_source
    else:
        eff_style_source = "style image"

    context = {
        "reference_subject": ref_subj,
        "subject": subj_desc,
        "subject_description": subj_desc,
        "scene_source": "scene image",
        "subject_source": "subject image",
        "outfit_source": eff_outfit_source,
        "style_source": eff_style_source,
    }

    base_template = ""
    base_key = ""

    if preset in IDENTITY_TEST_PRESETS:
        if not (has_s and has_sc):
            return False, "", "none"
        base_template = EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER
        base_key = preset
        return True, render_easy_prompt(base_template, context), base_key

    elif preset in ("subject_transfer_1", "subject_transfer_2"):
        if not (has_s and has_sc):
            return False, "", "none"

        base_template = EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER
        base_key = preset
        return True, render_easy_prompt(base_template, context), base_key

    elif preset in ("flexible_subject_transfer_1", "flexible_subject_transfer_2"):
        if not (has_s and has_sc):
            return False, "", "none"

        if not has_o or eff_outfit_source == "subject image":
            base_template = EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_NO_OUTFIT
            base_key = preset
        else:
            base_template = EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT
            if eff_outfit_source == "outfit image":
                base_key = "subject_transfer_outfit"
            elif eff_outfit_source == "scene image":
                base_key = "subject_transfer_scene_outfit"
            elif eff_outfit_source == "style image":
                base_key = "subject_transfer_style_outfit"
            else:
                base_key = "subject_transfer_outfit"

        rendered_base = render_easy_prompt(base_template, context)
        if has_st:
            rendered_style = render_easy_prompt(EASY_DEFAULT_PROMPT_STYLE, context)
            return True, f"{rendered_base}\n\n{rendered_style}", f"{base_key}_style"
        return True, rendered_base, base_key

    elif preset == "scene_reinterpretation":
        if not (has_s and has_sc):
            return False, "", "none"
        if has_o:
            base_template = EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION_WITH_OUTFIT
            base_key = "scene_reinterpretation_outfit"
        else:
            base_template = EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION
            base_key = "scene_reinterpretation"
        return True, render_easy_prompt(base_template, context), base_key

    elif preset == "outfit_transfer":
        if has_o:
            base_template = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
            base_key = "outfit_transfer"
        elif has_sc:
            base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
            base_key = "subject_scene"
    elif preset == "preserve_scene":
        if has_sc:
            if has_o:
                base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT
                base_key = "subject_scene_outfit"
            else:
                base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
                base_key = "subject_scene"
        elif has_o:
            base_template = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
            base_key = "outfit_transfer"
    elif preset == "style_transfer":
        if has_st:
            base_template = EASY_DEFAULT_PROMPT_STYLE
            base_key = "style"
        else:
            if has_sc and has_o:
                base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT
                base_key = "subject_scene_outfit"
            elif has_sc:
                base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
                base_key = "subject_scene"
            elif has_o:
                base_template = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
                base_key = "outfit_transfer"
    else:
        # Identity presets: flexible, balanced, consistent, preserve_identity, max_identity
        if has_sc and has_o:
            base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT
            base_key = "subject_scene_outfit"
        elif has_sc:
            base_template = EASY_DEFAULT_PROMPT_SUBJECT_SCENE
            base_key = "subject_scene"
        elif has_o:
            base_template = EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
            base_key = "outfit_transfer"

    # Style clause handling
    if has_st and preset != "style_transfer":
        rendered_style = render_easy_prompt(EASY_DEFAULT_PROMPT_STYLE, context)
        if base_template:
            rendered_base = render_easy_prompt(base_template, context)
            return True, f"{rendered_base}\n\n{rendered_style}", f"{base_key}_style"
        else:
            return True, rendered_style, "style"

    if base_template:
        return True, render_easy_prompt(base_template, context), base_key

    return False, "", "none"


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
    "subject_transfer_1": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "subject_transfer_2": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "flexible_subject_transfer_1": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER
    ),
    "flexible_subject_transfer_2": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER
    ),
    "preserve_scene": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_USER),
    "outfit_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "style_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_USER),
    "scene_reinterpretation": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_SCENE_AUTO
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
    outfit_source_kind: str  # "outfit" | "scene" | "style" | "subject"
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
    style_fidelity=1.0,
    style_processing="2x2",
    indirect_style_transfer=True,
    vision_instruction=EASY_DEFAULT_STYLE_INSTRUCTION,
)

SEMANTIC_OUTFIT_STYLE_CONFIG = EasyStyleConfig(
    style_fidelity=1.0,
    style_processing="2x2",
    indirect_style_transfer=False,
    vision_instruction=EASY_SEMANTIC_OUTFIT_INSTRUCTION,
)

SUBJECT_TRANSFER_SCENE_STYLE_CONFIG = EasyStyleConfig(
    style_fidelity=1.0,
    style_processing="2x2",
    indirect_style_transfer=True,
    vision_instruction=EASY_SUBJECT_TRANSFER_SCENE_STYLE_INSTRUCTION,
)


def make_scene_reinterpretation_style_config(reference_subject: str = "main subject") -> EasyStyleConfig:
    ref_subj = reference_subject.strip() if reference_subject and reference_subject.strip() else "main subject"
    return EasyStyleConfig(
        style_fidelity=1.0,
        style_processing="2x2",
        indirect_style_transfer=False,
        vision_instruction=EASY_SCENE_REINTERPRETATION_STYLE_INSTRUCTION.format(reference_subject=ref_subj),
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
    semantic_outfit_active: bool
    semantic_outfit_source: Optional[Any]
    semantic_outfit_config: EasyStyleConfig
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

    if preset in ("flexible_subject_transfer_1", "flexible_subject_transfer_2"):
        if outfit_source not in ("none", "subject image", "scene image"):
            # Flexible Subject Transfer defaults to preserving/reinforcing the
            # Subject outfit and deliberately does not consume the Outfit socket.
            outfit_source = "subject image"

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
        elif outfit_source == "subject image":
            outfit_source_kind = "subject"
            if subject is not None:
                effective_outfit = subject
            else:
                effective_outfit = None
                warnings.append("outfit_source set to 'subject image' but subject image is disconnected.")
        else:
            effective_outfit = outfit

    # Resolve style_source
    style_source_kind = "style"
    if caps.style_policy == STYLE_POLICY_DISABLED:
        effective_style = None
        style_source_kind = "disabled"
    elif caps.style_policy == STYLE_POLICY_SCENE_AUTO:
        # Automatic Scene style policy for presets with locked Scene guidance.
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
    reference_subject: str = "main subject",
) -> EasyPresetRoute:
    """Phase 2: Evaluate preset routing matrix across all Easy Edit presets.

    Exhaustive 8-combination coverage per preset:
        none, S, Sc, Ou, S+Sc, S+Ou, Sc+Ou, S+Sc+Ou
    """
    caps = get_easy_preset_capabilities(preset)
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

    elif preset in IDENTITY_TEST_PRESETS or preset in (
        "subject_transfer_1",
        "subject_transfer_2",
        "flexible_subject_transfer_1",
        "flexible_subject_transfer_2",
    ):
        if not has_s:
            preset_warnings.append(
                f"preset '{preset}' selected but Subject source is missing; no identity can be transferred."
            )
        if not has_sc:
            preset_warnings.append(
                f"preset '{preset}' selected but Scene source is missing; target Scene is missing."
            )

        if has_sc and has_s:
            if preset == "identity_transfer":
                target_content_mode = "image"
                target_content_source = Sc
                target_content_role = "scene"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                refs.append(_ref(Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"))
                refs.append(_ref(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"))

            elif preset in (
                "subject_transfer_1",
                "subject_transfer_2",
                "flexible_subject_transfer_1",
                "flexible_subject_transfer_2",
            ):
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                s_boost = 6.0 if preset in ("subject_transfer_2", "flexible_subject_transfer_2") else 5.0
                scene_boost = (
                    SUBJECT_TRANSFER_SCENE_BOOST
                    if preset in ("subject_transfer_1", "subject_transfer_2")
                    else FLEXIBLE_SUBJECT_TRANSFER_SCENE_BOOST
                )
                refs.append(_ref(Sc, scene_boost, "scene"))
                refs.append(_ref(S, s_boost, "subject"))

        else:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = "none"
            target_geometry_mode, target_geometry_source = "favor_image", (Sc if has_sc else (S if has_s else None))
            if has_s:
                refs.append(_ref(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"))
            elif has_sc and preset == "identity_transfer":
                refs.append(_ref(Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"))

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

        if has_s:
            refs.append(_ref(S, SCENE_REINTERPRETATION_SUBJECT_BOOST, "subject"))
        if has_o:
            refs.append(_ref(Ou, SCENE_REINTERPRETATION_OUTFIT_BOOST, "outfit"))
        if not has_s:
            preset_warnings.append("Scene Reinterpretation selected but Subject source is missing.")
        if not has_sc:
            preset_warnings.append("Scene Reinterpretation selected but Scene source is missing.")

    # Default geometry fallback if geometry source was not explicitly assigned
    if target_geometry_source is None:
        target_geometry_mode, target_geometry_source = _resolve_default_geometry_source(Sc, S, Ou)

    # Flexible Subject Transfer carries the selected Outfit through a separate direct
    # semantic StyleReferenceSpec because Scene and Subject occupy both appearance slots.
    semantic_outfit_active = preset in ("flexible_subject_transfer_1", "flexible_subject_transfer_2") and has_o
    semantic_outfit_source = Ou if semantic_outfit_active else None

    # Artistic Style configuration: indirect by default so only Qwen's semantic
    # interpretation remains after the source image rows are removed.
    style_active = has_st
    if preset in ("subject_transfer_1", "subject_transfer_2"):
        style_config = SUBJECT_TRANSFER_SCENE_STYLE_CONFIG
    elif preset == "scene_reinterpretation":
        style_config = make_scene_reinterpretation_style_config(reference_subject)
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
        semantic_outfit_active=semantic_outfit_active,
        semantic_outfit_source=semantic_outfit_source,
        semantic_outfit_config=SEMANTIC_OUTFIT_STYLE_CONFIG,
        style_active=style_active,
        style_source=St,
        style_config=style_config,
        warnings=tuple(preset_warnings),
    )
