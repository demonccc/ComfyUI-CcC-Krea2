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

EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_WITH_OUTFIT = (
    "Replace only the {reference_subject} of the {scene_source} with the {subject} of the {subject_source}.\n\n"
    "Preserve the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n"
    "Dress the transferred {subject} using the clothing and accessories from the {outfit_source}.\n\n"
    "Do not preserve the clothing or accessories of the {subject} from the {subject_source} when an explicit outfit source is selected. Use the clothing and accessories from the {outfit_source} instead.\n\n"
    "Keep every other person and the rest of the scene unchanged."
)

EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER = (
    "Transfer only the outfit and accessories from the {outfit_source} to the {subject}.\n\n"
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
    "Place the {subject} from the {subject_source} naturally into the {scene_source} wearing the outfit and accessories from the {outfit_source}.\n\n"
    "Preserve the {subject} identity, body shape, and body proportions.\n\n"
    "Preserve the scene composition, environment, framing, perspective, and spatial layout.\n\n"
    "Do not preserve the {subject} clothing.\n\n"
    "Fit the transferred outfit and accessories naturally to the {subject} and the scene.\n\n"
    "Keep accessories physically attached to the {subject} in a natural way and never floating.\n\n"
    "Do not duplicate accessories."
)

EASY_DEFAULT_PROMPT_STYLE = (
    "Apply the visual style from the {style_source} while preserving the {subject} identity, content, geometry, framing, and composition.\n\n"
    "Transfer only the visual style, including its color palette, texture, lighting character, and overall visual mood.\n\n"
    "Do not copy subjects, objects, or scene content from the {style_source}."
)


OUTFIT_POLICY_USER = "user"
OUTFIT_POLICY_DISABLED = "disabled"

STYLE_POLICY_USER = "user"
STYLE_POLICY_SCENE_AUTO = "scene_auto"
STYLE_POLICY_SCENE_OUTFIT_AUTO = "scene_outfit_auto"
STYLE_POLICY_DISABLED = "disabled"

SCENE_REINTERPRETATION_SUBJECT_BOOST = 4.0

EASY_SCENE_REINTERPRETATION_SCENE_INSTRUCTION = (
    "Use this reference for the main subject's action, activity, pose, body dynamics, environment, spatial context, "
    "camera framing, perspective, lighting, and broad outfit concept. Reinterpret these elements creatively for the "
    "subject reference. Do not use the scene subject's identity as the generated subject identity."
)

EASY_SCENE_OUTFIT_STYLE_INSTRUCTION = (
    "Use the Scene image only as a visual reference for the clothing and accessories worn by the target subject. "
    "Reinforce their garment type, cut, coverage, colors, materials, and accessories. "
    "Do not use this style conditioning to transfer the identity, body, pose, or other people from the Scene."
)

EASY_PRESET_DISPLAY_LABELS = {
    "flexible": "Flexible",
    "balanced": "Balanced",
    "consistent": "Consistent",
    "preserve_identity": "Preserve Identity",
    "max_identity": "Max Identity",
    "identity_transfer": "Identity Transfer",
    "transfer_identity_test_2": "Transfer Identity Test 2",
    "transfer_identity_test_3": "Transfer Identity Test 3 - Scene 4 / Subject 7",
    "transfer_identity_test_4": "Transfer Identity Test 4 - Scene 2.5 / Subject 9",
    "transfer_identity_test_5": "Transfer Identity Test 5",
    "transfer_identity_test_6": "Transfer Identity Test 6",
    "subject_transfer": "Subject Transfer",
    "preserve_scene": "Preserve Scene",
    "outfit_transfer": "Outfit Transfer",
    "style_transfer": "Style Transfer",
    "scene_reinterpretation": "Scene Reinterpretation",
    # Group A
    "transfer_identity_test_a_4_4": "Transfer Identity A - Scene 4 / Subject 4",
    "transfer_identity_test_a_4_5": "Transfer Identity A - Scene 4 / Subject 5",
    "transfer_identity_test_a_4_6": "Transfer Identity A - Scene 4 / Subject 6",
    # Group B
    "transfer_identity_test_b_2_5_4": "Transfer Identity B - Scene 2.5 / Subject 4",
    "transfer_identity_test_b_2_5_5": "Transfer Identity B - Scene 2.5 / Subject 5",
    "transfer_identity_test_b_2_5_6": "Transfer Identity B - Scene 2.5 / Subject 6",
    # Group C
    "transfer_identity_test_c_4_4": "Transfer Identity C - Scene 4 / Subject 4 + Outfit Style",
    "transfer_identity_test_c_4_5": "Transfer Identity C - Scene 4 / Subject 5 + Outfit Style",
    "transfer_identity_test_c_4_6": "Transfer Identity C - Scene 4 / Subject 6 + Outfit Style",
    "transfer_identity_test_c_4_7": "Transfer Identity C - Scene 4 / Subject 7 + Outfit Style",
    "transfer_identity_test_c_2_5_4": "Transfer Identity C - Scene 2.5 / Subject 4 + Outfit Style",
    "transfer_identity_test_c_2_5_5": "Transfer Identity C - Scene 2.5 / Subject 5 + Outfit Style",
    "transfer_identity_test_c_2_5_6": "Transfer Identity C - Scene 2.5 / Subject 6 + Outfit Style",
    "transfer_identity_test_c_2_5_9": "Transfer Identity C - Scene 2.5 / Subject 9 + Outfit Style",
    # Group D
    "transfer_identity_test_d_s2_5_o2_5": "Transfer Identity D - Subject 2.5 / Outfit 2.5",
    "transfer_identity_test_d_s2_5_o4": "Transfer Identity D - Subject 2.5 / Outfit 4",
    "transfer_identity_test_d_s4_o4": "Transfer Identity D - Subject 4 / Outfit 4",
    "transfer_identity_test_d_s5_o4": "Transfer Identity D - Subject 5 / Outfit 4",
    "transfer_identity_test_d_s6_o4": "Transfer Identity D - Subject 6 / Outfit 4",
    "transfer_identity_test_d_s7_o4": "Transfer Identity D - Subject 7 / Outfit 4",
}

IDENTITY_TEST_PRESETS = (
    "identity_transfer",
    "transfer_identity_test_2",
    "transfer_identity_test_3",
    "transfer_identity_test_4",
    "transfer_identity_test_5",
    "transfer_identity_test_6",
    # Group A
    "transfer_identity_test_a_4_4",
    "transfer_identity_test_a_4_5",
    "transfer_identity_test_a_4_6",
    # Group B
    "transfer_identity_test_b_2_5_4",
    "transfer_identity_test_b_2_5_5",
    "transfer_identity_test_b_2_5_6",
    # Group C
    "transfer_identity_test_c_4_4",
    "transfer_identity_test_c_4_5",
    "transfer_identity_test_c_4_6",
    "transfer_identity_test_c_4_7",
    "transfer_identity_test_c_2_5_4",
    "transfer_identity_test_c_2_5_5",
    "transfer_identity_test_c_2_5_6",
    "transfer_identity_test_c_2_5_9",
    # Group D
    "transfer_identity_test_d_s2_5_o2_5",
    "transfer_identity_test_d_s2_5_o4",
    "transfer_identity_test_d_s4_o4",
    "transfer_identity_test_d_s5_o4",
    "transfer_identity_test_d_s6_o4",
    "transfer_identity_test_d_s7_o4",
)

EASY_DEFAULT_PROMPT_SCENE_REINTERPRETATION = (
    "Create a new image of the {subject} from the {subject_source} performing the main action or activity shown by the {reference_subject} in the {scene_source}.\n\n"
    "Preserve the identity, facial features, hair, anatomy, body shape, and body proportions of the {subject} from the {subject_source}.\n\n"
    "Use the {scene_source} as inspiration for the action, pose, body dynamics, environment, spatial context, camera framing, perspective, and lighting, but reinterpret the scene creatively rather than reproducing it pixel-for-pixel.\n\n"
    "Adapt the {subject} naturally to the referenced action and environment.\n\n"
    "Creatively reinterpret the clothing and accessories worn by the {reference_subject} in the {scene_source} so they are appropriate for the {subject} and the newly generated image. Do not copy the original scene outfit literally.\n\n"
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

    fmt_context = {
        "reference_subject": ref_subj,
        "subject": subj,
        "subject_description": subj_desc,
        "scene_source": scene_src,
        "subject_source": subj_src,
        "outfit_source": outfit_src,
        "style_source": style_src,
    }
    return template.format(**fmt_context)


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
    - BFS Body Swap LoRA (bfs_body_swap_v1_krea2.safetensors) is naturally aligned with subject_transfer
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
    if outfit_source in {"outfit image", "scene image", "style image"}:
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

    elif preset == "subject_transfer":
        if not (has_s and has_sc):
            return False, "", "none"

        if not has_o:
            base_template = EASY_DEFAULT_PROMPT_SUBJECT_TRANSFER_NO_OUTFIT
            base_key = "subject_transfer"
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

        return True, render_easy_prompt(base_template, context), base_key

    elif preset == "scene_reinterpretation":
        if not (has_s and has_sc):
            return False, "", "none"
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
    "transfer_identity_test_2": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "transfer_identity_test_3": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "transfer_identity_test_4": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "transfer_identity_test_5": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED
    ),
    "transfer_identity_test_6": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    "subject_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_SCENE_AUTO),
    "preserve_scene": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_USER),
    "outfit_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_USER, style_policy=STYLE_POLICY_USER),
    "style_transfer": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_USER),
    "scene_reinterpretation": EasyPresetCapabilities(
        outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO
    ),
    # Group A — Test 3 family (Scene 4.0 / Subject variable)
    "transfer_identity_test_a_4_4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO),
    "transfer_identity_test_a_4_5": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO),
    "transfer_identity_test_a_4_6": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO),
    # Group B — Test 4 family (Scene 2.5 / Subject variable)
    "transfer_identity_test_b_2_5_4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO),
    "transfer_identity_test_b_2_5_5": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO),
    "transfer_identity_test_b_2_5_6": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_AUTO),
    # Group C — Test 3 & 4 families with Outfit Style
    "transfer_identity_test_c_4_4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_4_5": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_4_6": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_4_7": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_2_5_4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_2_5_5": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_2_5_6": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    "transfer_identity_test_c_2_5_9": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_SCENE_OUTFIT_AUTO),
    # Group D — Scene latent + Subject reference + Scene as Outfit reference
    "transfer_identity_test_d_s2_5_o2_5": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED),
    "transfer_identity_test_d_s2_5_o4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED),
    "transfer_identity_test_d_s4_o4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED),
    "transfer_identity_test_d_s5_o4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED),
    "transfer_identity_test_d_s6_o4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED),
    "transfer_identity_test_d_s7_o4": EasyPresetCapabilities(outfit_policy=OUTFIT_POLICY_DISABLED, style_policy=STYLE_POLICY_DISABLED),
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
    elif caps.style_policy in (STYLE_POLICY_SCENE_AUTO, STYLE_POLICY_SCENE_OUTFIT_AUTO):
        # Automatic Scene style policy for presets using STYLE_POLICY_SCENE_AUTO / STYLE_POLICY_SCENE_OUTFIT_AUTO
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

    elif preset in IDENTITY_TEST_PRESETS:
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
                target_content_source = S
                target_content_role = "subject"
                target_content_fit = "contain_no_upscale"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                refs.append(_ref(Sc, IDENTITY_TRANSFER_SCENE_BOOST, "scene"))
                refs.append(_ref(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"))

            elif preset == "transfer_identity_test_2":
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                refs.append(_ref(Sc, 2.5, "scene"))
                refs.append(_ref(S, 7.0, "subject"))

            elif preset == "transfer_identity_test_3":
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                refs.append(_ref(Sc, 4.0, "scene"))
                refs.append(_ref(S, 7.0, "subject"))

            elif preset == "transfer_identity_test_4":
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                refs.append(_ref(Sc, 2.5, "scene"))
                refs.append(_ref(S, 9.0, "subject"))

            elif preset in ("transfer_identity_test_5", "transfer_identity_test_6"):
                target_content_mode = "image"
                target_content_source = Sc
                target_content_role = "scene"
                target_content_fit = "contain_no_upscale"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                refs.append(_ref(S, 7.0, "subject"))

            elif preset.startswith("transfer_identity_test_a_"):
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                subj_boost = 4.0 if preset == "transfer_identity_test_a_4_4" else (5.0 if preset == "transfer_identity_test_a_4_5" else 6.0)
                refs.append(_ref(Sc, 4.0, "scene"))
                refs.append(_ref(S, subj_boost, "subject"))

            elif preset.startswith("transfer_identity_test_b_"):
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                subj_boost = 4.0 if preset == "transfer_identity_test_b_2_5_4" else (5.0 if preset == "transfer_identity_test_b_2_5_5" else 6.0)
                refs.append(_ref(Sc, 2.5, "scene"))
                refs.append(_ref(S, subj_boost, "subject"))

            elif preset.startswith("transfer_identity_test_c_"):
                target_content_mode = "empty"
                target_content_source = None
                target_content_role = "none"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                if preset.startswith("transfer_identity_test_c_4_"):
                    sc_boost = 4.0
                    s_val = preset.replace("transfer_identity_test_c_4_", "")
                else:
                    sc_boost = 2.5
                    s_val = preset.replace("transfer_identity_test_c_2_5_", "")
                subj_boost = float(s_val)

                refs.append(_ref(Sc, sc_boost, "scene"))
                refs.append(_ref(S, subj_boost, "subject"))

            elif preset.startswith("transfer_identity_test_d_"):
                target_content_mode = "image"
                target_content_source = Sc
                target_content_role = "scene"
                target_content_fit = "contain_no_upscale"
                target_geometry_mode, target_geometry_source = "favor_image", Sc

                if preset == "transfer_identity_test_d_s2_5_o2_5":
                    subj_boost, outfit_boost = 2.5, 2.5
                elif preset == "transfer_identity_test_d_s2_5_o4":
                    subj_boost, outfit_boost = 2.5, 4.0
                elif preset == "transfer_identity_test_d_s4_o4":
                    subj_boost, outfit_boost = 4.0, 4.0
                elif preset == "transfer_identity_test_d_s5_o4":
                    subj_boost, outfit_boost = 5.0, 4.0
                elif preset == "transfer_identity_test_d_s6_o4":
                    subj_boost, outfit_boost = 6.0, 4.0
                elif preset == "transfer_identity_test_d_s7_o4":
                    subj_boost, outfit_boost = 7.0, 4.0
                else:
                    subj_boost, outfit_boost = 7.0, 4.0

                refs.append(_ref(S, subj_boost, "subject"))
                refs.append(_ref(Sc, outfit_boost, "outfit"))
        else:
            target_content_mode = "empty"
            target_content_source = None
            target_content_role = "none"
            target_geometry_mode, target_geometry_source = "favor_image", (Sc if has_sc else (S if has_s else None))
            if has_s:
                refs.append(_ref(S, IDENTITY_TRANSFER_SUBJECT_BOOST, "subject"))
            elif has_sc:
                scene_boost = 4.0 if ("4_4" in preset or "4_5" in preset or "4_6" in preset or "4_7" in preset or "test_3" in preset) else 2.5
                refs.append(_ref(Sc, scene_boost, "scene"))

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
    if caps.style_policy == STYLE_POLICY_SCENE_OUTFIT_AUTO:
        style_config = EasyStyleConfig(
            style_fidelity=1.0,
            style_processing="2x2",
            indirect_style_transfer=False,
            vision_instruction=EASY_SCENE_OUTFIT_STYLE_INSTRUCTION,
        )
    elif preset == "style_transfer":
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

