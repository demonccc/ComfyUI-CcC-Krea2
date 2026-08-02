"""Central preset definitions, custom socket types, and precedence resolver for CcC Krea2."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger("CcCKrea2")

# Custom ComfyUI Socket Types
CCC_KREA2_IMAGE_ADVANCED_SETTINGS = "CCC_KREA2_IMAGE_ADVANCED_SETTINGS"
CCC_KREA2_EDIT_ADVANCED_SETTINGS = "CCC_KREA2_EDIT_ADVANCED_SETTINGS"

RESIZE_METHODS = [
    "auto",
    "nearest-exact",
    "bilinear",
    "bicubic",
    "area",
    "lanczos",
]

PRESET_CHOICES = ["balanced", "max_identity", "flexible"]
ROLE_CHOICES = ["subject", "scene", "outfit", "source"]


@dataclass(frozen=True)
class ImageRoleSettings:
    """Settings overrides for a specific image role."""

    boost: float = 1.0
    mask_invert: bool = False
    grounding_resize_mode: str = "normalize"
    grounding_px: int = 768
    grounding_min_px: int = 512
    grounding_max_px: int = 1024
    grounding_resize_method: str = "auto"
    reference_fit_mode: str = "fit"
    reference_resize_method: str = "auto"


@dataclass(frozen=True)
class ImageAdvancedSettingsBundle:
    """Immutable chained container mapping image roles to their ImageRoleSettings."""

    role_settings: Dict[str, ImageRoleSettings] = field(default_factory=dict)

    def with_role(self, role: str, settings: ImageRoleSettings) -> "ImageAdvancedSettingsBundle":
        """Return a new bundle with the specified role setting updated (last node wins)."""
        new_map = dict(self.role_settings)
        new_map[role] = settings
        return ImageAdvancedSettingsBundle(role_settings=new_map)


@dataclass(frozen=True)
class EditAdvancedSettings:
    """Global edit advanced settings."""

    batch_size: int = 1
    sampling_resize_mode: str = "fit"
    sampling_resize_method: str = "auto"
    attention_mask_mode: str = "hard"
    role_resolution_limit_mode: str = "max_megapixels"
    role_resolution_max_megapixels: float = 2.0
    custom_aspect_source: str = "auto"
    prompt_instructions_mode: str = "automatic"
    prompt_instructions: str = ""
    inpaint_mask_invert: bool = False
    inpaint_mask_grow: int = 0
    inpaint_mask_blur: int = 0


@dataclass(frozen=True)
class ResolvedRoleSettings:
    """Resolved settings for a single image role after applying presets and overrides."""

    boost: float
    mask_invert: bool
    grounding_resize_mode: str
    grounding_px: int
    grounding_min_px: int
    grounding_max_px: int
    grounding_resize_method: str
    reference_fit_mode: str
    reference_resize_method: str


@dataclass(frozen=True)
class ResolvedKrea2Settings:
    """Complete resolved settings bundle passed to engine execution."""

    batch_size: int
    sampling_resize_mode: str
    sampling_resize_method: str
    attention_mask_mode: str
    role_resolution_limit_mode: str
    role_resolution_max_megapixels: float
    custom_aspect_source: str
    prompt_instructions_mode: str
    prompt_instructions: str
    inpaint_mask_invert: bool
    inpaint_mask_grow: int
    inpaint_mask_blur: int
    roles: Dict[str, ResolvedRoleSettings]


def _get_balanced_role_defaults(boost: float = 1.0) -> Dict[str, Any]:
    return {
        "boost": boost,
        "mask_invert": False,
        "grounding_resize_mode": "normalize",
        "grounding_px": 768,
        "grounding_min_px": 512,
        "grounding_max_px": 1024,
        "grounding_resize_method": "auto",
        "reference_fit_mode": "fit",
        "reference_resize_method": "auto",
    }


# Centralized preset definitions
PRESET_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "global": {
            "batch_size": 1,
            "sampling_resize_mode": "fit",
            "sampling_resize_method": "auto",
            "attention_mask_mode": "hard",
            "role_resolution_limit_mode": "max_megapixels",
            "role_resolution_max_megapixels": 2.0,
            "custom_aspect_source": "auto",
            "prompt_instructions_mode": "automatic",
            "prompt_instructions": "",
            "inpaint_mask_invert": False,
            "inpaint_mask_grow": 0,
            "inpaint_mask_blur": 0,
        },
        "roles": {
            "subject": _get_balanced_role_defaults(boost=2.5),
            "scene": _get_balanced_role_defaults(boost=1.0),
            "outfit": _get_balanced_role_defaults(boost=1.0),
            "source": _get_balanced_role_defaults(boost=1.0),
        },
    },
    "max_identity": {
        "global": {
            "batch_size": 1,
            "sampling_resize_mode": "fit",
            "sampling_resize_method": "auto",
            "attention_mask_mode": "hard",
            "role_resolution_limit_mode": "max_megapixels",
            "role_resolution_max_megapixels": 2.0,
            "custom_aspect_source": "auto",
            "prompt_instructions_mode": "automatic",
            "prompt_instructions": "",
            "inpaint_mask_invert": False,
            "inpaint_mask_grow": 0,
            "inpaint_mask_blur": 0,
        },
        "roles": {
            "subject": {**_get_balanced_role_defaults(boost=4.0), "grounding_px": 1024},
            "scene": _get_balanced_role_defaults(boost=1.0),
            "outfit": _get_balanced_role_defaults(boost=1.0),
            "source": {**_get_balanced_role_defaults(boost=2.5), "grounding_px": 1024},
        },
    },
    "flexible": {
        "global": {
            "batch_size": 1,
            "sampling_resize_mode": "fit",
            "sampling_resize_method": "auto",
            "attention_mask_mode": "hard",
            "role_resolution_limit_mode": "max_megapixels",
            "role_resolution_max_megapixels": 2.0,
            "custom_aspect_source": "auto",
            "prompt_instructions_mode": "automatic",
            "prompt_instructions": "",
            "inpaint_mask_invert": False,
            "inpaint_mask_grow": 0,
            "inpaint_mask_blur": 0,
        },
        "roles": {
            "subject": {**_get_balanced_role_defaults(boost=1.5), "grounding_px": 512},
            "scene": _get_balanced_role_defaults(boost=1.0),
            "outfit": _get_balanced_role_defaults(boost=1.0),
            "source": {**_get_balanced_role_defaults(boost=1.5), "grounding_px": 512},
        },
    },
}


def resolve_krea2_settings(
    preset_name: str = "balanced",
    edit_settings: Optional[EditAdvancedSettings] = None,
    image_settings: Optional[ImageAdvancedSettingsBundle] = None,
    active_roles: Optional[List[str]] = None,
) -> ResolvedKrea2Settings:
    """Resolve full Krea2 configuration with precedence:

    internal defaults -> selected main preset -> edit_advanced_settings -> image_advanced_settings per role
    """
    preset_key = preset_name if preset_name in PRESET_DEFINITIONS else "balanced"
    preset_def = PRESET_DEFINITIONS[preset_key]

    # Global resolution
    g_def = dict(preset_def["global"])
    if edit_settings is not None:
        g_def["batch_size"] = edit_settings.batch_size
        g_def["sampling_resize_mode"] = edit_settings.sampling_resize_mode
        g_def["sampling_resize_method"] = edit_settings.sampling_resize_method
        g_def["attention_mask_mode"] = edit_settings.attention_mask_mode
        g_def["role_resolution_limit_mode"] = edit_settings.role_resolution_limit_mode
        g_def["role_resolution_max_megapixels"] = edit_settings.role_resolution_max_megapixels
        g_def["custom_aspect_source"] = edit_settings.custom_aspect_source
        g_def["prompt_instructions_mode"] = edit_settings.prompt_instructions_mode
        g_def["prompt_instructions"] = edit_settings.prompt_instructions
        g_def["inpaint_mask_invert"] = edit_settings.inpaint_mask_invert
        g_def["inpaint_mask_grow"] = edit_settings.inpaint_mask_grow
        g_def["inpaint_mask_blur"] = edit_settings.inpaint_mask_blur

    # Role resolution
    resolved_roles: Dict[str, ResolvedRoleSettings] = {}
    bundled_roles = image_settings.role_settings if image_settings is not None else {}

    # Warn if image_settings contains roles not used by current node
    if active_roles is not None:
        for r_name in bundled_roles:
            if r_name not in active_roles:
                logger.warning(
                    f"[CcC Krea2] Image Advanced Settings entry for role '{r_name}' "
                    f"is not used by current node (active: {active_roles}) and will be ignored."
                )

    for role in ROLE_CHOICES:
        r_preset = dict(preset_def["roles"].get(role, _get_balanced_role_defaults()))

        if role in bundled_roles:
            role_ov = bundled_roles[role]
            r_preset["boost"] = role_ov.boost
            r_preset["mask_invert"] = role_ov.mask_invert
            r_preset["grounding_resize_mode"] = role_ov.grounding_resize_mode
            r_preset["grounding_px"] = role_ov.grounding_px
            r_preset["grounding_min_px"] = role_ov.grounding_min_px
            r_preset["grounding_max_px"] = role_ov.grounding_max_px
            r_preset["grounding_resize_method"] = role_ov.grounding_resize_method
            r_preset["reference_fit_mode"] = role_ov.reference_fit_mode
            r_preset["reference_resize_method"] = role_ov.reference_resize_method

        resolved_roles[role] = ResolvedRoleSettings(
            boost=float(r_preset["boost"]),
            mask_invert=bool(r_preset["mask_invert"]),
            grounding_resize_mode=str(r_preset["grounding_resize_mode"]),
            grounding_px=int(r_preset["grounding_px"]),
            grounding_min_px=int(r_preset["grounding_min_px"]),
            grounding_max_px=int(r_preset["grounding_max_px"]),
            grounding_resize_method=str(r_preset["grounding_resize_method"]),
            reference_fit_mode=str(r_preset["reference_fit_mode"]),
            reference_resize_method=str(r_preset["reference_resize_method"]),
        )

    return ResolvedKrea2Settings(
        batch_size=int(g_def["batch_size"]),
        sampling_resize_mode=str(g_def["sampling_resize_mode"]),
        sampling_resize_method=str(g_def["sampling_resize_method"]),
        attention_mask_mode=str(g_def["attention_mask_mode"]),
        role_resolution_limit_mode=str(g_def["role_resolution_limit_mode"]),
        role_resolution_max_megapixels=float(g_def["role_resolution_max_megapixels"]),
        custom_aspect_source=str(g_def["custom_aspect_source"]),
        prompt_instructions_mode=str(g_def["prompt_instructions_mode"]),
        prompt_instructions=str(g_def["prompt_instructions"]),
        inpaint_mask_invert=bool(g_def["inpaint_mask_invert"]),
        inpaint_mask_grow=int(g_def["inpaint_mask_grow"]),
        inpaint_mask_blur=int(g_def["inpaint_mask_blur"]),
        roles=resolved_roles,
    )
