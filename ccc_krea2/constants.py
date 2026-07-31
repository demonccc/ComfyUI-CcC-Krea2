"""Constants and default configurations for CcC Krea2."""

import enum

NODE_CATEGORY = "CcC/Krea2"
LOGGER_PREFIX = "[CcC Krea2]"

class ReferenceRole(str, enum.Enum):
    SUBJECT = "subject"
    OUTFIT = "outfit"
    SCENE = "scene"
    SOURCE = "source"

# Centralized reference role order mappings.
ROLE_ORDER_SUBJECT = [ReferenceRole.SUBJECT]
ROLE_ORDER_SUBJECT_OUTFIT = [ReferenceRole.OUTFIT, ReferenceRole.SUBJECT]
ROLE_ORDER_SUBJECT_SCENE = [ReferenceRole.SCENE, ReferenceRole.SUBJECT]
ROLE_ORDER_SUBJECT_SCENE_OUTFIT = [ReferenceRole.SCENE, ReferenceRole.OUTFIT, ReferenceRole.SUBJECT]
ROLE_ORDER_INPAINT = [ReferenceRole.SOURCE]
ROLE_ORDER_INPAINT_SUBJECT_OUTFIT = [ReferenceRole.OUTFIT, ReferenceRole.SUBJECT]
ROLE_ORDER_INPAINT_SUBJECT_SCENE = [ReferenceRole.SCENE, ReferenceRole.SUBJECT]

GROUNDING_RESIZE_MODES = ["normalize", "downscale_only", "clamp", "none"]
GROUNDING_PRESETS = ["balanced", "max_identity", "custom"]
SAMPLING_RESIZE_MODES = ["fit", "crop", "stretch"]
REFERENCE_FIT_MODES = ["fit", "crop"]
ATTENTION_MASK_MODES = ["hard", "soft"]

DEFAULT_GROUNDING_PX_SUBJECT = 768
DEFAULT_GROUNDING_PX_SCENE = 768
DEFAULT_GROUNDING_PX_OUTFIT = 768
DEFAULT_GROUNDING_PX_SOURCE = 768

DEFAULT_GROUNDING_MIN_PX = 512
DEFAULT_GROUNDING_MAX_PX = 1024

DEFAULT_BOOST_SUBJECT = 2.5
DEFAULT_BOOST_SCENE = 1.0
DEFAULT_BOOST_OUTFIT = 1.0
DEFAULT_BOOST_SOURCE = 1.0

DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)

VISION_PAD_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

EXPERIMENTAL_OUTFIT_WARNING = (
    "Reference interpretation depends on the loaded Krea 2 edit LoRA, "
    "reference order and prompt. Identity Edit v1.2 is the recommended starting point."
)
