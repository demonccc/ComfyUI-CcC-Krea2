"""ComfyUI Custom Node mappings for the CcC Krea2 suite."""

from .lora import CcCKrea2LoRAPromptSettings, CcCKrea2LoRAStack
from .t2i import CcCKrea2TextToImage
from .modular_nodes.visual_reference_node import CcCKrea2VisualReference
from .modular_nodes.semantic_reference_node import CcCKrea2SemanticReference
from .modular_nodes.latent_node import CcCKrea2Latent
from .modular_nodes.edit_node import CcCKrea2Edit


NODE_CLASS_MAPPINGS = {
    "CcCKrea2VisualReference": CcCKrea2VisualReference,
    "CcCKrea2SemanticReference": CcCKrea2SemanticReference,
    "CcCKrea2Latent": CcCKrea2Latent,
    "CcCKrea2Edit": CcCKrea2Edit,
    "CcCKrea2LoRAPromptSettings": CcCKrea2LoRAPromptSettings,
    "CcCKrea2LoRAStack": CcCKrea2LoRAStack,
    "CcCKrea2TextToImage": CcCKrea2TextToImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CcCKrea2VisualReference": "Krea2 CcC Visual Reference",
    "CcCKrea2SemanticReference": "Krea2 CcC Semantic Reference",
    "CcCKrea2Latent": "Krea2 CcC Latent",
    "CcCKrea2Edit": "Krea2 CcC Edit",
    "CcCKrea2LoRAPromptSettings": "CcC Krea2 - LoRA Prompt Settings",
    "CcCKrea2LoRAStack": "CcC Krea2 - LoRA Stack",
    "CcCKrea2TextToImage": "CcC Krea2 - Text to Image",
}
