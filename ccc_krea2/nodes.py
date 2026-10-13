"""ComfyUI custom node mappings for Krea2 CcC Edit."""

from .lora import CcCKrea2LoRAPromptSettings, CcCKrea2LoRAStack
from .lora_loader import CcCKrea2LoRALoader
from .t2i import CcCKrea2TextToImage
from .modular_nodes.attention_region_node import CcCKrea2AttentionRegion
from .modular_nodes.visual_reference_node import CcCKrea2VisualReference
from .modular_nodes.vision_reference_node import CcCKrea2VisionReference
from .modular_nodes.size_resolver_node import CcCKrea2SizeResolver
from .modular_nodes.latent_node import CcCKrea2Latent
from .modular_nodes.edit_node import CcCKrea2Edit
from .modular_nodes.edit_prompt_creator_public_node import CcCKrea2EditPromptCreator
from .modular_nodes.character_sheet_node import CcCKrea2CharacterSheet
from .modular_nodes.paint_geometry_node import CcCKrea2PaintRestore
from .modular_nodes.paint_prepare_node import CcCKrea2PaintPrepare
from .modular_nodes.paint_node import CcCKrea2Paint

NODE_CLASS_MAPPINGS = {
    "CcCKrea2AttentionRegion": CcCKrea2AttentionRegion,
    "CcCKrea2VisualReference": CcCKrea2VisualReference,
    "CcCKrea2VisionReference": CcCKrea2VisionReference,
    "CcCKrea2SizeResolver": CcCKrea2SizeResolver,
    "CcCKrea2Latent": CcCKrea2Latent,
    "CcCKrea2Edit": CcCKrea2Edit,
    "CcCKrea2EditPromptCreator": CcCKrea2EditPromptCreator,
    "CcCKrea2CharacterSheet": CcCKrea2CharacterSheet,
    "CcCKrea2PaintPrepare": CcCKrea2PaintPrepare,
    "CcCKrea2PaintRestore": CcCKrea2PaintRestore,
    "CcCKrea2Paint": CcCKrea2Paint,
    "CcCKrea2LoRAPromptSettings": CcCKrea2LoRAPromptSettings,
    "CcCKrea2LoRAStack": CcCKrea2LoRAStack,
    "CcCKrea2LoRALoader": CcCKrea2LoRALoader,
    "CcCKrea2TextToImage": CcCKrea2TextToImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CcCKrea2AttentionRegion": "Krea2 CcC Attention Region",
    "CcCKrea2VisualReference": "Krea2 CcC Visual Reference",
    "CcCKrea2VisionReference": "Krea2 CcC Vision Reference",
    "CcCKrea2SizeResolver": "Krea2 CcC Size Resolver",
    "CcCKrea2Latent": "Krea2 CcC Latent",
    "CcCKrea2Edit": "Krea2 CcC Edit",
    "CcCKrea2EditPromptCreator": "Krea2 CcC Edit Prompt Creator",
    "CcCKrea2CharacterSheet": "Krea2 CcC Character Sheet",
    "CcCKrea2PaintPrepare": "Krea2 CcC Paint Prepare",
    "CcCKrea2PaintRestore": "Krea2 CcC Paint Restore",
    "CcCKrea2Paint": "Krea2 CcC Paint",
    "CcCKrea2LoRAPromptSettings": "CcC Krea2 - LoRA Prompt Settings",
    "CcCKrea2LoRAStack": "CcC Krea2 - LoRA Stack",
    "CcCKrea2LoRALoader": "Krea2 CcC LoRA Loader",
    "CcCKrea2TextToImage": "CcC Krea2 - Text to Image",
}
