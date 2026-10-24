"""ComfyUI custom node mappings for Krea2 CcC."""

from .lora import CcCKrea2LoRAStack, CcCKrea2LoRAStackWithPrompt
from .lora_loader import CcCKrea2LoRALoader
from .t2i import CcCKrea2TextToImage
from .modular_nodes.reference_node import CcCKrea2Reference
from .modular_nodes.size_resolver_node import CcCKrea2SizeResolver
from .modular_nodes.latent_node import CcCKrea2Latent
from .modular_nodes.edit_node import CcCKrea2Edit
from .modular_nodes.edit_prompt_creator_node import CcCKrea2EditPromptCreator
from .modular_nodes.layout_node import CcCKrea2LayoutCell, CcCKrea2LayoutComposer
from .modular_nodes.paint_geometry_node import CcCKrea2PaintRestore
from .modular_nodes.paint_prepare_node import CcCKrea2PaintPrepare
from .modular_nodes.paint_node import CcCKrea2Paint

NODE_CLASS_MAPPINGS = {
    "CcCKrea2Reference": CcCKrea2Reference,
    "CcCKrea2SizeResolver": CcCKrea2SizeResolver,
    "CcCKrea2Latent": CcCKrea2Latent,
    "CcCKrea2Edit": CcCKrea2Edit,
    "CcCKrea2EditPromptCreator": CcCKrea2EditPromptCreator,
    "CcCKrea2LayoutCell": CcCKrea2LayoutCell,
    "CcCKrea2LayoutComposer": CcCKrea2LayoutComposer,
    "CcCKrea2PaintPrepare": CcCKrea2PaintPrepare,
    "CcCKrea2PaintRestore": CcCKrea2PaintRestore,
    "CcCKrea2Paint": CcCKrea2Paint,
    "CcCKrea2LoRAStackWithPrompt": CcCKrea2LoRAStackWithPrompt,
    "CcCKrea2LoRAStack": CcCKrea2LoRAStack,
    "CcCKrea2LoRALoader": CcCKrea2LoRALoader,
    "CcCKrea2TextToImage": CcCKrea2TextToImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CcCKrea2Reference": "Krea2 CcC Reference",
    "CcCKrea2SizeResolver": "Krea2 CcC Size Resolver",
    "CcCKrea2Latent": "Krea2 CcC Target Latent",
    "CcCKrea2Edit": "Krea2 CcC Edit",
    "CcCKrea2EditPromptCreator": "Krea2 CcC Edit Prompt Creator",
    "CcCKrea2LayoutCell": "Krea2 CcC Layout Cell",
    "CcCKrea2LayoutComposer": "Krea2 CcC Layout Composer",
    "CcCKrea2PaintPrepare": "Krea2 CcC Paint Prepare",
    "CcCKrea2PaintRestore": "Krea2 CcC Paint Restore",
    "CcCKrea2Paint": "Krea2 CcC Paint",
    "CcCKrea2LoRAStackWithPrompt": "Krea2 CcC Lora Stack with Prompt",
    "CcCKrea2LoRAStack": "Krea2 CcC Lora Stack",
    "CcCKrea2LoRALoader": "Krea2 CcC Lora Stack Sliders",
    "CcCKrea2TextToImage": "Krea2 CcC Text To Image",
}
