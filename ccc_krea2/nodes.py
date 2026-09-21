"""ComfyUI custom node mappings for Krea2 CcC Edit."""

from .lora import CcCKrea2LoRAPromptSettings, CcCKrea2LoRAStack
from .t2i import CcCKrea2TextToImage
from .modular_nodes.attention_region_node import CcCKrea2AttentionRegion
from .modular_nodes.visual_reference_node import CcCKrea2VisualReference
from .modular_nodes.semantic_reference_node import CcCKrea2SemanticReference
from .modular_nodes.size_resolver_node import CcCKrea2SizeResolver
from .modular_nodes.latent_node import CcCKrea2Latent
from .modular_nodes.edit_node import CcCKrea2Edit
from .modular_nodes.character_sheet_node import CcCKrea2CharacterSheet
from .modular_nodes.paint_geometry_node import CcCKrea2PaintGeometry, CcCKrea2PaintRestore
from .modular_nodes.paint_prepare_node import CcCKrea2PaintPrepare
from .modular_nodes.paint_node import CcCKrea2Paint
from .modular_nodes.cache_nodes import (
    CcCKrea2ReferenceCacheCreate,
    CcCKrea2ReferenceCacheSave,
    CcCKrea2ReferenceCacheLoad,
    CcCKrea2CachedVisualReference,
)


NODE_CLASS_MAPPINGS = {
    "CcCKrea2AttentionRegion": CcCKrea2AttentionRegion,
    "CcCKrea2VisualReference": CcCKrea2VisualReference,
    "CcCKrea2SemanticReference": CcCKrea2SemanticReference,
    "CcCKrea2SizeResolver": CcCKrea2SizeResolver,
    "CcCKrea2Latent": CcCKrea2Latent,
    "CcCKrea2Edit": CcCKrea2Edit,
    "CcCKrea2CharacterSheet": CcCKrea2CharacterSheet,
    "CcCKrea2PaintGeometry": CcCKrea2PaintGeometry,
    "CcCKrea2PaintPrepare": CcCKrea2PaintPrepare,
    "CcCKrea2PaintRestore": CcCKrea2PaintRestore,
    "CcCKrea2Paint": CcCKrea2Paint,
    "CcCKrea2ReferenceCacheCreate": CcCKrea2ReferenceCacheCreate,
    "CcCKrea2ReferenceCacheSave": CcCKrea2ReferenceCacheSave,
    "CcCKrea2ReferenceCacheLoad": CcCKrea2ReferenceCacheLoad,
    "CcCKrea2CachedVisualReference": CcCKrea2CachedVisualReference,
    "CcCKrea2LoRAPromptSettings": CcCKrea2LoRAPromptSettings,
    "CcCKrea2LoRAStack": CcCKrea2LoRAStack,
    "CcCKrea2TextToImage": CcCKrea2TextToImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "CcCKrea2AttentionRegion": "Krea2 CcC Attention Region",
    "CcCKrea2VisualReference": "Krea2 CcC Visual Reference",
    "CcCKrea2SemanticReference": "Krea2 CcC Semantic Reference",
    "CcCKrea2SizeResolver": "Krea2 CcC Size Resolver",
    "CcCKrea2Latent": "Krea2 CcC Latent",
    "CcCKrea2Edit": "Krea2 CcC Edit",
    "CcCKrea2CharacterSheet": "Krea2 CcC Character Sheet",
    "CcCKrea2PaintGeometry": "Krea2 CcC Paint Geometry",
    "CcCKrea2PaintPrepare": "Krea2 CcC Paint Prepare",
    "CcCKrea2PaintRestore": "Krea2 CcC Paint Restore",
    "CcCKrea2Paint": "Krea2 CcC Paint",
    "CcCKrea2ReferenceCacheCreate": "Krea2 CcC Reference Cache Create",
    "CcCKrea2ReferenceCacheSave": "Krea2 CcC Reference Cache Save",
    "CcCKrea2ReferenceCacheLoad": "Krea2 CcC Reference Cache Load",
    "CcCKrea2CachedVisualReference": "Krea2 CcC Cached Visual Reference",
    "CcCKrea2LoRAPromptSettings": "CcC Krea2 - LoRA Prompt Settings",
    "CcCKrea2LoRAStack": "CcC Krea2 - LoRA Stack",
    "CcCKrea2TextToImage": "CcC Krea2 - Text to Image",
}
