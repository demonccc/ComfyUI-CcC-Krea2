"""CcC Krea2 - Reference Image node (Generic Reference Node)."""

from typing import Optional, Dict, Any, Tuple
import torch
from ..reference_specs import ReferenceSpec, ReferenceChain, PreparedVisionImage
from ..constants import REFERENCE_FIT_MODES


class CcCKrea2ReferenceImage:
    """Generic reference node establishing edit or style reference specifications."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("REFERENCE_CHAIN",)
    RETURN_NAMES = ("reference_chain",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_path": (["edit", "style"], {"default": "edit"}),
                "prepared_image": ("PREPARED_VISION_IMAGE",),
                "vision_slot": (["auto", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], {"default": "auto"}),
                "alias": ("STRING", {"default": ""}),
                "vision_instruction": ("STRING", {"multiline": True, "default": ""}),
                "attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "masked_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "visual_reference_fit": (REFERENCE_FIT_MODES, {"default": "auto"}),
                "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "style_processing": (["full", "2x2", "4x4"], {"default": "2x2"}),
                "indirect_style_transfer": ("BOOLEAN", {"default": True}),
                "style_directive": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_references": ("REFERENCE_CHAIN",),
                "attention_mask": ("MASK",),
            }
        }

    def process(
        self,
        reference_path: str,
        prepared_image: PreparedVisionImage,
        vision_slot: str = "auto",
        alias: str = "",
        vision_instruction: str = "",
        attention_boost: float = 1.0,
        masked_attention_boost: float = 1.0,
        visual_reference_fit: str = "auto",
        style_fidelity: float = 0.5,
        style_processing: str = "2x2",
        indirect_style_transfer: bool = True,
        style_directive: bool = True,
        previous_references: Optional[ReferenceChain] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[ReferenceChain]:
        chain = previous_references if previous_references is not None else ReferenceChain()

        slot_int = None
        if vision_slot and vision_slot.lower() != "auto":
            try:
                slot_int = int(vision_slot)
            except ValueError:
                slot_int = None

        spec = ReferenceSpec(
            reference_path=reference_path,
            prepared_image=prepared_image,
            requested_vision_slot=slot_int,
            alias=alias.strip(),
            vision_instruction=vision_instruction.strip(),
            attention_boost=attention_boost,
            masked_attention_boost=masked_attention_boost,
            attention_mask=attention_mask,
            visual_reference_fit=visual_reference_fit,
            style_fidelity=style_fidelity,
            style_processing=style_processing,
            indirect_style_transfer=indirect_style_transfer,
            style_directive=style_directive,
        )

        return (chain.append(spec),)
