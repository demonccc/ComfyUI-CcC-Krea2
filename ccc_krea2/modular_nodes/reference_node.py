"""CcC Krea2 - Reference Image node (Generic Reference Node)."""

from typing import Optional, Dict, Any, Tuple
import torch
from ..reference_specs import ReferenceSpec, ReferenceChain, PreparedVisionImage
from ..constants import GENERIC_REFERENCE_FIT_MODES


class CcCKrea2ReferenceImage:
    """Generic reference node establishing edit or style reference specifications."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("REFERENCE_CHAIN",)
    RETURN_NAMES = ("reference_chain",)
    FUNCTION = "process"

    DESCRIPTION = "Generic Reference Image node for edit (spatial VAE + Qwen vision) and style (Moodboard grid) references."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_path": (["edit", "style"], {"default": "edit", "tooltip": "Selects edit (spatial appearance/semantic reference) or style (Moodboard grid reference)."}),
                "prepared_image": ("PREPARED_VISION_IMAGE", {"tooltip": "Prepared vision image from CcC Krea2 Qwen Vision Image Prep."}),
                "vision_slot": (["auto", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], {"default": "auto", "tooltip": "Controls Qwen semantic image ordering. This is not the VAE appearance frame."}),
                "alias": ("STRING", {"default": "", "tooltip": "Alias name for reference tag in prompt."}),
                "vision_instruction": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional explicit instruction describing what Qwen should use from this reference."}),
                "attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "CcC Krea2 Edit only. Multiplies target-to-reference attention. 1.0 is neutral."}),
                "masked_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "CcC Krea2 Edit only. Additional multiplier inside the attention mask."}),
                "visual_reference_fit": (GENERIC_REFERENCE_FIT_MODES, {"default": "auto", "tooltip": "Specifies fit mode for Krea2 geometry scaling: auto, fit, or crop."}),
                "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Style path only. Controls style fidelity blending."}),
                "style_processing": (["full", "2x2", "4x4"], {"default": "2x2", "tooltip": "Style path only. Moodboard tile grid resolution."}),
                "indirect_style_transfer": ("BOOLEAN", {"default": True, "tooltip": "Style path only. Removes style vision rows post-encoding when True."}),
            },

            "optional": {
                "previous_references": ("REFERENCE_CHAIN", {"tooltip": "Chained input from previous reference node."}),
                "attention_mask": ("MASK", {"tooltip": "Optional spatial attention mask."}),
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
        style_directive: bool = False,
        previous_references: Optional[ReferenceChain] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[ReferenceChain]:
        if "style_directive" in kwargs:
            style_directive = kwargs["style_directive"]
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
