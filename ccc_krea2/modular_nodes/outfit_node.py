"""CcC Krea2 - Outfit Image node."""

from typing import Optional
from ccc_krea2.reference_specs import OutfitReferenceSpec, ReferenceChain
from ccc_krea2.reference_slots import parse_aliases


def parse_vision_slot(val) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, str):
        v_str = val.strip()
        if not v_str or v_str.lower() == "auto":
            return None
    try:
        res = int(val)
        return res if res > 0 else None
    except ValueError:
        return None


class CcCKrea2OutfitImage:
    """Declarative node registering an Outfit reference into an immutable reference chain."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("REFERENCE_CHAIN",)
    RETURN_NAMES = ("reference_chain",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_image": ("PREPARED_VISION_IMAGE",),
                "visual_fit_mode": (["auto", "fit", "crop"], {"default": "auto"}),
                "attention_boost": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05}),
                "masked_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05}),
                "outfit_anchor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Implementation: Vision directive only"}),
                "extra_vision_directive": ("STRING", {"default": "", "multiline": True}),
                "vision_slot": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "aliases": ("STRING", {"default": ""}),
            },
            "optional": {
                "attention_mask": ("MASK",),
                "reference_chain": ("REFERENCE_CHAIN",),
            }
        }

    def process(
        self,
        prepared_image,
        visual_fit_mode="auto",
        attention_boost=1.0,
        masked_attention_boost=1.0,
        outfit_anchor=0.0,
        extra_vision_directive="",
        vision_slot=0,
        aliases="",
        attention_mask=None,
        reference_chain=None,
        **kwargs
    ):
        chain = reference_chain or kwargs.get("previous_references")
        if chain is None:
            chain = ReferenceChain()

        # Handle legacy serialized fit modes
        if visual_fit_mode == "exact":
            visual_fit_mode = "auto"
        elif visual_fit_mode == "stretch":
            visual_fit_mode = "crop"

        alias_str = aliases.strip() if aliases.strip() else "outfit_image, Image {slot}"

        spec = OutfitReferenceSpec(
            role="outfit",
            prepared_image=prepared_image,
            requested_vision_slot=parse_vision_slot(vision_slot),
            aliases_template=alias_str,
            parsed_aliases=parse_aliases(alias_str),
            extra_vision_directive=extra_vision_directive,
            attention_boost=attention_boost,
            outfit_anchor=outfit_anchor,
            attention_mask=attention_mask,
            masked_attention_boost=masked_attention_boost,
            visual_fit_mode=visual_fit_mode
        )

        return (chain.append(spec),)
