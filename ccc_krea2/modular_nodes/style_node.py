"""CcC Krea2 - Style Image node."""

from typing import Optional
from ccc_krea2.reference_specs import StyleReferenceSpec, ReferenceChain
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


class CcCKrea2StyleImage:
    """Declarative node registering a Style reference into an immutable reference chain."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("REFERENCE_CHAIN",)
    RETURN_NAMES = ("reference_chain",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_image": ("PREPARED_VISION_IMAGE",),
                "style_processing": (["2x2", "4x4", "full"], {"default": "2x2"}),
                "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "indirect_style_transfer": ("BOOLEAN", {"default": True}),
                "style_directive": ("BOOLEAN", {"default": True}),
                "extra_vision_directive": ("STRING", {"default": "", "multiline": True}),
                "vision_slot": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "aliases": ("STRING", {"default": ""}),
            },
            "optional": {
                "reference_chain": ("REFERENCE_CHAIN",),
            }
        }

    def process(
        self,
        prepared_image,
        style_processing="2x2",
        style_fidelity=0.5,
        indirect_style_transfer=True,
        style_directive=True,
        extra_vision_directive="",
        vision_slot=0,
        aliases="",
        reference_chain=None,
        **kwargs
    ):
        chain = reference_chain or kwargs.get("previous_references")
        if chain is None:
            chain = ReferenceChain()

        alias_str = aliases.strip() if aliases.strip() else "style_image"

        spec = StyleReferenceSpec(
            role="style",
            prepared_image=prepared_image,
            requested_vision_slot=parse_vision_slot(vision_slot),
            aliases_template=alias_str,
            parsed_aliases=parse_aliases(alias_str),
            extra_vision_directive=extra_vision_directive,
            style_processing=style_processing,
            style_fidelity=style_fidelity,
            indirect_style_transfer=indirect_style_transfer,
            style_directive=style_directive
        )

        return (chain.append(spec),)
