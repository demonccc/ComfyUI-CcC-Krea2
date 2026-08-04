"""CcC Krea2 - Style Image node."""

from ccc_krea2.reference_specs import StyleReferenceSpec, ReferenceChain
from ccc_krea2.reference_slots import parse_aliases
from ccc_krea2.modular_nodes.subject_node import parse_vision_slot


class CcCKrea2StyleImage:
    """Declarative node registering a Moodboard Style reference into an immutable reference chain."""

    CATEGORY = "CcC/Krea2"
    RETURN_TYPES = ("CCC_KREA2_REFERENCE_CHAIN",)
    RETURN_NAMES = ("references",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepared_image": ("CCC_KREA2_PREPARED_IMAGE",),
                "vision_slot": ("STRING", {"default": "auto"}),
                "reference_aliases": ("STRING", {"default": "style_image"}),
                "extra_vision_directive": ("STRING", {"default": "", "multiline": True}),
                "style_fidelity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "style_reference_processing": (["full", "2x2", "4x4"], {"default": "2x2"}),
                "indirect_style_transfer": ("BOOLEAN", {"default": True}),
                "style_directive": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "previous_references": ("CCC_KREA2_REFERENCE_CHAIN",),
            }
        }

    def process(
        self,
        prepared_image,
        vision_slot="auto",
        reference_aliases="style_image",
        extra_vision_directive="",
        style_fidelity=0.5,
        style_reference_processing="2x2",
        indirect_style_transfer=True,
        style_directive=True,
        previous_references=None
    ):
        chain = previous_references if previous_references is not None else ReferenceChain()

        spec = StyleReferenceSpec(
            role="style",
            prepared_image=prepared_image,
            requested_vision_slot=parse_vision_slot(vision_slot),
            aliases_template=reference_aliases,
            parsed_aliases=parse_aliases(reference_aliases),
            extra_vision_directive=extra_vision_directive,
            style_fidelity=style_fidelity,
            style_processing=style_reference_processing,
            indirect_style_transfer=indirect_style_transfer,
            style_directive=style_directive
        )

        return (chain.append(spec),)
