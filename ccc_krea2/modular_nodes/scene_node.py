"""CcC Krea2 - Scene Image node."""

from ccc_krea2.reference_specs import SceneReferenceSpec, ReferenceChain
from ccc_krea2.reference_slots import parse_aliases
from ccc_krea2.modular_nodes.subject_node import parse_vision_slot


class CcCKrea2SceneImage:
    """Declarative node registering a Scene reference into an immutable reference chain."""

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
                "reference_aliases": ("STRING", {"default": "scene_image, Image {slot}"}),
                "extra_vision_directive": ("STRING", {"default": "", "multiline": True}),
                "scene_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "scene_anchor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "masked_attention_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "masked_region_anchor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "visual_reference_fit": (["auto", "fit", "crop"], {"default": "auto"}),
            },
            "optional": {
                "attention_mask": ("MASK",),
                "previous_references": ("CCC_KREA2_REFERENCE_CHAIN",),
            }
        }

    def process(
        self,
        prepared_image,
        vision_slot="auto",
        reference_aliases="scene_image, Image {slot}",
        extra_vision_directive="",
        scene_attention_boost=1.0,
        scene_anchor=0.0,
        masked_attention_boost=1.0,
        masked_region_anchor=1.0,
        visual_reference_fit="auto",
        attention_mask=None,
        previous_references=None
    ):
        chain = previous_references if previous_references is not None else ReferenceChain()

        spec = SceneReferenceSpec(
            role="scene",
            prepared_image=prepared_image,
            requested_vision_slot=parse_vision_slot(vision_slot),
            aliases_template=reference_aliases,
            parsed_aliases=parse_aliases(reference_aliases),
            extra_vision_directive=extra_vision_directive,
            attention_boost=scene_attention_boost,
            scene_anchor=scene_anchor,
            attention_mask=attention_mask,
            masked_attention_boost=masked_attention_boost,
            masked_region_anchor=masked_region_anchor,
            visual_fit_mode=visual_reference_fit
        )

        return (chain.append(spec),)
