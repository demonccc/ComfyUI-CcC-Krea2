"""Prompt Creator reference routing tests (no model or GPU required)."""

import torch

from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
    CcCKrea2EditPromptCreator, _collect_analysis_images,
)
from ccc_krea2.modular_nodes.edit_reference_types import ImageReferenceEntry


def _image(v):
    return torch.full((1, 32, 48, 3), float(v))


def test_edit_first_and_image_only_available_with_both_controls_off():
    edit_image, other_image = _image(0.2), _image(0.8)
    edit, _ = CcCKrea2Reference().process(
        edit_image, vae_rope_enable=False, vision_enable=False,
    )
    refs, _ = CcCKrea2Reference().process(
        other_image, vae_rope_enable=False, vision_enable=False,
    )
    images, labels, warnings = _collect_analysis_images(refs, edit)
    assert len(images) == 2
    assert images[0] is edit_image
    assert images[1] is other_image
    assert "Image 1" in labels[0]
    assert "Image 2" in labels[1]
    assert not warnings
    assert isinstance(edit.entries[0], ImageReferenceEntry)


def test_paired_visual_and_vision_analyzed_only_once():
    source = _image(0.5)
    refs, _ = CcCKrea2Reference().process(source)
    images, labels, warnings = _collect_analysis_images(refs, None)
    assert len(images) == 1
    assert images[0] is source
    assert not warnings


def test_vision_only_analysis_uses_original_source():
    source = torch.rand((1, 1200, 1600, 3))
    refs, _ = CcCKrea2Reference().process(
        source, vae_rope_enable=False, vision_enable=True,
        vision_resize=True, vision_max_edge=768,
    )
    images, _, _ = _collect_analysis_images(refs, None)
    assert len(images) == 1
    assert images[0] is source


class _FakeClip:
    def __init__(self):
        self.images = None
        self.prompt = None
        self.system_prompt = None

    def tokenize(self, prompt, **kwargs):
        self.images = kwargs["images"]
        self.prompt = prompt
        self.system_prompt = kwargs.get("system_prompt")
        return {"prompt": prompt}

    def generate(self, tokens, **kwargs):
        return "generated"

    def decode(self, ids):
        return "The woman is standing in a garden."


def test_creator_returns_edit_chain_before_other_references():
    edit_image, other_image = _image(0.2), _image(0.8)
    edit, _ = CcCKrea2Reference().process(
        edit_image, vae_rope_enable=False, vision_enable=False,
    )
    others, _ = CcCKrea2Reference().process(
        other_image, vae_rope_enable=False, vision_enable=True,
    )
    clip = _FakeClip()
    result = CcCKrea2EditPromptCreator().create(
        clip=clip, reference_edit_image=edit, references=others,
        mode="create_from_image_i2i", instructions="Describe the images.",
    )
    assert clip.images[0] is edit_image
    assert clip.images[1] is other_image
    assert result[1].entries == edit.entries + others.entries
    assert "Image 1" in result[2] and "Image 2" in result[2]


def test_enhance_concatenates_preset_user_and_guidelines_without_images():
    from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
        _resolve_preset, KREA2_PROMPT_GUIDELINES,
    )
    clip = _FakeClip()
    image_chain, _ = CcCKrea2Reference().process(_image(0.5))
    result = CcCKrea2EditPromptCreator().create(
        clip=clip, mode="enhance_t2i", instructions="a dog", references=image_chain,
    )
    assert clip.images == []
    assert result[1] is not None and result[1].entries == image_chain.entries
    # The preset is included in the generation prompt once, not duplicated in system_prompt.
    assert clip.prompt.startswith(_resolve_preset("enhance_t2i") + "\n\nINSTRUCTIONS:\na dog")
    assert clip.prompt.endswith(KREA2_PROMPT_GUIDELINES)
    assert clip.system_prompt == ""


def test_other_presets_do_not_receive_shared_guidelines_yet():
    from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
        _build_generation_prompt,
    )
    prompt = _build_generation_prompt(
        mode="none", instructions="a dog", mapping_lines=[],
    )
    assert "KREA2 PROMPT GUIDELINES:" in prompt


def test_analysis_only_source_does_not_steal_edit_image_numbers():
    from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
        _edit_image_slot_map, _remap_created_image_labels,
    )
    src = _image(0.1)
    subject = _image(0.9)
    analysis, _ = CcCKrea2Reference().process(
        src, vae_rope_enable=False, vision_enable=False,
    )
    active, _ = CcCKrea2Reference().process(subject)
    slots = _edit_image_slot_map(active, analysis)
    assert slots == [None, 1]
    prompt, unmatched = _remap_created_image_labels(
        "The woman from Image 2 stands in the setting of Image 1.", slots,
    )
    assert prompt == (
        "The woman from Image 1 stands in the setting of the analyzed source image."
    )
    assert unmatched == [1]


def test_analysis_only_interleaved_reference_is_skipped_without_shifting_edit():
    from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
        _edit_image_slot_map, _remap_created_image_labels,
    )
    first, _ = CcCKrea2Reference().process(_image(0.1))
    passive, _ = CcCKrea2Reference().process(
        _image(0.2), vae_rope_enable=False, vision_enable=False,
        previous_references=first,
    )
    refs, _ = CcCKrea2Reference().process(
        _image(0.3), previous_references=passive,
    )
    assert _edit_image_slot_map(refs, None) == [1, None, 2]
    prompt, unmatched = _remap_created_image_labels(
        "Image 3 and Image 1 in a scene.", [1, None, 2]
    )
    assert prompt == "Image 2 and Image 1 in a scene."
    assert unmatched == []


def test_visual_only_reference_is_not_assigned_a_qwen_image_slot():
    from ccc_krea2.modular_nodes.edit_prompt_creator_node import _edit_image_slot_map
    ref, _ = CcCKrea2Reference().process(
        _image(0.4), vae_rope_enable=True, vision_enable=False,
    )
    assert _edit_image_slot_map(ref, None) == [None]


def test_creator_remaps_qwen_output_against_edit_slots():
    from ccc_krea2.modular_nodes.edit_prompt_creator_node import CcCKrea2EditPromptCreator

    class LabelClip(_FakeClip):
        def decode(self, ids):
            return "The woman of Image 2 is wearing a red coat."

    scene, _ = CcCKrea2Reference().process(
        _image(0.1), vae_rope_enable=False, vision_enable=False,
    )
    woman, _ = CcCKrea2Reference().process(_image(0.2))
    result = CcCKrea2EditPromptCreator().create(
        clip=LabelClip(), references=woman, reference_edit_image=scene,
        mode="create_from_image_i2i", instructions="Use the woman from Image 2.",
    )
    assert result[0] == "The woman of Image 1 is wearing a red coat."
    assert "Analysis Image 1 -> analysis only" in result[2]
    assert "Analysis Image 2 -> Edit Image 1" in result[2]
