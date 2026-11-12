"""Regression tests for the single ordered Prompt Creator reference chain."""
import torch

from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
from ccc_krea2.modular_nodes.edit_prompt_creator_node import (
    CcCKrea2EditPromptCreator, _collect_analysis_images, _edit_image_slot_map,
    _remap_created_image_labels,
)

def image(x):
    return torch.full((1, 32, 48, 3), float(x))

class Clip:
    def __init__(self, response="The subject from Image 2 is in Image 1's scene."):
        self.response = response
        self.images = None
    def tokenize(self, prompt, **kwargs):
        self.images = kwargs["images"]
        return prompt
    def generate(self, tokens, **kwargs):
        return tokens
    def decode(self, tokens):
        return self.response

def test_scene_first_subject_second_in_one_chain():
    scene_image, subject_image = image(0.1), image(0.9)
    scene, _ = CcCKrea2Reference().process(
        scene_image, vae_rope_enable=False, vision_enable=True, vision_mode="none")
    refs, _ = CcCKrea2Reference().process(
        subject_image, vae_rope_enable=True, vision_enable=False,
        previous_references=scene)
    analyzed, labels, warnings = _collect_analysis_images(refs)
    assert len(analyzed) == 2
    assert analyzed[0] is scene_image
    assert analyzed[1] is subject_image
    assert not warnings
    assert _edit_image_slot_map(refs) == [1, None]
    clip = Clip()
    created, passthrough, info, thinking = CcCKrea2EditPromptCreator().create(
        clip=clip, references=refs, mode="create_from_image_i2i",
        instructions="Use the first scene and second subject.")
    assert len(clip.images) == 2
    assert passthrough is refs
    assert "Image 1" in info and "Image 2" in info
    assert thinking == ""

def test_analysis_only_reference_does_not_change_edit_slot():
    scene, _ = CcCKrea2Reference().process(
        image(0.1), vae_rope_enable=False, vision_enable=False)
    refs, _ = CcCKrea2Reference().process(
        image(0.9), previous_references=scene)
    assert _edit_image_slot_map(refs) == [None, 1]
    text, missing = _remap_created_image_labels(
        "Image 2 uses the scene of Image 1.", [None, 1])
    assert "Image 1" in text
    assert missing == [1]

def test_schema_has_only_one_reference_input():
    schema = CcCKrea2EditPromptCreator.INPUT_TYPES()
    assert "references" in schema["optional"]
    assert "reference_edit_image" not in schema["optional"]
