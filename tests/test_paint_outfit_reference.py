"""Optional second outfit image must not alter ordinary single-image Paint."""

import torch

from ccc_krea2.modular_nodes.paint_node import (
    CcCKrea2Paint,
    _encode_qwen_with_reference,
)


class _RecordingClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"prompt": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


def test_paint_outfit_input_is_optional():
    sockets = CcCKrea2Paint.INPUT_TYPES()["optional"]
    assert "outfit_reference" in sockets
    assert sockets["outfit_reference"][0] == "IMAGE"


def test_paint_single_image_conditioning_is_unchanged():
    clip = _RecordingClip()
    image = torch.zeros((1, 32, 32, 3))
    _encode_qwen_with_reference(clip, "Keep the scene.", image)
    prompt, kwargs = clip.calls[-1]
    assert prompt.count("<|image_pad|>") == 1
    assert "Picture 2" not in prompt
    assert len(kwargs["images"]) == 1


def test_paint_outfit_image_is_second_qwen_visual_reference():
    clip = _RecordingClip()
    image = torch.zeros((1, 32, 32, 3))
    outfit = torch.ones((1, 48, 48, 3))
    _encode_qwen_with_reference(clip, "Copy the outfit from Picture 2.", image, outfit)
    prompt, kwargs = clip.calls[-1]
    assert prompt.count("<|image_pad|>") == 2
    assert "Picture 2" in prompt
    assert len(kwargs["images"]) == 2
    torch.testing.assert_close(kwargs["images"][0], image)
    torch.testing.assert_close(kwargs["images"][1], outfit)
