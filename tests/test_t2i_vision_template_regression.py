"""Regression for T2I image rows being lost under Qwen's automatic image template."""

import torch

from ccc_krea2 import conditioning
from ccc_krea2.constants import VISION_PAD_TOKEN
from ccc_krea2.modular_nodes.edit_reference_types import VisionReferenceChain, VisionReferenceEntry
from ccc_krea2.t2i import _encode_t2i_vision


class TemplateCheckingClip:
    def __init__(self):
        self.seen_template = None
        self.seen_prompt = None

    def tokenize(self, text, *, images, llama_template):
        self.seen_template = llama_template
        self.seen_prompt = text
        assert llama_template == conditioning.KREA2_TEMPLATE
        assert text.count(VISION_PAD_TOKEN) == len(images) == 1
        # Synthetic token stream with system and user turns, like the real
        # Krea2Tokenizer; the image is embedded in the user message.
        tokens = [
            151644, 8948, 198, 123, 151645, 198,
            151644, 872, 198, 151652,
            {"type": "image", "data": images[0]}, 151653,
            123, 151645, 198, 151644, 77091, 198,
        ]
        return {"qwen3vl_4b": [[(t, 1.0) for t in tokens]]}

    def encode_from_tokens_scheduled(self, tokens):
        # The Krea2 system and user opening (9 rows) are stripped.
        # Image is represented by 4 visual rows in this synthetic case.
        return [[torch.ones((1, 21, 8)), {}]]


def test_t2i_vision_uses_krea2_template_and_preserves_visual_spans(monkeypatch):
    monkeypatch.setattr(
        conditioning, "calculate_qwen_rows_from_embedded_image",
        lambda elem, clip, test_mode=False: 4,
    )
    clip = TemplateCheckingClip()
    image = torch.zeros((1, 64, 64, 3))
    chain = VisionReferenceChain().append(
        VisionReferenceEntry(image=image, mode="visual", instruction="")
    )
    conditioning_out, items, prompt, analyzed = _encode_t2i_vision(
        clip=clip, prompt="A fictional seaside city.", references=chain,
    )
    assert clip.seen_template == conditioning.KREA2_TEMPLATE
    assert prompt == VISION_PAD_TOKEN + "A fictional seaside city."
    assert conditioning_out[0][0].shape[1] == 21
    assert items[0]["mode"] == "visual"
    assert analyzed == 0


def test_t2i_vision_semantic_removes_only_image_rows(monkeypatch):
    monkeypatch.setattr(
        conditioning, "calculate_qwen_rows_from_embedded_image",
        lambda elem, clip, test_mode=False: 4,
    )
    clip = TemplateCheckingClip()
    image = torch.zeros((1, 64, 64, 3))
    chain = VisionReferenceChain().append(
        VisionReferenceEntry(image=image, mode="semantic", instruction="")
    )
    conditioning_out, _, _, _ = _encode_t2i_vision(
        clip=clip, prompt="A fictional seaside city.", references=chain,
    )
    assert conditioning_out[0][0].shape[1] == 17
