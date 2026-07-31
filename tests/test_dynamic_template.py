"""CPU-safe unit tests for dynamic Qwen3-VL llama template generation."""

from ccc_krea2.conditioning import build_krea2_qwen_template
from ccc_krea2.constants import VISION_PAD_TOKEN


def test_dynamic_template_single_ref():
    tmpl = build_krea2_qwen_template(num_images=1)
    assert tmpl.count(VISION_PAD_TOKEN) == 1
    assert "{prompt}" in tmpl
    assert "<|im_start|>system" in tmpl


def test_dynamic_template_multi_ref():
    tmpl_dual = build_krea2_qwen_template(num_images=2)
    assert tmpl_dual.count(VISION_PAD_TOKEN) == 2

    tmpl_tri = build_krea2_qwen_template(num_images=3)
    assert tmpl_tri.count(VISION_PAD_TOKEN) == 3
