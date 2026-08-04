"""Unit tests for Qwen Vision image prep logic and node."""

import torch
from ccc_krea2.vision_prep import (
    prepare_vision_image,
    format_vision_info
)
from ccc_krea2.modular_nodes.vision_prep_node import CcCKrea2QwenVisionImagePrep


def test_qwen_vision_prep_native_mode():
    img = torch.rand(1, 1080, 1920, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")

    # Original image must be untouched
    assert torch.equal(prep.original_image, img)
    assert prep.prep_spec.mode == "native"
    assert prep.vision_image.shape[-1] == 3
    # Check alignment (factor 32 for Qwen3-VL)
    _, vh, vw, _ = prep.vision_image.shape
    assert vh % 32 == 0
    assert vw % 32 == 0


def test_qwen_vision_prep_adaptive_mode():
    img = torch.rand(1, 4000, 3000, 3)
    prep = prepare_vision_image(
        image=img,
        clip=None,
        mode="adaptive",
        min_mp=0.0,
        max_mp=1.0
    )
    th, tw = prep.debug_metadata["target_hw"]
    assert (th * tw) <= 1_050_000


def test_qwen_vision_prep_fixed_mode():
    img = torch.rand(1, 1024, 1024, 3)
    prep = prepare_vision_image(
        image=img,
        clip=None,
        mode="fixed",
        fixed_mp=0.5,
        downscale_method="area"
    )
    th, tw = prep.debug_metadata["target_hw"]
    assert abs((th * tw) - 500_000) < 100_000


def test_qwen_vision_prep_info_formatting():
    img = torch.rand(1, 512, 512, 3)
    prep = prepare_vision_image(image=img, clip=None, mode="native")
    info = format_vision_info(prep)
    assert "Vision Encoder:" in info
    assert "Mode: native" in info
    assert "Prepared Size:" in info


def test_qwen_vision_prep_node_execution():
    node = CcCKrea2QwenVisionImagePrep()
    img = torch.rand(1, 800, 600, 3)
    prep_obj, vis_img, info_str = node.process(
        clip=None,
        image=img,
        vision_preparation_mode="native"
    )
    assert vis_img.shape[0] == 1
    assert "Prepared Size:" in info_str
