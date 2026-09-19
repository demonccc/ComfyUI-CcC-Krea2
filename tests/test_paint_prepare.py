"""Krea2 CcC Paint Prepare mask/canvas contract."""

import torch

from ccc_krea2.modular_nodes.paint_prepare_node import prepare_paint_context


def _base_image(height=9, width=9):
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.4
    image[..., 2] = 0.6
    return image


def _prepare(mask, **overrides):
    params = {
        "expand_left": 0,
        "expand_top": 0,
        "expand_right": 0,
        "expand_bottom": 0,
        "mask_grow": 0,
        "mask_blur_mode": "gaussian_sigma",
        "mask_blur_amount": 0.0,
        "mask_blur_direction": "outside",
    }
    params.update(overrides)
    return prepare_paint_context(_base_image(), mask, **params)


def test_native_canvas_is_not_resized_and_alignment_only_adds_pixels():
    image = _base_image(height=5, width=7)
    mask = torch.zeros((1, 5, 7), dtype=torch.float32)
    context, prepared, _, generated, keep = prepare_paint_context(
        image,
        mask,
        expand_left=3,
        expand_top=2,
        expand_right=4,
        expand_bottom=1,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )

    assert prepared.shape == (1, 16, 16, 3)
    assert context["canvas_width"] == 16
    assert context["canvas_height"] == 16
    assert torch.allclose(prepared[:, 2:7, 3:10], image)
    assert torch.all(generated[:, 2:7, 3:10] == 0)
    assert torch.all(keep[:, 2:7, 3:10] == 1)
    assert context["alignment_right"] == 2
    assert context["alignment_bottom"] == 8


def test_signed_mask_grow_expands_and_shrinks():
    image = _base_image(height=16, width=16)
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, 8, 8] = 1.0

    _, _, _, grown, _ = prepare_paint_context(
        image,
        mask,
        expand_left=0,
        expand_top=0,
        expand_right=0,
        expand_bottom=0,
        mask_grow=2,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )
    assert int((grown > 0.5).sum()) == 25

    wide = torch.zeros((1, 16, 16), dtype=torch.float32)
    wide[:, 6:11, 6:11] = 1.0
    _, _, _, shrunk, _ = prepare_paint_context(
        image,
        wide,
        expand_left=0,
        expand_top=0,
        expand_right=0,
        expand_bottom=0,
        mask_grow=-2,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )
    assert int((shrunk > 0.5).sum()) == 1


def test_blur_amount_zero_means_no_feather():
    mask = torch.zeros((1, 9, 9), dtype=torch.float32)
    mask[:, 3:6, 3:6] = 1.0

    outputs = []
    for direction in ("outside", "inside", "both"):
        _, _, _, generated, _ = _prepare(
            mask,
            mask_blur_mode="gaussian_sigma",
            mask_blur_amount=0.0,
            mask_blur_direction=direction,
        )
        outputs.append(generated)

    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_directional_feather_has_distinct_inside_outside_and_both_semantics():
    mask = torch.zeros((1, 9, 9), dtype=torch.float32)
    mask[:, 3:6, 3:6] = 1.0

    _, _, _, outside, _ = _prepare(
        mask,
        mask_blur_amount=1.0,
        mask_blur_direction="outside",
    )
    _, _, _, inside, _ = _prepare(
        mask,
        mask_blur_amount=1.0,
        mask_blur_direction="inside",
    )
    _, _, _, both, _ = _prepare(
        mask,
        mask_blur_amount=1.0,
        mask_blur_direction="both",
    )

    assert outside[0, 3, 3] == 1.0
    assert outside[0, 2, 4] > 0.0
    assert inside[0, 2, 4] == 0.0
    assert 0.0 < inside[0, 3, 3] < 1.0
    assert both[0, 2, 4] > 0.0
    assert 0.0 < both[0, 3, 3] < 1.0


def test_standard_and_gaussian_feather_are_separate_algorithms():
    mask = torch.zeros((1, 9, 9), dtype=torch.float32)
    mask[:, 3:6, 3:6] = 1.0

    _, _, _, standard, _ = _prepare(
        mask,
        mask_blur_mode="standard",
        mask_blur_amount=2.0,
        mask_blur_direction="both",
    )
    _, _, _, gaussian, _ = _prepare(
        mask,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=2.0,
        mask_blur_direction="both",
    )
    assert not torch.allclose(standard, gaussian)


def test_semantic_reference_neutralizes_generated_pixels():
    image = _base_image()
    image[:, 3:6, 3:6] = 1.0
    mask = torch.zeros((1, 9, 9), dtype=torch.float32)
    mask[:, 3:6, 3:6] = 1.0

    context, prepared, semantic, generated, _ = prepare_paint_context(
        image,
        mask,
        expand_left=0,
        expand_top=0,
        expand_right=0,
        expand_bottom=0,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )

    assert context["source_width"] == 9
    assert prepared[0, 4, 4].mean() == 1.0
    assert semantic[0, 4, 4].mean() < 1.0
    assert generated[0, 4, 4] == 1.0
