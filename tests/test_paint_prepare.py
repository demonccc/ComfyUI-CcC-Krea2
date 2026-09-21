"""Krea2 CcC Paint Prepare mask/latent contract."""

import torch

from ccc_krea2.modular_nodes.paint_prepare_node import prepare_paint_context


class FakeVAE:
    def encode(self, image):
        batch, height, width, _ = image.shape
        return torch.zeros((batch, 16, height // 8, width // 8), dtype=image.dtype)


def _geometry(mask=None, height=16, width=16):
    image = torch.zeros((1, height, width, 3), dtype=torch.float32)
    image[..., 0] = 0.2
    image[..., 1] = 0.4
    image[..., 2] = 0.6
    if mask is None:
        mask = torch.zeros((1, height, width), dtype=torch.float32)
    return {
        "mode": "pad",
        "working_image": image,
        "working_mask": mask,
    }


def _prepare(mask=None, **overrides):
    params = {
        "mask_grow": 0,
        "mask_blur_mode": "gaussian_sigma",
        "mask_blur_amount": 0.0,
        "mask_blur_direction": "outside",
    }
    params.update(overrides)
    return prepare_paint_context(FakeVAE(), _geometry(mask), **params)


def test_prepare_creates_sampling_latent_and_token_aligned_noise_mask():
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, 4:12, 4:12] = 1.0
    context, latent, prepared, semantic, generated, keep = _prepare(mask)

    assert prepared.shape == (1, 16, 16, 3)
    assert latent["samples"].shape == (1, 16, 2, 2)
    assert latent["noise_mask"].shape == (1, 1, 2, 2)
    assert context["reference_latent"].shape == (1, 16, 2, 2)
    assert semantic.shape[-1] == 3
    assert torch.allclose(keep, 1.0 - generated)


def test_signed_mask_grow_expands_and_shrinks():
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, 8, 8] = 1.0
    _, _, _, _, grown, _ = _prepare(mask, mask_grow=2)
    assert int((grown > 0.5).sum()) == 25

    wide = torch.zeros((1, 16, 16), dtype=torch.float32)
    wide[:, 6:11, 6:11] = 1.0
    _, _, _, _, shrunk, _ = _prepare(wide, mask_grow=-2)
    assert int((shrunk > 0.5).sum()) == 1


def test_blur_amount_zero_means_no_feather():
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, 5:11, 5:11] = 1.0
    outputs = []
    for direction in ("outside", "inside", "both"):
        _, _, _, _, generated, _ = _prepare(
            mask,
            mask_blur_mode="gaussian_sigma",
            mask_blur_amount=0.0,
            mask_blur_direction=direction,
        )
        outputs.append(generated)
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_directional_feather_has_distinct_semantics():
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, 5:11, 5:11] = 1.0

    _, _, _, _, outside, _ = _prepare(mask, mask_blur_amount=1.0, mask_blur_direction="outside")
    _, _, _, _, inside, _ = _prepare(mask, mask_blur_amount=1.0, mask_blur_direction="inside")
    _, _, _, _, both, _ = _prepare(mask, mask_blur_amount=1.0, mask_blur_direction="both")

    assert outside[0, 5, 5] == 1.0
    assert outside[0, 4, 7] > 0.0
    assert inside[0, 4, 7] == 0.0
    assert 0.0 < inside[0, 5, 5] < 1.0
    assert both[0, 4, 7] > 0.0


def test_semantic_reference_neutralizes_generated_pixels():
    geometry = _geometry()
    geometry["working_image"][:, 5:11, 5:11] = 1.0
    geometry["working_mask"][:, 5:11, 5:11] = 1.0
    context, _, prepared, semantic, generated, _ = prepare_paint_context(
        FakeVAE(),
        geometry,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )
    assert prepared[0, 7, 7].mean() == 1.0
    assert semantic[0, 7, 7].mean() < 1.0
    assert generated[0, 7, 7] == 1.0
    assert context["geometry_mode"] == "pad"
