"""Krea2 CcC Paint Prepare mask/latent contract."""

import torch

from ccc_krea2.modular_nodes.paint_prepare_node import CcCKrea2PaintPrepare, prepare_paint_context, _token_aligned_soft_noise_mask


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
        "fill_holes": False,
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


def test_sampler_vae_encodes_original_known_canvas_and_masked_semantic_reference():
    class TrackingVAE(FakeVAE):
        def __init__(self):
            self.inputs = []

        def encode(self, image):
            self.inputs.append(image.clone())
            return super().encode(image)

    geometry = _geometry(height=32, width=32)
    geometry["working_image"][:, 10:22, 10:22, :] = 1.0
    geometry["working_mask"][:, 10:22, 10:22] = 1.0
    vae = TrackingVAE()

    context, latent, prepared, semantic, generated, keep = prepare_paint_context(
        vae, geometry,
        mask_grow=0, mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0, mask_blur_direction="outside",
    )

    # AnyPaint: semantic reference excludes the selected subject, while the
    # full-resolution sampling latent encodes the known image. noise_mask
    # determines which tokens get regenerated during denoising.
    assert len(vae.inputs) == 2
    torch.testing.assert_close(vae.inputs[1], prepared)
    assert vae.inputs[0][0, 15, 15].mean() < 0.9
    torch.testing.assert_close(vae.inputs[0][0, 15, 15], semantic[0, 15, 15])
    assert latent["noise_mask"].max() > 0
    assert generated[0, 15, 15] == 1
    assert context["semantic_reference"] is semantic


def test_anypaint_token_alignment_regenerates_entire_partially_masked_patch():
    # Four latent cells are grouped into one 2x2 DiT patch.
    # Averaging the mask would leave residual known content; a single
    # selected cell must instead regenerate all four.
    generated = torch.zeros((1, 8, 8))
    generated[:, 2:4, 2:4] = 1.0
    noise = _token_aligned_soft_noise_mask(generated, latent_h=4, latent_w=4, patch_size=2)
    assert noise.shape == (1, 1, 4, 4)
    torch.testing.assert_close(noise[0, 0, :2, :2], torch.ones((2, 2)))
    torch.testing.assert_close(noise[0, 0, 2:, 2:], torch.zeros((2, 2)))
    assert torch.all((noise == 0) | (noise == 1))


def test_anypaint_token_alignment_never_partially_preserves_feathered_patch():
    generated = torch.zeros((1, 8, 8))
    generated[:, 1:3, 1:3] = 0.8
    noise = _token_aligned_soft_noise_mask(generated, latent_h=4, latent_w=4, patch_size=2)
    assert torch.all((noise == 0) | (noise == 1))


def test_sampler_vae_preserves_explicit_outpaint_fill_context():
    class TrackingVAE(FakeVAE):
        def __init__(self):
            self.inputs = []

        def encode(self, image):
            self.inputs.append(image.clone())
            return super().encode(image)

    geometry = _geometry(height=32, width=32)
    geometry["working_image"][:, :, 24:, :] = 0.9
    geometry["working_mask"][:, :, 24:] = 1.0
    expansion = torch.zeros_like(geometry["working_mask"])
    expansion[:, :, 24:] = 1.0
    geometry["working_expansion_mask"] = expansion

    vae = TrackingVAE()
    prepare_paint_context(
        vae, geometry, mask_grow=0,
        mask_blur_mode="gaussian_sigma", mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )
    torch.testing.assert_close(vae.inputs[1][:, :, 24:], geometry["working_image"][:, :, 24:])


def test_fill_holes_fills_only_enclosed_mask_interiors():
    ring = torch.zeros((1, 16, 16), dtype=torch.float32)
    ring[:, 3:13, 3] = 1.0
    ring[:, 3:13, 12] = 1.0
    ring[:, 3, 3:13] = 1.0
    ring[:, 12, 3:13] = 1.0

    _, _, _, _, original, _ = _prepare(ring, fill_holes=False)
    _, _, _, _, filled, _ = _prepare(ring, fill_holes=True)

    assert original[0, 8, 8] == 0.0
    assert filled[0, 8, 8] == 1.0
    assert filled[0, 0, 0] == 0.0


def test_fill_holes_does_not_close_an_open_boundary():
    open_ring = torch.zeros((1, 16, 16), dtype=torch.float32)
    open_ring[:, 3:13, 3] = 1.0
    open_ring[:, 3:13, 12] = 1.0
    open_ring[:, 12, 3:13] = 1.0
    open_ring[:, 3, 3:7] = 1.0
    open_ring[:, 3, 9:13] = 1.0

    _, _, _, _, filled, _ = _prepare(open_ring, fill_holes=True)
    assert filled[0, 8, 8] == 0.0


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


def test_semantic_reference_preserves_explicit_outpaint_context_but_neutralizes_manual_mask():
    geometry = _geometry()
    geometry["working_image"][:, :, 12:, :] = 0.95
    geometry["working_image"][:, 5:9, 5:9, :] = 1.0

    geometry["working_mask"][:, :, 12:] = 1.0
    geometry["working_mask"][:, 5:9, 5:9] = 1.0

    expansion = torch.zeros_like(geometry["working_mask"])
    expansion[:, :, 12:] = 1.0
    geometry["working_expansion_mask"] = expansion

    context, _, prepared, semantic, generated, _ = prepare_paint_context(
        FakeVAE(),
        geometry,
        fill_holes=False,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )

    # The explicit outpaint margin remains fully generable.
    assert torch.all(generated[:, :, 12:] == 1.0)
    # But its edge/reflect/etc. pixels remain visible in the semantic reference.
    assert torch.allclose(semantic[:, :, 12:, :], prepared[:, :, 12:, :])
    # A normal user mask is still neutralized.
    assert semantic[0, 6, 6].mean() < prepared[0, 6, 6].mean()
    assert torch.equal(context["expansion_context_mask"], expansion)
    assert torch.all(context["semantic_neutralize_mask"][:, :, 12:] == 0.0)


def test_semantic_reference_neutralizes_generated_pixels():
    geometry = _geometry()
    geometry["working_image"][:, 5:11, 5:11] = 1.0
    geometry["working_mask"][:, 5:11, 5:11] = 1.0
    context, _, prepared, semantic, generated, _ = prepare_paint_context(
        FakeVAE(),
        geometry,
        fill_holes=False,
        mask_grow=0,
        mask_blur_mode="gaussian_sigma",
        mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )
    assert prepared[0, 7, 7].mean() == 1.0
    assert semantic[0, 7, 7].mean() < 1.0
    assert generated[0, 7, 7] == 1.0
    assert context["geometry_mode"] == "pad"


def test_public_paint_prepare_owns_geometry_controls_and_restore_context_output():
    schema = CcCKrea2PaintPrepare.INPUT_TYPES()
    required = schema["required"]
    optional = schema["optional"]

    assert "image" in required
    assert "vae" in required
    assert "geometry_mode" not in required
    assert required["padding_fill"][0] == ("edge", "reflect", "neutral", "white")
    assert "expand_left" in required
    assert "fill_holes" in required
    assert "mask_grow" in required
    assert "mask" in optional
    assert CcCKrea2PaintPrepare.RETURN_NAMES == (
        "paint_context",
        "latent",
        "generated_mask",
        "paint_geometry",
        "prepared_image",
        "paint_prepare_info",
    )
    assert len(CcCKrea2PaintPrepare.RETURN_TYPES) == 6
    assert "semantic_reference" not in CcCKrea2PaintPrepare.RETURN_NAMES
    assert "keep_mask" not in CcCKrea2PaintPrepare.RETURN_NAMES
