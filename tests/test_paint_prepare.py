"""Krea2 CcC Paint Prepare mask/latent contract."""

import torch

from ccc_krea2.modular_nodes.paint_prepare_node import CcCKrea2PaintPrepare, prepare_paint_context, _token_aligned_soft_noise_mask, _grow_or_shrink


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


def test_sampler_vae_neutralizes_explicit_outpaint_fill_but_preserves_source():
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
    # Expansion pixel values in the sampling VAE input must not reproduce
    # edge/reflect/white padding. Source pixels retain their exact values.
    torch.testing.assert_close(vae.inputs[1][:, :, :24], geometry["working_image"][:, :, :24])
    torch.testing.assert_close(vae.inputs[1][:, :, 24:], geometry["working_image"][:, :, :24].mean(dim=(1, 2), keepdim=True).expand(-1, 32, 8, -1))
    assert not torch.allclose(vae.inputs[1][:, :, 24:], geometry["working_image"][:, :, 24:])


def test_outpaint_sampling_latent_is_independent_of_padding_fill_pixels():
    class TrackingVAE(FakeVAE):
        def __init__(self):
            self.inputs = []

        def encode(self, image):
            self.inputs.append(image.clone())
            return super().encode(image)

    expansion = torch.zeros((1, 32, 32))
    expansion[:, :, 24:] = 1.0
    sampler_inputs = []
    for padding_value in (0.0, 0.5, 1.0):
        geometry = _geometry(height=32, width=32)
        geometry["working_image"][:, :16, :24, :] = 0.1
        geometry["working_image"][:, 16:, :24, :] = 0.7
        geometry["working_image"][:, :, 24:, :] = padding_value
        geometry["working_mask"] = expansion.clone()
        geometry["working_expansion_mask"] = expansion.clone()
        geometry["working_user_mask"] = torch.zeros_like(expansion)
        vae = TrackingVAE()
        _, _, prepared, _, _, _ = prepare_paint_context(
            vae, geometry, mask_grow=0, mask_blur_mode="gaussian_sigma",
            mask_blur_amount=0.0, mask_blur_direction="outside",
        )
        torch.testing.assert_close(vae.inputs[1][:, :, :24], prepared[:, :, :24])
        sampler_inputs.append(vae.inputs[1])

    for candidate in sampler_inputs[1:]:
        torch.testing.assert_close(candidate, sampler_inputs[0])


def _outpaint_denoise_mask(*, grow=0, blur=0.0, mode="gaussian_sigma", direction="outside"):
    geometry = _geometry(height=64, width=64)
    expansion = torch.zeros_like(geometry["working_mask"])
    expansion[:, :, 32:] = 1.0
    geometry["working_mask"] = expansion.clone()
    geometry["working_expansion_mask"] = expansion
    geometry["working_user_mask"] = torch.zeros_like(expansion)
    _, latent, _, _, _, _ = prepare_paint_context(
        FakeVAE(), geometry, mask_grow=grow,
        mask_blur_mode=mode, mask_blur_amount=blur,
        mask_blur_direction=direction,
    )
    return latent["noise_mask"]


def test_outpaint_blur_zero_is_binary_and_matches_token_aligned_grown_mask():
    for grow in (0, 8, 16):
        mask = _outpaint_denoise_mask(grow=grow, blur=0.0)
        assert mask.shape == (1, 1, 8, 8)
        assert torch.all((mask == 0) | (mask == 1))

        expansion = torch.zeros((1, 64, 64))
        expansion[:, :, 32:] = 1.0
        expected = _token_aligned_soft_noise_mask(
            _grow_or_shrink(expansion, grow), latent_h=8, latent_w=8, patch_size=2
        )
        torch.testing.assert_close(mask, expected)


def test_outpaint_explicit_blur_produces_soft_mask_and_honors_direction():
    no_blur = _outpaint_denoise_mask(blur=0.0)
    outside = _outpaint_denoise_mask(blur=4.0, direction="outside")
    inside = _outpaint_denoise_mask(blur=4.0, direction="inside")
    both = _outpaint_denoise_mask(blur=4.0, direction="both")
    assert torch.any((outside > 0.0) & (outside < 1.0))
    assert torch.any((inside > 0.0) & (inside < 1.0))
    assert torch.any((both > 0.0) & (both < 1.0))
    assert not torch.equal(outside, no_blur)
    assert not torch.equal(inside, no_blur)
    assert torch.all(outside >= both)
    assert torch.all(both >= inside)
    assert not torch.equal(_outpaint_denoise_mask(blur=4.0, mode="standard"), outside)


def test_pure_inpaint_noise_mask_remains_binary_even_with_blur():
    in_geometry = _geometry(height=64, width=64)
    in_geometry["working_mask"][:, 20:44, 20:44] = 1.0
    _, in_latent, _, _, _, _ = prepare_paint_context(
        FakeVAE(), in_geometry, mask_grow=0,
        mask_blur_mode="gaussian_sigma", mask_blur_amount=12.0,
        mask_blur_direction="both",
    )
    assert torch.all((in_latent["noise_mask"] == 0) | (in_latent["noise_mask"] == 1))


def test_temporary_reflect_padding_is_neutralized_only_for_outpaint():
    class TrackingVAE(FakeVAE):
        def __init__(self):
            self.inputs = []

        def encode(self, image):
            self.inputs.append(image.clone())
            return super().encode(image)

    geometry = _geometry(height=64, width=64)
    geometry["working_image"][:, :, 48:, :] = 0.95
    expansion = torch.zeros_like(geometry["working_mask"])
    expansion[:, :, 40:48] = 1.0
    temporary = torch.zeros_like(expansion)
    temporary[:, :, 48:] = 1.0
    geometry["working_mask"] = expansion.clone()
    geometry["working_expansion_mask"] = expansion
    geometry["working_krea_padding_mask"] = temporary
    geometry["working_user_mask"] = torch.zeros_like(expansion)
    vae = TrackingVAE()
    context, latent, prepared, semantic, _, _ = prepare_paint_context(
        vae, geometry, mask_grow=0,
        mask_blur_mode="gaussian_sigma", mask_blur_amount=0.0,
        mask_blur_direction="outside",
    )
    # Neither of the VAE inputs can contain reflected content in the
    # disposable geometry padding.
    assert not torch.allclose(vae.inputs[0][:, :, 48:], prepared[:, :, 48:])
    assert not torch.allclose(vae.inputs[1][:, :, 48:], prepared[:, :, 48:])
    torch.testing.assert_close(vae.inputs[1][:, :, :40], prepared[:, :, :40])
    assert torch.all(context["semantic_neutralize_mask"][:, :, 48:] == 1)


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


def test_separable_mask_grow_matches_square_max_pool_for_soft_and_binary_masks():
    import torch.nn.functional as F
    torch.manual_seed(1042)
    for height, width in ((9, 11), (32, 48)):
        soft = torch.rand((2, height, width))
        binary = (soft > 0.5).float()
        for mask in (soft, binary):
            for amount in (-16, -8, -3, -1, 0, 1, 3, 8, 16):
                if amount == 0:
                    expected = mask
                else:
                    radius = abs(amount)
                    source = mask if amount > 0 else 1.0 - mask
                    pooled = F.max_pool2d(
                        source.unsqueeze(1),
                        kernel_size=2 * radius + 1, stride=1, padding=radius,
                    ).squeeze(1)
                    expected = pooled if amount > 0 else (1.0 - pooled).clamp(0.0, 1.0)
                torch.testing.assert_close(
                    _grow_or_shrink(mask, amount),
                    expected,
                    rtol=0,
                    atol=0,
                )


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


def test_outpaint_reference_hides_fill_but_preserves_original_source_colors():
    geometry = _geometry()
    # Source is varied, while edge/reflect/white padding could be quite different.
    geometry["working_image"][:, :8, :12, :] = 0.2
    geometry["working_image"][:, 8:, :12, :] = 0.8
    geometry["working_image"][:, :, 12:, :] = 0.95
    geometry["working_mask"][:, :, 12:] = 1.0
    expansion = torch.zeros_like(geometry["working_mask"])
    expansion[:, :, 12:] = 1.0
    geometry["working_expansion_mask"] = expansion
    geometry["working_user_mask"] = torch.zeros_like(expansion)

    class TrackingVAE(FakeVAE):
        def __init__(self):
            self.inputs = []

        def encode(self, image):
            self.inputs.append(image.clone())
            return super().encode(image)

    vae = TrackingVAE()
    context, _, prepared, semantic, generated, _ = prepare_paint_context(
        vae, geometry, mask_grow=2, mask_blur_mode="gaussian_sigma",
        mask_blur_amount=1.0, mask_blur_direction="outside",
    )

    assert torch.all(generated[:, :, 12:] == 1.0)
    assert generated[0, 4, 11] > 0  # overlap is still generated
    torch.testing.assert_close(semantic[:, :, :12, :], prepared[:, :, :12, :])
    # Mean of the actual source is ~0.5, not the 0.95 padding value.
    torch.testing.assert_close(semantic[0, 4, 14], torch.full((3,), 0.5))
    assert torch.all(context["semantic_neutralize_mask"][:, :, 12:] == 1.0)
    torch.testing.assert_close(vae.inputs[0], semantic)
    # Sampling latent preserves source pixels, but replaces only expansion fill
    # with the neutral source-derived color instead of using reflect/edge pixels.
    torch.testing.assert_close(vae.inputs[1][:, :, :12], prepared[:, :, :12])
    torch.testing.assert_close(vae.inputs[1][0, 4, 14], torch.full((3,), 0.5))
    assert not torch.allclose(vae.inputs[1][:, :, 12:], prepared[:, :, 12:])


def test_combined_outpaint_preserves_original_overlap_and_neutralizes_user_inpaint():
    geometry = _geometry(height=32, width=32)
    geometry["working_image"][:, :, :] = 0.25
    geometry["working_image"][:, 10:15, 10:15] = 1.0
    expansion = torch.zeros_like(geometry["working_mask"])
    expansion[:, :, 24:] = 1.0
    user_mask = torch.zeros_like(expansion)
    user_mask[:, 10:15, 10:15] = 1.0
    geometry["working_mask"] = torch.maximum(user_mask, expansion)
    geometry["working_expansion_mask"] = expansion
    geometry["working_user_mask"] = user_mask

    context, _, prepared, semantic, generated, _ = prepare_paint_context(
        FakeVAE(), geometry, mask_grow=2,
        mask_blur_mode="gaussian_sigma", mask_blur_amount=1.0,
        mask_blur_direction="outside",
    )

    assert generated[0, 20, 23] > 0.0
    torch.testing.assert_close(semantic[0, 20, 23], prepared[0, 20, 23])
    assert semantic[0, 12, 12].mean() < prepared[0, 12, 12].mean()
    assert context["semantic_neutralize_mask"][0, 12, 12] == 1.0
    assert context["semantic_neutralize_mask"][0, 20, 26] == 1.0


def test_pure_inpaint_semantic_reference_remains_unchanged_by_outpaint_logic():
    geometry = _geometry(height=32, width=32)
    geometry["working_image"][:, 10:20, 10:20] = 0.95
    geometry["working_mask"][:, 10:20, 10:20] = 1.0
    context, _, prepared, semantic, generated, keep = prepare_paint_context(
        FakeVAE(), geometry, mask_grow=1,
        mask_blur_mode="gaussian_sigma", mask_blur_amount=1.0,
        mask_blur_direction="both",
    )
    torch.testing.assert_close(context["semantic_neutralize_mask"], generated)
    assert semantic[0, 15, 15].mean() < prepared[0, 15, 15].mean()
    torch.testing.assert_close(keep, 1 - generated)


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
