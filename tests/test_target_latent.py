"""Unit tests for Target Latent node and resolution logic."""

import torch
import pytest
from unittest.mock import MagicMock
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.target_latent import (
    calculate_subject_aware_scene_geometry,
    create_target_latent,
    normalize_vae_output,
    resolve_target_geometry,
)
from ccc_krea2.modular_nodes.target_latent_node import CcCKrea2TargetLatent


def test_target_geometry_fixed():
    th, tw, geom_src, active_mp, src_dims, warnings = resolve_target_geometry(
        target_geometry="fixed", fixed_mp=1.0, fixed_aspect_ratio="16:9"
    )
    assert th % 16 == 0
    assert tw % 16 == 0
    assert abs((tw / float(th)) - (16.0 / 9.0)) < 0.1
    assert len(warnings) == 0


def test_target_geometry_favor_subject():
    img = torch.rand(1, 1200, 800, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    th, tw, geom_src, active_mp, src_dims, warnings = resolve_target_geometry(
        target_geometry="favor_subject", subject_image=subj_prep, maximum_mp=2.0
    )
    assert th % 16 == 0
    assert tw % 16 == 0
    assert abs((tw / float(th)) - (800.0 / 1200.0)) < 0.1


def test_target_latent_empty_content():
    mock_vae = MagicMock()
    lat_dict, info = create_target_latent(
        vae=mock_vae, target_latent_content="empty", target_geometry="fixed", fixed_mp=1.0, fixed_aspect_ratio="1:1"
    )
    assert "samples" in lat_dict
    assert lat_dict["samples"].shape == (1, 16, 124, 124)
    assert "Latent Content: empty" in info


def test_target_latent_missing_vae_raises():
    img = torch.rand(1, 512, 512, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    with pytest.raises(ValueError, match="VAE is required"):
        create_target_latent(vae=None, target_latent_content="subject", subject_image=subj_prep)


def test_target_latent_empty_without_vae():
    lat_dict, info = create_target_latent(
        vae=None, target_latent_content="empty", target_geometry="fixed", fixed_mp=1.0, fixed_aspect_ratio="1:1"
    )
    assert "samples" in lat_dict
    assert lat_dict["samples"].shape == (1, 16, 124, 124)
    assert "VAE Encode Applied: no" in info


def test_target_latent_subject_content():
    img = torch.rand(1, 600, 400, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="subject",
        subject_image=subj_prep,
        target_geometry="favor_subject",
        maximum_mp=1.0,
        batch_size=2,
    )
    assert lat_dict["samples"].shape == (2, 16, 64, 64)
    assert "Latent Content: subject" in info
    assert "VAE Encode Applied: yes" in info
    mock_vae.encode.assert_called_once()


def test_target_latent_scene_content_missing_raises():
    with pytest.raises(ValueError, match="Scene image is required"):
        create_target_latent(vae=MagicMock(), target_latent_content="scene", scene_image=None)


def test_target_latent_node_execution():
    node = CcCKrea2TargetLatent()
    lat_dict, info = node.process(
        vae=None,
        target_content="empty",
        geometry_mode="fixed",
        target_megapixels=2.0,
        fixed_megapixels=1.0,
        aspect_ratio="1:1",
    )
    assert lat_dict["samples"].shape[1] == 16
    assert "Geometry Strategy: fixed" in info


def test_normalize_vae_output_accepts_native_5d():
    latent = torch.zeros((1, 16, 1, 144, 216))
    res = normalize_vae_output(latent, batch_size=1)
    assert res.shape == (1, 16, 1, 144, 216)
    assert res is latent


def test_normalize_vae_output_5d_batch_expansion():
    latent = torch.zeros((1, 16, 1, 144, 216))
    res = normalize_vae_output(latent, batch_size=2)
    assert res.shape == (2, 16, 1, 144, 216)
    assert res.shape[2] == 1  # T remains 1
    assert res.shape[3] == 144
    assert res.shape[4] == 216


def test_target_latent_scene_content_5d_latent():
    img = torch.rand(1, 600, 400, 3)
    scene_prep = prepare_vision_image(image=img, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 1, 144, 216))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="scene",
        scene_image=scene_prep,
        target_geometry="favor_scene",
        maximum_mp=1.0,
        batch_size=1,
    )
    assert lat_dict["samples"].shape == (1, 16, 1, 144, 216)
    assert "Latent Content: scene" in info
    assert "VAE Encode Applied: yes" in info


def test_target_latent_subject_content_5d_latent():
    img = torch.rand(1, 600, 400, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = torch.zeros((1, 16, 1, 144, 216))

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="subject",
        subject_image=subj_prep,
        target_geometry="favor_subject",
        maximum_mp=1.0,
        batch_size=2,
    )
    assert lat_dict["samples"].shape == (2, 16, 1, 144, 216)


def test_normalize_vae_output_invalid_dimensions_raises():
    tensor_3d = torch.zeros((16, 144, 216))
    with pytest.raises(ValueError, match="Normalized VAE latent must be a 4D or 5D tensor"):
        normalize_vae_output(tensor_3d, batch_size=1)

    tensor_6d = torch.zeros((1, 16, 1, 1, 144, 216))
    with pytest.raises(ValueError, match="Normalized VAE latent must be a 4D or 5D tensor"):
        normalize_vae_output(tensor_6d, batch_size=1)


def test_target_latent_legacy_translation_precedence():
    img_subj = torch.rand(1, 500, 300, 3)
    img_scene = torch.rand(1, 800, 600, 3)
    subj_prep = prepare_vision_image(image=img_subj, clip=None, mode="native")
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="scene",
        subject_image=subj_prep,
        scene_image=scene_prep,
        target_geometry="favor_scene",
        maximum_mp=1.0,
        batch_size=1,
    )
    assert lat_dict["target_vision_context"].target_image is scene_prep


def test_target_latent_legacy_geometry_favor_subject():
    img_subj = torch.rand(1, 400, 300, 3)  # 3:4
    img_scene = torch.rand(1, 300, 400, 3)  # 4:3
    subj_prep = prepare_vision_image(image=img_subj, clip=None, mode="native")
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="subject",
        subject_image=subj_prep,
        scene_image=scene_prep,
        target_geometry="favor_subject",
        maximum_mp=1.0,
        batch_size=1,
    )
    assert "Content Source Size: 300 x 400" in info


def test_target_latent_legacy_geometry_favor_scene():
    img_subj = torch.rand(1, 400, 300, 3)  # 3:4
    img_scene = torch.rand(1, 300, 400, 3)  # 4:3
    subj_prep = prepare_vision_image(image=img_subj, clip=None, mode="native")
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="subject",
        subject_image=subj_prep,
        scene_image=scene_prep,
        target_geometry="favor_scene",
        maximum_mp=1.0,
        batch_size=1,
    )
    assert "Content Source Size: 300 x 400" in info
    assert "Content Target Size: 400 x 304" in info


def test_target_latent_force_target_megapixels():
    img_scene = torch.rand(1, 300, 400, 3)  # 4:3 (~0.12 MP)
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="scene",
        scene_image=scene_prep,
        target_geometry="favor_scene",
        maximum_mp=1.0,
        batch_size=1,
        force_target_megapixels=True,
    )
    assert "Content Target Size: 1152 x 864" in info


def test_generic_favor_image_does_not_upscale_small_images():
    img_scene = torch.rand(1, 300, 400, 3)  # 4:3 (~0.12 MP)
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")
    mock_vae = MagicMock()
    mock_vae.encode.return_value = {"samples": torch.zeros((1, 16, 64, 64))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="scene",
        scene_image=scene_prep,
        target_geometry="favor_scene",
        maximum_mp=2.0,
        batch_size=1,
        force_target_megapixels=False,
    )
    # Small image (0.12 MP) should NOT upscale to max 2.0 MP by default
    assert "Content Target Size: 400 x 304" in info


def test_target_latent_strict_image_content():
    img_subj = torch.rand(1, 400, 300, 3)
    subj_prep = prepare_vision_image(image=img_subj, clip=None, mode="native")
    mock_vae = MagicMock()

    with pytest.raises(ValueError, match="Target image is required when target_content is 'image'"):
        create_target_latent(
            vae=mock_vae,
            target_latent_content="image",
            target_image=None,
            subject_image=subj_prep,
            target_geometry="favor_image",
            maximum_mp=1.0,
            batch_size=1,
        )


def test_target_latent_contain_no_upscale_portrait_into_landscape():
    img_subj = torch.rand(1, 1448, 1086, 3)  # 1086 x 1448 (W x H)
    img_scene = torch.rand(1, 1152, 1728, 3)  # 1728 x 1152 (W x H)
    subj_prep = prepare_vision_image(image=img_subj, clip=None, mode="native")
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")

    mock_vae = MagicMock()

    def mock_encode(x):
        if x.ndim == 4 and x.shape[-1] in (1, 3, 4):
            h, w = x.shape[1], x.shape[2]
        else:
            h, w = x.shape[2], x.shape[3]
        return {"samples": torch.ones((1, 16, h // 8, w // 8))}

    mock_vae.encode.side_effect = mock_encode

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="subject",
        subject_image=subj_prep,
        scene_image=scene_prep,
        target_geometry="favor_scene",
        content_fit="contain_no_upscale",
        maximum_mp=2.0,
        batch_size=1,
    )

    samples = lat_dict["samples"]
    assert samples.shape == (1, 16, 144, 216)
    # Check centered offset: lw = 108, lw offset = (216 - 108) // 2 = 54
    # Padding left (0..54) and right (162..216) must be zeros
    assert torch.all(samples[:, :, :, :54] == 0)
    assert torch.all(samples[:, :, :, 162:] == 0)
    assert torch.all(samples[:, :, :, 54:162] == 1.0)

    assert "Content Fit: contain_no_upscale" in info
    assert "Content Source Size: 1086 x 1448" in info
    assert "Content Resolved Size: 864 x 1152" in info
    assert "Content Crop Rectangle: none" in info
    assert "Content Latent Placement: centered" in info
    assert "Content Latent Offset: X=54, Y=0" in info
    assert "Content Target Size: 1728 x 1152" in info


def test_target_latent_contain_no_upscale_small_image():
    img_subj = torch.rand(1, 300, 400, 3)  # 400 x 300 (W x H) small subject
    img_scene = torch.rand(1, 1000, 1000, 3)  # 1000 x 1000 scene target
    subj_prep = prepare_vision_image(image=img_subj, clip=None, mode="native")
    scene_prep = prepare_vision_image(image=img_scene, clip=None, mode="native")

    mock_vae = MagicMock()
    mock_vae.encode.side_effect = lambda x: {"samples": torch.ones((1, 16, x.shape[2] // 8, x.shape[3] // 8))}

    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="subject",
        subject_image=subj_prep,
        scene_image=scene_prep,
        target_geometry="favor_scene",
        content_fit="contain_no_upscale",
        maximum_mp=1.0,
        batch_size=1,
    )

    # 400x300 fits without resampling; only the height is padded to the next /16 boundary.
    assert "Content Fit: contain_no_upscale" in info
    assert "Content Source Size: 400 x 300" in info
    assert "Content Resolved Size: 400 x 304" in info


def _shape_only_image(width, height):
    return torch.empty((1, height, width, 3), device="meta")


def test_subject_aware_geometry_keeps_scene_when_subject_already_fits():
    plan = calculate_subject_aware_scene_geometry(
        scene_image=_shape_only_image(1200, 800),
        subject_image=_shape_only_image(600, 700),
    )
    assert (plan.target_width, plan.target_height) == (1200, 800)
    assert plan.scene_was_downscaled is False
    assert plan.latent_was_expanded is False
    assert plan.subject_requires_downscale is False


def test_subject_aware_geometry_uses_subject_ceiling_alignment_as_fit_constraint():
    plan = calculate_subject_aware_scene_geometry(
        scene_image=_shape_only_image(1200, 800),
        subject_image=_shape_only_image(1199, 799),
    )
    assert (plan.aligned_subject_width, plan.aligned_subject_height) == (1200, 800)
    assert (plan.target_width, plan.target_height) == (1200, 800)
    assert plan.latent_was_expanded is False
    assert plan.subject_requires_downscale is False


def test_subject_aware_geometry_expands_scene_below_two_mp_without_touching_subject():
    plan = calculate_subject_aware_scene_geometry(
        scene_image=_shape_only_image(1200, 800),
        subject_image=_shape_only_image(900, 960),
    )
    assert (plan.target_width, plan.target_height) == (1440, 960)
    assert plan.latent_was_expanded is True
    assert plan.latent_was_capped is False
    assert plan.subject_requires_downscale is False


def test_subject_aware_geometry_resizes_scene_to_two_mp_before_fit_check():
    plan = calculate_subject_aware_scene_geometry(
        scene_image=_shape_only_image(2400, 1600),
        subject_image=_shape_only_image(1000, 1000),
    )
    assert (plan.target_width, plan.target_height) == (1728, 1152)
    assert plan.scene_was_downscaled is True
    assert plan.subject_requires_downscale is False


def test_subject_aware_geometry_uses_full_two_mp_when_expansion_from_small_scene_exceeds_cap():
    plan = calculate_subject_aware_scene_geometry(
        scene_image=_shape_only_image(1200, 800),
        subject_image=_shape_only_image(1400, 1600),
    )
    assert (plan.target_width, plan.target_height) == (1728, 1152)
    assert plan.scene_was_downscaled is False
    assert plan.latent_was_expanded is True
    assert plan.latent_was_capped is True
    assert plan.subject_requires_downscale is True


def test_subject_aware_geometry_downscales_subject_only_when_two_mp_canvas_cannot_contain_it():
    plan = calculate_subject_aware_scene_geometry(
        scene_image=_shape_only_image(2400, 1600),
        subject_image=_shape_only_image(1400, 1600),
    )
    assert (plan.target_width, plan.target_height) == (1728, 1152)
    assert plan.scene_was_downscaled is True
    assert plan.latent_was_expanded is False
    assert plan.latent_was_capped is True
    assert plan.subject_requires_downscale is True
