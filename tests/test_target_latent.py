"""Unit tests for Target Latent node and resolution logic."""

import torch
import pytest
from unittest.mock import MagicMock
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.target_latent import create_target_latent, resolve_target_geometry, normalize_vae_output
from ccc_krea2.modular_nodes.target_latent_node import CcCKrea2TargetLatent


def test_target_geometry_fixed():
    th, tw, geom_src, active_mp, src_dims, warnings = resolve_target_geometry(
        target_geometry="fixed",
        fixed_mp=1.0,
        fixed_aspect_ratio="16:9"
    )
    assert th % 16 == 0
    assert tw % 16 == 0
    assert abs((tw / float(th)) - (16.0 / 9.0)) < 0.1
    assert len(warnings) == 0


def test_target_geometry_favor_subject():
    img = torch.rand(1, 1200, 800, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    th, tw, geom_src, active_mp, src_dims, warnings = resolve_target_geometry(
        target_geometry="favor_subject",
        subject_image=subj_prep,
        maximum_mp=2.0
    )
    assert th % 16 == 0
    assert tw % 16 == 0
    assert abs((tw / float(th)) - (800.0 / 1200.0)) < 0.1


def test_target_latent_empty_content():
    mock_vae = MagicMock()
    lat_dict, info = create_target_latent(
        vae=mock_vae,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=1.0,
        fixed_aspect_ratio="1:1"
    )
    assert "samples" in lat_dict
    assert lat_dict["samples"].shape == (1, 16, 124, 124)
    assert "Latent Content: empty" in info


def test_target_latent_missing_vae_raises():
    img = torch.rand(1, 512, 512, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    with pytest.raises(ValueError, match="VAE is required"):
        create_target_latent(
            vae=None,
            target_latent_content="subject",
            subject_image=subj_prep
        )


def test_target_latent_empty_without_vae():
    lat_dict, info = create_target_latent(
        vae=None,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=1.0,
        fixed_aspect_ratio="1:1"
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
        batch_size=2
    )
    assert lat_dict["samples"].shape == (2, 16, 64, 64)
    assert "Latent Content: subject" in info
    assert "VAE Encode Applied: yes" in info
    mock_vae.encode.assert_called_once()


def test_target_latent_scene_content_missing_raises():
    with pytest.raises(ValueError, match="Scene image is required"):
        create_target_latent(
            vae=MagicMock(),
            target_latent_content="scene",
            scene_image=None
        )


def test_target_latent_node_execution():
    node = CcCKrea2TargetLatent()
    lat_dict, info = node.process(
        vae=None,
        target_content="empty",
        geometry_mode="fixed",
        target_megapixels=2.0,
        fixed_megapixels=1.0,
        aspect_ratio="1:1"
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
        batch_size=1
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
        batch_size=2
    )
    assert lat_dict["samples"].shape == (2, 16, 1, 144, 216)


def test_normalize_vae_output_invalid_dimensions_raises():
    tensor_3d = torch.zeros((16, 144, 216))
    with pytest.raises(ValueError, match="Normalized VAE latent must be a 4D or 5D tensor"):
        normalize_vae_output(tensor_3d, batch_size=1)

    tensor_6d = torch.zeros((1, 16, 1, 1, 144, 216))
    with pytest.raises(ValueError, match="Normalized VAE latent must be a 4D or 5D tensor"):
        normalize_vae_output(tensor_6d, batch_size=1)
