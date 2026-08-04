"""Unit tests for Target Latent node and resolution logic."""

import torch
import pytest
from ccc_krea2.vision_prep import prepare_vision_image
from ccc_krea2.target_latent import create_target_latent, resolve_target_geometry
from ccc_krea2.modular_nodes.target_latent_node import CcCKrea2TargetLatent


def test_target_geometry_fixed():
    th, tw, warn = resolve_target_geometry(
        target_geometry="fixed",
        fixed_mp=1.0,
        fixed_aspect_ratio="16:9"
    )
    assert th % 16 == 0
    assert tw % 16 == 0
    assert abs((tw / float(th)) - (16.0 / 9.0)) < 0.1
    assert warn == ""


def test_target_geometry_favor_subject():
    img = torch.rand(1, 1200, 800, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    th, tw, _ = resolve_target_geometry(
        target_geometry="favor_subject",
        subject_image=subj_prep,
        maximum_mp=2.0
    )
    assert th % 16 == 0
    assert tw % 16 == 0
    assert abs((tw / float(th)) - (800.0 / 1200.0)) < 0.1


def test_target_latent_empty_content():
    lat_dict, info = create_target_latent(
        vae=None,
        target_latent_content="empty",
        target_geometry="fixed",
        fixed_mp=1.0,
        fixed_aspect_ratio="1:1"
    )
    assert "samples" in lat_dict
    assert lat_dict["samples"].shape == (1, 16, 124, 124)
    assert "Latent Content: Empty" in info


def test_target_latent_missing_vae_raises():
    img = torch.rand(1, 512, 512, 3)
    subj_prep = prepare_vision_image(image=img, clip=None, mode="native")
    with pytest.raises(ValueError, match="VAE is required"):
        create_target_latent(
            vae=None,
            target_latent_content="subject",
            subject_image=subj_prep
        )


def test_target_latent_node_execution():
    node = CcCKrea2TargetLatent()
    lat_dict, info = node.process(
        target_latent_content="empty",
        target_geometry="fixed",
        maximum_mp=2.0,
        fixed_mp=1.0,
        fixed_aspect_ratio="1:1"
    )
    assert lat_dict["samples"].shape[1] == 16
    assert "Geometry Strategy: Fixed" in info
