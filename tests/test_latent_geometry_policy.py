"""Unit tests for Krea2 CcC Latent geometry and image-fit policies."""

import pytest
import torch

from ccc_krea2.krea_geometry import (
    nearest_krea_geometry,
    preserve_aspect_krea_bounds,
)
from ccc_krea2.modular_nodes.latent_node import _fit_content_image


def test_nearest_krea_geometry_uses_known_preset_geometry():
    assert nearest_krea_geometry(1920, 1080) == (2048, 1152)


def test_preserve_aspect_keeps_already_valid_geometry():
    assert preserve_aspect_krea_bounds(1536, 1024) == (1536, 1024)


def test_preserve_aspect_scales_small_geometry_into_krea_bounds():
    assert preserve_aspect_krea_bounds(800, 1200) == (1024, 1536)


def test_preserve_aspect_scales_large_geometry_into_krea_bounds():
    width, height = preserve_aspect_krea_bounds(4000, 2667)
    assert width == 2048
    assert 1024 <= height <= 2048
    assert width % 16 == 0
    assert height % 16 == 0


def test_preserve_aspect_rejects_geometry_that_cannot_fit_bounds():
    with pytest.raises(ValueError, match="cannot fit this aspect ratio"):
        preserve_aspect_krea_bounds(3000, 1000)


def test_content_fit_crop_fills_target_without_padding():
    image = torch.rand(1, 100, 200, 3)
    fitted, meta = _fit_content_image(
        image=image,
        target_w=100,
        target_h=100,
        content_fit="crop",
        resize_method="bilinear",
    )
    assert fitted.shape == (1, 100, 100, 3)
    assert meta["padding"] == (0, 0, 0, 0)


def test_content_fit_contain_uses_white_padding():
    image = torch.zeros(1, 100, 200, 3)
    fitted, meta = _fit_content_image(
        image=image,
        target_w=100,
        target_h=100,
        content_fit="contain",
        resize_method="bilinear",
    )
    assert fitted.shape == (1, 100, 100, 3)
    assert meta["padding"] != (0, 0, 0, 0)
    assert torch.all(fitted[:, 0, :, :] == 1.0)
    assert torch.all(fitted[:, -1, :, :] == 1.0)


def test_content_fit_stretch_fills_target_directly():
    image = torch.rand(1, 100, 200, 3)
    fitted, meta = _fit_content_image(
        image=image,
        target_w=96,
        target_h=128,
        content_fit="stretch",
        resize_method="bilinear",
    )
    assert fitted.shape == (1, 128, 96, 3)
    assert meta["fitted_size"] == (96, 128)


def test_latent_keeps_original_outputs_and_adds_vision_metadata_only():
    from ccc_krea2.modular_nodes.latent_node import CcCKrea2Latent

    class FakeVAE:
        def encode(self, image):
            batch, height, width, _ = image.shape
            return torch.zeros((batch, 16, height // 8, width // 8))

    image = torch.rand((1, 128, 128, 3))
    node = CcCKrea2Latent()
    latent, report, target = node.process(
        vae=FakeVAE(),
        dimensions="fixed", width=1024, height=1024,
        content="from_image", content_image=image,
        latent_semantic=True,
        latent_vision_mode="context",
        latent_vision_instruction="Use the image only as scene context.",
        latent_vision_resize=True,
        latent_vision_max_edge=768,
    )
    assert node.RETURN_TYPES == ("LATENT", "STRING", "IMAGE")
    assert latent["samples"].shape == (1, 16, 128, 128)
    assert target.shape == (1, 1024, 1024, 3)
    assert latent["ccc_krea2_latent_semantic"]["image"] is image
    assert latent["ccc_krea2_latent_semantic"]["mode"] == "context"
    assert "Latent Semantic: enabled" in report


def test_latent_vision_fields_match_reference_vision_contract():
    from ccc_krea2.modular_nodes.latent_node import CcCKrea2Latent
    from ccc_krea2.modular_nodes.vision_reference_node import CcCKrea2VisionReference

    fields = CcCKrea2Latent.INPUT_TYPES()["required"]
    for name, spec in CcCKrea2VisionReference.INPUT_TYPES()["required"].items():
        if name != "image":
            assert fields["latent_vision_" + name] == spec
