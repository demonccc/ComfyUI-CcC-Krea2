"""Contracts for the split Krea2 Edit nodes."""

import torch

from ccc_krea2.krea2edit_geometry import resolve_krea2edit_geometry
from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit, _runtime_latent
from ccc_krea2.modular_nodes.latent_node import CcCKrea2Latent, _center_place, _resolve_dimensions
from ccc_krea2.modular_nodes.rope_position import build_incontext_3d_rope_pos_ids, resolve_rope_axes
from ccc_krea2.modular_nodes.semantic_reference_node import CcCKrea2SemanticReference
from ccc_krea2.modular_nodes.size_resolver_node import CcCKrea2SizeResolver, _resolve_size
from ccc_krea2.modular_nodes.visual_reference_node import CcCKrea2VisualReference
from ccc_krea2.patch import attach_reference_boosts_to_conditioning, _normalize_runtime_boosts


def test_visual_reference_exposes_three_axis_rope_controls_without_reference_sizing_toggle():
    inputs = CcCKrea2VisualReference.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert list(required) == [
        "image",
        "boost",
        "rope_grid",
        "rope_horizontal",
        "rope_vertical",
        "semantic",
        "semantic_role",
        "instruction",
        "grounding_px",
    ]
    assert list(optional) == ["previous_references"]
    assert "fit_to_latent" not in required
    assert required["rope_grid"][0] == ("inside", "outside")
    assert required["rope_horizontal"][0] == ("center", "left", "right")
    assert required["rope_vertical"][0] == ("center", "up", "down")


def test_visual_reference_semantic_role_is_ignored_when_semantic_is_disabled():
    image = torch.zeros((1, 64, 64, 3))
    (chain,) = CcCKrea2VisualReference().process(
        image=image,
        semantic=False,
        semantic_role="subject image",
        instruction="Use subject identity",
    )
    entry = chain.entries[0]
    assert entry.semantic is False
    assert entry.semantic_role == ""
    assert entry.instruction == ""


def test_visual_reference_chain_preserves_order_boost_and_rope_axes():
    scene = torch.zeros((1, 64, 96, 3))
    subject = torch.zeros((1, 96, 64, 3))
    node = CcCKrea2VisualReference()

    (scene_chain,) = node.process(image=scene, semantic_role="scene image")
    (chain,) = node.process(
        image=subject,
        boost=4.0,
        rope_grid="inside",
        rope_horizontal="left",
        rope_vertical="up",
        semantic_role="subject image",
        previous_references=scene_chain,
    )

    assert len(chain.entries) == 2
    assert chain.entries[1].boost == 4.0
    assert not hasattr(chain.entries[1], "fit_to_latent")
    assert chain.entries[1].rope_position == "inside:left:up"


def test_krea2_v12_fit_regression_for_1719x1164_reference_against_992_square_target():
    geom = resolve_krea2edit_geometry(
        src_h=1164,
        src_w=1719,
        tgt_h=992,
        tgt_w=992,
        fit_mode="fit",
    )

    assert geom.mode_resolved == "fit"
    assert geom.vae_input_pixel_size == (992, 656)
    assert geom.vae_latent_grid_size == (124, 82)
    assert geom.target_grid_size == (124, 124)
    assert geom.interpolation_method == "bicubic"
    assert geom.whether_interpolation_occurred is True


def test_rope_center_matches_rednode_integer_center_after_v12_fit():
    # VAE latent grids 124x82 (ref) and 124x124 (target) become DiT patch grids
    # 62x41 and 62x62 with Krea2 patch size 2.
    pos = build_incontext_3d_rope_pos_ids(
        batch_size=1,
        txt_len=0,
        ref_token_grids=[(41, 62)],
        target_grid=(62, 62),
        ref_rope_positions=["inside:center:center"],
        device=torch.device("cpu"),
    )

    ref = pos[0, : 41 * 62]
    assert ref[:, 1].min().item() == 10.0
    assert ref[:, 2].min().item() == 0.0


def test_rope_axes_preserve_optional_outside_displacement():
    assert resolve_rope_axes("none") == ("inside", "center", "center")
    assert resolve_rope_axes("up") == ("outside", "center", "up")
    assert resolve_rope_axes("inside:left:up") == ("inside", "left", "up")

    pos = build_incontext_3d_rope_pos_ids(
        batch_size=1,
        txt_len=0,
        ref_token_grids=[(2, 3)],
        target_grid=(6, 8),
        ref_rope_positions=["outside:left:center"],
        device=torch.device("cpu"),
    )
    ref = pos[0, :6]
    assert ref[:, 1].min().item() == 2.0
    assert ref[:, 2].min().item() == -3.0


def test_positive_reference_boost_metadata_is_explicit_and_negative_defaults_to_neutral():
    positive_conditioning = [[torch.zeros((1, 4, 8)), {}]]
    negative_conditioning = [[torch.zeros((1, 4, 8)), {}]]

    positive = attach_reference_boosts_to_conditioning(positive_conditioning, [4.0])

    assert positive[0][1]["reference_boosts"] == [4.0]
    assert "reference_boosts" not in negative_conditioning[0][1]
    assert _normalize_runtime_boosts(None, 1) == [1.0]


def test_runtime_reference_boosts_default_to_neutral_and_align_to_reference_count():
    assert _normalize_runtime_boosts(None, 2) == [1.0, 1.0]
    assert _normalize_runtime_boosts([4.0], 2) == [1.0, 4.0]
    assert _normalize_runtime_boosts([2.0, 3.0, 4.0], 2) == [3.0, 4.0]


def test_semantic_reference_exposes_advanced_semantic_controls():
    required = CcCKrea2SemanticReference.INPUT_TYPES()["required"]
    assert required["mode"][0] == ("semantic_only", "style_direct", "style_indirect")
    assert required["processing"][0] == ("full", "2x2", "4x4")


def test_size_resolver_outputs_only_width_and_height_from_two_images():
    size_image = torch.zeros((1, 1600, 1400, 3))
    aspect_image = torch.zeros((1, 640, 1024, 3))

    width, height = _resolve_size(size_image, aspect_image)
    assert (width, height) == (1600, 1000)

    node = CcCKrea2SizeResolver()
    assert node.RETURN_TYPES == ("INT", "INT")
    assert node.RETURN_NAMES == ("width", "height")
    assert node.process(size_image, aspect_image) == (1600, 1000)


def test_latent_dimension_modes_are_explicit_and_align_to_16():
    image = torch.zeros((1, 1003, 1501, 3))

    width, height, source = _resolve_dimensions(
        dimensions="from_image",
        dimensions_image=image,
        width=1024,
        height=1024,
        resolution="1.0 MP",
        aspect_ratio="1:1",
    )
    assert (width, height) == (1504, 1008)
    assert source == "image dimensions 1501 x 1003"

    fixed_w, fixed_h, _ = _resolve_dimensions(
        dimensions="fixed",
        dimensions_image=None,
        width=1501,
        height=1003,
        resolution="1.0 MP",
        aspect_ratio="1:1",
    )
    assert (fixed_w, fixed_h) == (1504, 1008)

    preset_w, preset_h, _ = _resolve_dimensions(
        dimensions="preset",
        dimensions_image=None,
        width=1024,
        height=1024,
        resolution="1.0 MP",
        aspect_ratio="16:9",
    )
    assert preset_w % 16 == 0
    assert preset_h % 16 == 0


def test_latent_surface_separates_dimensions_and_content():
    inputs = CcCKrea2Latent.INPUT_TYPES()
    required = inputs["required"]
    optional = inputs["optional"]

    assert list(required) == [
        "vae",
        "dimensions",
        "width",
        "height",
        "resolution",
        "aspect_ratio",
        "content",
        "image_fit",
        "resize_method",
        "latent_semantic",
        "latent_semantic_instruction",
        "latent_grounding_px",
        "batch_size",
    ]
    assert required["dimensions"][0] == ("from_image", "fixed", "preset")
    assert required["resolution"][0] == ("0.5 MP", "1.0 MP", "1.5 MP", "2.0 MP", "2.5 MP")
    assert required["aspect_ratio"][0] == ("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")
    assert required["content"][0] == ("empty", "from_image")
    assert required["image_fit"][0] == ("long_edge", "native", "stretch")
    assert list(optional) == ["dimensions_image", "content_image"]


def test_content_native_keeps_image_size_centered():
    image = torch.zeros((1, 300, 500, 3))
    canvas, placement = _center_place(
        image=image,
        target_w=800,
        target_h=600,
        image_fit="native",
        resize_method="auto",
    )
    assert canvas.shape == (1, 600, 800, 3)
    assert placement["fitted_size"] == (500, 300)
    assert placement["scale"] == 1.0


def test_content_long_edge_matches_longest_target_edge():
    image = torch.zeros((1, 500, 300, 3))
    canvas, placement = _center_place(
        image=image,
        target_w=800,
        target_h=600,
        image_fit="long_edge",
        resize_method="bicubic",
    )
    assert canvas.shape == (1, 600, 800, 3)
    assert placement["fitted_size"] == (480, 800)
    assert placement["scale"] == 1.6


def test_content_stretch_fills_target_geometry():
    image = torch.zeros((1, 500, 300, 3))
    canvas, placement = _center_place(
        image=image,
        target_w=800,
        target_h=600,
        image_fit="stretch",
        resize_method="bicubic",
    )
    assert canvas.shape == (1, 600, 800, 3)
    assert placement["fitted_size"] == (800, 600)


class _MockVAE:
    def encode(self, image):
        batch, height, width, _ = image.shape
        return torch.zeros((batch, 16, height // 8, width // 8))


def test_latent_semantic_metadata_uses_content_image():
    image = torch.zeros((1, 128, 128, 3))
    latent, _ = CcCKrea2Latent().process(
        vae=_MockVAE(),
        dimensions="fixed",
        width=128,
        height=128,
        content="from_image",
        content_image=image,
        image_fit="native",
        latent_semantic=True,
        latent_semantic_instruction="Reimagine the target content",
    )
    metadata = latent["ccc_krea2_latent_semantic"]
    assert metadata["enabled"] is True
    assert metadata["image"] is image
    assert metadata["instruction"] == "Reimagine the target content"


def test_edit_consumes_prebuilt_latent_and_no_longer_owns_target_controls():
    required = CcCKrea2Edit.INPUT_TYPES()["required"]
    optional = CcCKrea2Edit.INPUT_TYPES()["optional"]
    assert "latent" in required
    assert list(optional) == ["visual_references", "semantic_references"]
    for moved in (
        "dimensions",
        "dimensions_image",
        "width",
        "height",
        "resolution",
        "aspect_ratio",
        "content",
        "content_image",
        "image_fit",
        "resize_method",
    ):
        assert moved not in required
        assert moved not in optional


def test_edit_runtime_latent_matches_pre_split_contract():
    samples = torch.zeros((1, 16, 64, 80))
    target_vision_context = object()
    latent = {
        "samples": samples,
        "batch_index": [0],
        "target_vision_context": target_vision_context,
        "ccc_krea2_latent_semantic": {"enabled": True, "image": object()},
        "ccc_krea2_latent_info": "diagnostics",
    }

    runtime = _runtime_latent(latent)
    assert set(runtime) == {"samples", "batch_index", "target_vision_context"}
    assert runtime["samples"] is samples
    assert runtime["target_vision_context"] is target_vision_context
