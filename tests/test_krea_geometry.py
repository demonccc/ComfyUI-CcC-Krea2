"""Tests for the shared Krea geometry library."""

from ccc_krea2.krea_geometry import (
    DEFAULT_KREA_PRESET_SIZE,
    KREA_PRESET_GEOMETRIES,
    KREA_PRESET_SIZES,
    preset_geometry,
    resolve_krea_geometry,
)
from ccc_krea2.modular_nodes.latent_node import CcCKrea2Latent
from ccc_krea2.t2i import CcCKrea2TextToImage


def test_shared_krea_presets_are_the_node_source_of_truth():
    latent_required = CcCKrea2Latent.INPUT_TYPES()["required"]
    t2i_required = CcCKrea2TextToImage.INPUT_TYPES()["required"]

    assert latent_required["preset_size"][0] == KREA_PRESET_SIZES
    assert t2i_required["preset_size"][0] == KREA_PRESET_SIZES
    assert latent_required["preset_size"][1]["default"] == DEFAULT_KREA_PRESET_SIZE
    assert t2i_required["preset_size"][1]["default"] == DEFAULT_KREA_PRESET_SIZE


def test_shared_krea_preset_resolution():
    assert preset_geometry("1024 x 1024 | 1:1 | ~1.05 MP") == (1024, 1024)
    assert preset_geometry("2048 x 1152 | 16:9 | ~2.36 MP") == (2048, 1152)
    assert len(KREA_PRESET_SIZES) == len(KREA_PRESET_GEOMETRIES)


def test_shared_krea_fixed_geometry_resolution():
    assert resolve_krea_geometry(1328, 752, "preserve_aspect_krea_bounds") == (1808, 1024)
    assert resolve_krea_geometry(1328, 752, "nearest_krea_aspect") == (2048, 1152)
