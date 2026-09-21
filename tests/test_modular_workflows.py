"""Unit tests for the split Edit modular surface."""

from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


def test_split_edit_nodes_are_registered_and_valid():
    for name in (
        "CcCKrea2AttentionRegion",
        "CcCKrea2VisualReference",
        "CcCKrea2SemanticReference",
        "CcCKrea2SizeResolver",
        "CcCKrea2Latent",
        "CcCKrea2Edit",
    ):
        cls = NODE_CLASS_MAPPINGS[name]
        assert hasattr(cls, "INPUT_TYPES")
        assert hasattr(cls, "RETURN_TYPES")
        assert hasattr(cls, "FUNCTION")
        assert cls.CATEGORY == "Krea2 CcC Edit"


def test_reference_size_resolver_and_latent_socket_types_are_distinct():
    region = NODE_CLASS_MAPPINGS["CcCKrea2AttentionRegion"]
    visual = NODE_CLASS_MAPPINGS["CcCKrea2VisualReference"]
    semantic = NODE_CLASS_MAPPINGS["CcCKrea2SemanticReference"]
    resolver = NODE_CLASS_MAPPINGS["CcCKrea2SizeResolver"]
    latent = NODE_CLASS_MAPPINGS["CcCKrea2Latent"]
    edit = NODE_CLASS_MAPPINGS["CcCKrea2Edit"]

    assert region.RETURN_TYPES == ("KREA2_ATTENTION_REGION_CHAIN",)
    assert visual.RETURN_TYPES == ("KREA2_VISUAL_REFERENCE_CHAIN",)
    assert semantic.RETURN_TYPES == ("KREA2_SEMANTIC_REFERENCE_CHAIN",)
    assert resolver.RETURN_TYPES == ("INT", "INT")
    assert resolver.RETURN_NAMES == ("width", "height")
    assert latent.RETURN_TYPES == ("LATENT", "STRING")
    assert latent.INPUT_TYPES()["required"]["width"][0] == "INT"
    assert latent.INPUT_TYPES()["required"]["height"][0] == "INT"
    assert "preset_size" in latent.INPUT_TYPES()["required"]
    assert latent.INPUT_TYPES()["optional"]["attention_regions"][0] == "KREA2_ATTENTION_REGION_CHAIN"
    assert latent.INPUT_TYPES()["required"]["geometry_policy"][0] == (
        "nearest_krea_aspect",
        "preserve_aspect_krea_bounds",
    )
    assert latent.INPUT_TYPES()["required"]["content_fit"][0] == ("crop", "contain", "stretch")
    assert "image_fit" not in latent.INPUT_TYPES()["required"]
    assert "resolution" not in latent.INPUT_TYPES()["required"]
    assert "aspect_ratio" not in latent.INPUT_TYPES()["required"]
    assert edit.INPUT_TYPES()["required"]["latent"][0] == "LATENT"


def test_latent_preset_sizes_are_explicit_krea_geometries():
    latent = NODE_CLASS_MAPPINGS["CcCKrea2Latent"]
    preset_values = latent.INPUT_TYPES()["required"]["preset_size"][0]
    assert preset_values[0] == "1024 x 1024 | 1:1 | ~1.05 MP"
    assert "1216 x 832 | ~3:2 | ~1.01 MP" in preset_values
    assert "832 x 1216 | ~2:3 | ~1.01 MP" in preset_values
    assert "2048 x 2048 | 1:1 | ~4.19 MP" in preset_values
    for label in preset_values:
        size = label.split(" | ", 1)[0]
        width, height = [int(part.strip()) for part in size.split("x")]
        assert width % 16 == 0
        assert height % 16 == 0
