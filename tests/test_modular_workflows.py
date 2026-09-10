"""Unit tests for the split Edit modular surface."""

from ccc_krea2.nodes import NODE_CLASS_MAPPINGS


def test_split_edit_nodes_are_registered_and_valid():
    for name in (
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
        assert cls.CATEGORY == "CcC/Krea2"


def test_reference_size_resolver_and_latent_socket_types_are_distinct():
    visual = NODE_CLASS_MAPPINGS["CcCKrea2VisualReference"]
    semantic = NODE_CLASS_MAPPINGS["CcCKrea2SemanticReference"]
    resolver = NODE_CLASS_MAPPINGS["CcCKrea2SizeResolver"]
    latent = NODE_CLASS_MAPPINGS["CcCKrea2Latent"]
    edit = NODE_CLASS_MAPPINGS["CcCKrea2Edit"]

    assert visual.RETURN_TYPES == ("KREA2_VISUAL_REFERENCE_CHAIN",)
    assert semantic.RETURN_TYPES == ("KREA2_SEMANTIC_REFERENCE_CHAIN",)
    assert resolver.RETURN_TYPES == ("INT", "INT")
    assert resolver.RETURN_NAMES == ("width", "height")
    assert latent.RETURN_TYPES == ("LATENT", "STRING")
    assert latent.INPUT_TYPES()["required"]["width"][0] == "INT"
    assert latent.INPUT_TYPES()["required"]["height"][0] == "INT"
    assert edit.INPUT_TYPES()["required"]["latent"][0] == "LATENT"
