"""Unit tests verifying modular pipeline workflow node class IDs."""

from ccc_krea2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def test_modular_nodes_registered_and_valid():
    modular_classes = [
        "CcCKrea2QwenVisionImagePrep",
        "CcCKrea2TargetLatent",
        "CcCKrea2SubjectImage",
        "CcCKrea2SceneImage",
        "CcCKrea2OutfitImage",
        "CcCKrea2StyleImage",
        "CcCKrea2Edit",
    ]

    for name in modular_classes:
        assert name in NODE_CLASS_MAPPINGS
        assert name in NODE_DISPLAY_NAME_MAPPINGS
        cls = NODE_CLASS_MAPPINGS[name]
        assert hasattr(cls, "INPUT_TYPES")
        assert hasattr(cls, "RETURN_TYPES")
        assert hasattr(cls, "FUNCTION")
        assert cls.CATEGORY == "CcC/Krea2"
