"""Public node mapping contract."""

from ccc_krea2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def test_public_node_mappings():
    expected = {
        "CcCKrea2AttentionRegion",
        "CcCKrea2VisualReference",
        "CcCKrea2SemanticReference",
        "CcCKrea2SizeResolver",
        "CcCKrea2Latent",
        "CcCKrea2Edit",
        "CcCKrea2EditPromptCreator",
        "CcCKrea2CharacterSheet",
        "CcCKrea2PaintPrepare",
        "CcCKrea2PaintRestore",
        "CcCKrea2Paint",
        "CcCKrea2LoRAPromptSettings",
        "CcCKrea2LoRAStack",
        "CcCKrea2TextToImage",
    }
    assert set(NODE_CLASS_MAPPINGS) == expected
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == expected


def test_split_edit_display_names():
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2AttentionRegion"] == "Krea2 CcC Attention Region"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2VisualReference"] == "Krea2 CcC Visual Reference"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2SemanticReference"] == "Krea2 CcC Semantic Reference"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2SizeResolver"] == "Krea2 CcC Size Resolver"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2Latent"] == "Krea2 CcC Latent"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2Edit"] == "Krea2 CcC Edit"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2EditPromptCreator"] == "Krea2 CcC Edit Prompt Creator"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2CharacterSheet"] == "Krea2 CcC Character Sheet"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2PaintPrepare"] == "Krea2 CcC Paint Prepare"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2PaintRestore"] == "Krea2 CcC Paint Restore"
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2Paint"] == "Krea2 CcC Paint"


def test_only_current_edit_nodes_are_registered():
    assert "CcCKrea2Edit" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2EditPromptCreator" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2AttentionRegion" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2VisualReference" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2SemanticReference" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2CharacterSheet" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2PaintPrepare" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2PaintRestore" in NODE_CLASS_MAPPINGS
    assert "CcCKrea2Paint" in NODE_CLASS_MAPPINGS


def test_paint_nodes_use_public_category():
    assert NODE_CLASS_MAPPINGS["CcCKrea2PaintPrepare"].CATEGORY == "Krea2 CcC Edit"
    assert NODE_CLASS_MAPPINGS["CcCKrea2PaintRestore"].CATEGORY == "Krea2 CcC Edit"
    assert NODE_CLASS_MAPPINGS["CcCKrea2Paint"].CATEGORY == "Krea2 CcC Edit"
