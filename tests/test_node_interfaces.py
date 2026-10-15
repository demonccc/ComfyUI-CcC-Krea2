"""Public node registration contract after unified reference refactor."""
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

EXPECTED = {
    "CcCKrea2Reference",
    "CcCKrea2SizeResolver",
    "CcCKrea2Latent",
    "CcCKrea2Edit",
    "CcCKrea2EditPromptCreator",
    "CcCKrea2CharacterSheet",
    "CcCKrea2PaintPrepare",
    "CcCKrea2PaintRestore",
    "CcCKrea2Paint",
    "CcCKrea2LoRAStackWithPrompt",
    "CcCKrea2LoRAStack",
    "CcCKrea2LoRALoader",
    "CcCKrea2TextToImage",
}

def test_public_nodes():
    assert set(NODE_CLASS_MAPPINGS) == EXPECTED
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == EXPECTED
    assert NODE_DISPLAY_NAME_MAPPINGS["CcCKrea2Reference"] == "Krea2 CcC Reference"
    assert "CcCKrea2AttentionRegion" not in NODE_CLASS_MAPPINGS
    assert "CcCKrea2VisualReference" not in NODE_CLASS_MAPPINGS
    assert "CcCKrea2VisionReference" not in NODE_CLASS_MAPPINGS

def test_edit_and_paint_node_categories():
    for name in ("CcCKrea2Reference", "CcCKrea2Edit", "CcCKrea2PaintPrepare", "CcCKrea2PaintRestore", "CcCKrea2Paint"):
        assert NODE_CLASS_MAPPINGS[name].CATEGORY == "Krea2 CcC Edit"
