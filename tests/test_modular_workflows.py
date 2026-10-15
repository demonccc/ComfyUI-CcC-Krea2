"""Unified reference and independent Qwen/VAE sockets."""
from ccc_krea2.nodes import NODE_CLASS_MAPPINGS

def test_current_edit_nodes_are_registered():
    for name in ("CcCKrea2Reference", "CcCKrea2SizeResolver", "CcCKrea2Latent", "CcCKrea2Edit"):
        cls = NODE_CLASS_MAPPINGS[name]
        assert hasattr(cls, "INPUT_TYPES")
        assert hasattr(cls, "RETURN_TYPES")
        assert hasattr(cls, "FUNCTION")

def test_unified_reference_contract():
    cls = NODE_CLASS_MAPPINGS["CcCKrea2Reference"]
    assert cls.RETURN_TYPES == (
        "KREA2_VISUAL_REFERENCE_CHAIN", "KREA2_VISION_REFERENCE_CHAIN", "STRING"
    )
    required = cls.INPUT_TYPES()["required"]
    for key in ("tag", "semantic_enabled", "visual_enabled", "vision_strength", "visual_boost"):
        assert key in required
    assert "attention_scope" not in required
    assert "region_tag" not in required

def test_vision_attention_is_opt_in():
    required = NODE_CLASS_MAPPINGS["CcCKrea2Edit"].INPUT_TYPES()["required"]
    assert required["vision_attention_patch"] == ("BOOLEAN", {"default": False})
    assert required["latent"][0] == "LATENT"

def test_latent_has_no_regional_mask_input():
    latent = NODE_CLASS_MAPPINGS["CcCKrea2Latent"]
    assert "attention_regions" not in latent.INPUT_TYPES()["optional"]
    assert latent.RETURN_TYPES == ("LATENT", "STRING")
