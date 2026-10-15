"""Independent paths and tag merging for the unified reference node."""
import torch
from ccc_krea2.modular_nodes.reference_node import CcCKrea2Reference
from ccc_krea2.modular_nodes.edit_node import _combine_reference_chains

IMAGE = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

def build(semantic, visual, tag="subject", strength=0.45, boost=4.0):
    return CcCKrea2Reference().process(
        image=IMAGE,
        tag=tag,
        semantic_enabled=semantic,
        visual_enabled=visual,
        vision_mode="semantic",
        vision_instruction="Preserve clothing details but not extra people.",
        vision_strength=strength,
        visual_boost=boost,
    )

def test_both_paths_share_one_tag_and_reference():
    appearance, vision, _ = build(True, True)
    assert len(appearance.entries) == 1
    assert len(vision.entries) == 1
    assert appearance.entries[0].reference_tag == vision.entries[0].reference_tag == "subject"
    chain = _combine_reference_chains(None, appearance, vision, {})
    assert len(chain.references) == 1
    ref = chain.references[0]
    assert ref.appearance_reference is True
    assert ref.include_in_vision is True
    assert ref.alias == "subject"
    assert ref.attention_boost == 4.0
    assert ref.vision_strength == 0.45
    assert ref.vision_reference_mode == "semantic"

def test_qwen_only_never_creates_appearance_latent():
    appearance, vision, _ = build(True, False)
    assert len(appearance.entries) == 0
    chain = _combine_reference_chains(None, appearance, vision, {})
    assert len(chain.references) == 1
    assert chain.references[0].appearance_reference is False
    assert chain.references[0].vision_strength == 0.45

def test_vae_only_does_not_send_qwen_image():
    appearance, vision, _ = build(False, True)
    assert len(vision.entries) == 0
    chain = _combine_reference_chains(None, appearance, vision, {})
    assert len(chain.references) == 1
    assert chain.references[0].include_in_vision is False
    assert chain.references[0].attention_boost == 4.0
