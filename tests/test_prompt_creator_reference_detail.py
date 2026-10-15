"""Create-from-image analyzes visual evidence before writing the edit prompt."""
from ccc_krea2.modular_nodes.edit_prompt_creator_node import PRESET_SYSTEM_PROMPTS, _resolve_system_prompt

def test_create_from_image_has_two_internal_phases():
    prompt = PRESET_SYSTEM_PROMPTS["create_from_image"]
    assert "First, internally inspect" in prompt
    assert "Then use that analysis" in prompt
    assert "clothing" in prompt and "spatial relationships" in prompt
    assert "do not invent details" in prompt
    assert "one plain-text paragraph" in prompt
    assert "Do not output the intermediate analysis" in prompt

def test_custom_prompt_is_unchanged():
    custom = "Describe exactly the single detail I name."
    assert _resolve_system_prompt("custom", custom) == custom

def test_other_presets_are_not_modified():
    assert "authoritative guide" in PRESET_SYSTEM_PROMPTS["enhance"]
    assert "theme or idea" in PRESET_SYSTEM_PROMPTS["create_from_theme"]
