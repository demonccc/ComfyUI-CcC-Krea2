"""Unit tests for settings precedence, presets, and immutability."""

import pytest
from ccc_krea2.settings import (
    ImageRoleSettings,
    ImageAdvancedSettingsBundle,
    EditAdvancedSettings,
    resolve_krea2_settings,
)


def test_preset_defaults():
    # 1. Balanced preset
    res_b = resolve_krea2_settings(preset_name="balanced")
    assert res_b.batch_size == 1
    assert res_b.attention_mask_mode == "hard"
    assert res_b.roles["subject"].boost == 2.5
    assert res_b.roles["subject"].grounding_px == 768
    assert res_b.roles["scene"].boost == 1.0
    assert res_b.roles["outfit"].boost == 1.0
    assert res_b.roles["source"].boost == 1.0

    # 2. Max Identity preset
    res_m = resolve_krea2_settings(preset_name="max_identity")
    assert res_m.roles["subject"].boost == 4.0
    assert res_m.roles["subject"].grounding_px == 1024
    assert res_m.roles["source"].boost == 2.5
    assert res_m.roles["source"].grounding_px == 1024
    assert res_m.roles["scene"].boost == 1.0  # Inherits balanced

    # 3. Flexible preset
    res_f = resolve_krea2_settings(preset_name="flexible")
    assert res_f.roles["subject"].boost == 1.5
    assert res_f.roles["subject"].grounding_px == 512
    assert res_f.roles["source"].boost == 1.5
    assert res_f.roles["source"].grounding_px == 512


def test_edit_settings_override():
    edit = EditAdvancedSettings(
        batch_size=4,
        sampling_resize_mode="crop",
        attention_mask_mode="soft",
        prompt_instructions_mode="append",
        prompt_instructions="Use vivid lighting",
    )
    resolved = resolve_krea2_settings(preset_name="balanced", edit_settings=edit)
    assert resolved.batch_size == 4
    assert resolved.sampling_resize_mode == "crop"
    assert resolved.attention_mask_mode == "soft"
    assert resolved.prompt_instructions_mode == "append"
    assert resolved.prompt_instructions == "Use vivid lighting"
    # Roles retain preset defaults
    assert resolved.roles["subject"].boost == 2.5


def test_image_settings_group_overrides_and_unconfigured_inheritance():
    # Only override_attention is True
    subject_override = ImageRoleSettings(
        override_attention=True,
        boost=3.8,
        mask_invert=True,
        override_grounding=False,
        grounding_px=1024,  # Should be ignored because override_grounding=False
    )
    bundle = ImageAdvancedSettingsBundle().with_role("subject", subject_override)

    resolved = resolve_krea2_settings(preset_name="balanced", image_settings=bundle)
    # Attention group is overridden
    assert resolved.roles["subject"].boost == 3.8
    assert resolved.roles["subject"].mask_invert is True
    # Grounding group is NOT overridden, inherits preset
    assert resolved.roles["subject"].grounding_px == 768


def test_chained_image_settings_accumulation_and_last_wins():
    b0 = ImageAdvancedSettingsBundle()
    b1 = b0.with_role("scene", ImageRoleSettings(override_attention=True, boost=1.8))
    b2 = b1.with_role("outfit", ImageRoleSettings(override_attention=True, boost=2.2))
    b3 = b2.with_role("scene", ImageRoleSettings(override_attention=True, boost=3.5))  # Last occurrence of scene wins

    # Verify immutability: b1 and b2 are unchanged
    assert "outfit" not in b1.role_settings
    assert b1.role_settings["scene"].boost == 1.8
    assert b3.role_settings["scene"].boost == 3.5
    assert b3.role_settings["outfit"].boost == 2.2


def test_runtime_regression_requirement_balanced():
    """Verify that preset=balanced with no advanced settings resolves to exact tested core values."""
    resolved = resolve_krea2_settings(preset_name="balanced", active_roles=["subject"])
    sub = resolved.roles["subject"]

    assert sub.boost == 2.5
    assert sub.grounding_resize_mode == "normalize"
    assert sub.grounding_px == 768
    assert sub.grounding_min_px == 512
    assert sub.grounding_max_px == 1024
    assert resolved.attention_mask_mode == "hard"
    assert sub.reference_fit_mode == "fit"
    assert resolved.batch_size == 1
