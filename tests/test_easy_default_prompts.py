"""Unit tests for Easy Edit default positive prompt resolution, prompt selection, and reporting."""

import pytest
import torch

from ccc_krea2.easy_routing import (
    EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER,
    EASY_DEFAULT_PROMPT_SUBJECT_SCENE,
    EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT,
    EASY_DEFAULT_PROMPT_STYLE,
    resolve_default_positive_prompt,
)
from ccc_krea2.modular_nodes.easy_edit_node import CcCKrea2EasyEdit, CcCKrea2EasyEditOstris


@pytest.fixture
def dummy_images():
    S = torch.ones((1, 64, 64, 3), dtype=torch.float32)
    Sc = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.5
    Ou = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.2
    St = torch.ones((1, 64, 64, 3), dtype=torch.float32) * 0.8
    return S, Sc, Ou, St


class TestDefaultPromptResolver:
    def test_1_subject_only(self):
        has_def, text, key = resolve_default_positive_prompt(
            preset="balanced",
            has_s=True,
            has_sc=False,
            has_o=False,
            has_st=False,
        )
        assert has_def is False
        assert text == ""
        assert key == "none"

    def test_2_subject_scene(self):
        has_def, text, key = resolve_default_positive_prompt(
            preset="balanced",
            has_s=True,
            has_sc=True,
            has_o=False,
            has_st=False,
        )
        assert has_def is True
        assert key == "subject_scene"
        assert text == EASY_DEFAULT_PROMPT_SUBJECT_SCENE

    def test_3_subject_outfit_outfit_transfer(self):
        has_def, text, key = resolve_default_positive_prompt(
            preset="outfit_transfer",
            has_s=True,
            has_sc=False,
            has_o=True,
            has_st=False,
        )
        assert has_def is True
        assert key == "outfit_transfer"
        expected = (
            "Transfer only the outfit and accessories from the outfit reference to the subject. "
            "Preserve the subject identity, body, pose, framing, and composition. "
            "Do not preserve the subject clothing. "
            "Fit the transferred outfit and accessories naturally to the subject. "
            "Keep accessories physically attached to the subject in a natural way and never floating. "
            "Do not duplicate accessories."
        )
        assert text == expected
        assert text == EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER

    def test_4_subject_scene_outfit(self):
        has_def, text, key = resolve_default_positive_prompt(
            preset="balanced",
            has_s=True,
            has_sc=True,
            has_o=True,
            has_st=False,
        )
        assert has_def is True
        assert key == "subject_scene_outfit"
        assert text == EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT

    def test_5_style(self):
        has_def, text, key = resolve_default_positive_prompt(
            preset="style_transfer",
            has_s=True,
            has_sc=False,
            has_o=False,
            has_st=True,
        )
        assert has_def is True
        assert key == "style"
        assert text == EASY_DEFAULT_PROMPT_STYLE
        assert "Do not copy subjects, objects, or scene content from the style reference" in text


class TestEasyEditNodeDefaultPromptBehavior:
    def test_6_use_default_prompt_true_overrides_custom_text(self, dummy_images, monkeypatch):
        S, Sc, _, _ = dummy_images
        node = CcCKrea2EasyEdit()

        captured = {}

        def mock_orchestrator(*args, **kwargs):
            captured["positive_prompt"] = kwargs.get("positive_prompt")
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        _, _, _, _, report = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="Custom prompt text",
            use_default_prompt=True,
            preset="balanced",
            subject=S,
            scene=Sc,
        )

        assert captured["positive_prompt"] == EASY_DEFAULT_PROMPT_SUBJECT_SCENE
        assert "Use Default Prompt: yes" in report
        assert "Prompt Source: default" in report
        assert "Default Prompt Key: subject_scene" in report

    def test_7_use_default_prompt_false_preserves_custom_text(self, dummy_images, monkeypatch):
        S, Sc, _, _ = dummy_images
        node = CcCKrea2EasyEdit()

        captured = {}

        def mock_orchestrator(*args, **kwargs):
            captured["positive_prompt"] = kwargs.get("positive_prompt")
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        _, _, _, _, report = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="Put the subject beside a tree",
            use_default_prompt=False,
            preset="balanced",
            subject=S,
            scene=Sc,
        )

        assert captured["positive_prompt"] == "Put the subject beside a tree"
        assert "Use Default Prompt: no" in report
        assert "Prompt Source: custom" in report
        assert "Default Prompt Key: none" in report

    def test_8_subject_only_empty_custom_prompt_raises(self, dummy_images):
        S, _, _, _ = dummy_images
        node = CcCKrea2EasyEdit()

        with pytest.raises(ValueError, match="Subject-only Easy Edit requires a positive prompt"):
            node.process(
                model="model",
                clip="clip",
                vae="vae",
                positive_prompt="",
                use_default_prompt=True,
                preset="balanced",
                subject=S,
            )

    def test_9_prompt_source_report_and_ostris_parity(self, dummy_images, monkeypatch):
        S, _, Ou, _ = dummy_images
        node = CcCKrea2EasyEditOstris()

        captured = {}

        class DummyVAE:
            def encode(self, x):
                return torch.zeros((1, 16, 8, 8), dtype=torch.float32)

        def mock_orchestrator(*args, **kwargs):
            captured["positive_prompt"] = kwargs.get("positive_prompt")
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        _, _, _, _, report = node.process(
            model="model",
            clip="clip",
            vae=DummyVAE(),
            positive_prompt="",
            use_default_prompt=True,
            preset="outfit_transfer",
            subject=S,
            outfit=Ou,
        )

        assert captured["positive_prompt"] == EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER
        assert "Use Default Prompt: yes" in report
        assert "Prompt Source: default" in report
        assert "Default Prompt Key: outfit_transfer" in report

    def test_10_subject_only_with_custom_prompt_succeeds(self, dummy_images, monkeypatch):
        S, _, _, _ = dummy_images
        node = CcCKrea2EasyEdit()

        captured = {}

        def mock_orchestrator(*args, **kwargs):
            captured["positive_prompt"] = kwargs.get("positive_prompt")
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        _, _, _, _, report = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="Custom subject edit prompt",
            use_default_prompt=True,  # Is overridden to false for subject-only
            preset="balanced",
            subject=S,
        )

        assert captured["positive_prompt"] == "Custom subject edit prompt"
        assert "Use Default Prompt: no" in report
        assert "Prompt Source: custom" in report
        assert "Default Prompt Key: none" in report

    def test_11_reconnect_scene_with_use_default_false_keeps_custom_prompt(self, dummy_images, monkeypatch):
        S, Sc, _, _ = dummy_images
        node = CcCKrea2EasyEdit()

        captured = {}

        def mock_orchestrator(*args, **kwargs):
            captured["positive_prompt"] = kwargs.get("positive_prompt")
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        _, _, _, _, report = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="My preserved prompt",
            use_default_prompt=False,
            preset="balanced",
            subject=S,
            scene=Sc,
        )

        assert captured["positive_prompt"] == "My preserved prompt"
        assert "Use Default Prompt: no" in report
        assert "Prompt Source: custom" in report
        assert "Default Prompt Key: none" in report


class TestEasyEditWorkflowMigration:
    LEGACY_PRESETS = {
        "flexible",
        "balanced",
        "consistent",
        "preserve_identity",
        "max_identity",
        "preserve_scene",
        "outfit_transfer",
        "style_transfer",
    }

    def _migrate_widgets(self, widgets_values):
        vals = list(widgets_values)
        if len(vals) < 2:
            return vals
        val_at_1 = vals[1]
        if isinstance(val_at_1, bool):
            return vals
        if isinstance(val_at_1, str) and val_at_1 in self.LEGACY_PRESETS:
            vals.insert(1, False)
            return vals
        return vals

    def test_case_1_legacy_krea_easy_edit(self):
        input_vals = ["My old prompt", "balanced", "outfit image", "style image", True, "bad quality"]
        migrated = self._migrate_widgets(input_vals)
        assert migrated == ["My old prompt", False, "balanced", "outfit image", "style image", True, "bad quality"]

    def test_case_2_already_new_krea_easy_edit(self):
        input_vals = ["My prompt", True, "balanced", "outfit image", "style image", True, "bad quality"]
        migrated = self._migrate_widgets(input_vals)
        assert migrated == input_vals

    def test_case_3_legacy_custom_prompt_ownership(self, dummy_images, monkeypatch):
        S, Sc, _, _ = dummy_images
        node = CcCKrea2EasyEdit()
        captured = {}

        def mock_orchestrator(*args, **kwargs):
            captured["positive_prompt"] = kwargs.get("positive_prompt")
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        input_vals = ["My old prompt", "balanced", "outfit image", "style image", True, "bad quality"]
        migrated = self._migrate_widgets(input_vals)

        # Process with migrated use_default_prompt = False and changed preset "outfit_transfer"
        node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt=migrated[0],
            use_default_prompt=migrated[1],
            preset="outfit_transfer",
            subject=S,
            scene=Sc,
        )
        assert captured["positive_prompt"] == "My old prompt"

    def test_case_4_legacy_empty_prompt(self):
        input_vals = ["", "balanced", "outfit image", "style image", True, "bad quality"]
        migrated = self._migrate_widgets(input_vals)
        assert migrated == ["", False, "balanced", "outfit image", "style image", True, "bad quality"]

    def test_case_5_legacy_ostris_easy_edit(self):
        input_vals = ["Ostris prompt", "flexible", "outfit image", "style image", True, False, "bad quality"]
        migrated = self._migrate_widgets(input_vals)
        assert migrated == [
            "Ostris prompt",
            False,
            "flexible",
            "outfit image",
            "style image",
            True,
            False,
            "bad quality",
        ]

    def test_case_6_idempotency(self):
        input_vals = ["My prompt", "balanced", "outfit image", "style image", True, "bad quality"]
        pass1 = self._migrate_widgets(input_vals)
        pass2 = self._migrate_widgets(pass1)
        assert pass1 == pass2
        assert pass2 == ["My prompt", False, "balanced", "outfit image", "style image", True, "bad quality"]

    def test_case_7_unknown_value(self):
        input_vals = ["My prompt", "unexpected value", "outfit image", "style image", True, "bad quality"]
        migrated = self._migrate_widgets(input_vals)
        assert migrated == input_vals
