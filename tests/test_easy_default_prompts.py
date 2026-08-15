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
