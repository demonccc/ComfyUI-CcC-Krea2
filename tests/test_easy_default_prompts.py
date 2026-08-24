"""Unit tests for Easy Edit default positive prompt resolution, prompt selection, and reporting."""

import pytest
import torch

from ccc_krea2.easy_routing import (
    EASY_DEFAULT_PROMPT_OUTFIT_TRANSFER,
    EASY_DEFAULT_PROMPT_SUBJECT_SCENE,
    EASY_DEFAULT_PROMPT_SUBJECT_SCENE_OUTFIT,
    EASY_DEFAULT_PROMPT_STYLE,
    EASY_DEFAULT_PROMPT_IDENTITY_TRANSFER,
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
        assert "Place the main subject from the subject image naturally into the scene image." in text

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
        assert "Transfer only the principal outfit identified in the outfit image to the main subject." in text

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
        assert "Place the main subject from the subject image naturally into the scene image wearing the principal outfit identified in the outfit image." in text

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
        assert "Use the style image only as a visual style reference." in text
        assert "Apply its color palette, lighting character, contrast, texture" in text
        assert "Do not transfer subjects, identities, facial features" in text

    def test_flexible_subject_transfer_subject_outfit_and_optional_style(self):
        has_def, text, key = resolve_default_positive_prompt(
            preset="flexible_subject_transfer_1",
            has_s=True,
            has_sc=True,
            has_o=True,
            has_st=False,
            outfit_source="subject image",
        )
        assert has_def is True
        assert key == "flexible_subject_transfer_1"
        assert "Preserve the face, body shape, body proportions, clothing, and accessories" in text
        assert "Do not preserve the clothing" not in text

        has_def, text, key = resolve_default_positive_prompt(
            preset="flexible_subject_transfer_2",
            has_s=True,
            has_sc=True,
            has_o=True,
            has_st=True,
            outfit_source="scene image",
            style_source="style image",
        )
        assert has_def is True
        assert key == "subject_transfer_scene_outfit_style"
        assert "Dress the transferred main subject using the clothing, footwear, and accessories worn by the main subject in the scene image." in text
        assert "Use the style image only as a visual style reference." in text

    def test_placeholder_substitution_matrix(self):
        # Example A: identity_transfer with custom subjects
        has_def, text, key = resolve_default_positive_prompt(
            preset="identity_transfer",
            has_s=True,
            has_sc=True,
            has_o=False,
            has_st=False,
            reference_subject="woman playing volleyball",
            subject_description="woman from the portrait",
        )
        assert has_def is True
        expected = (
            "Replace only the identity of the woman playing volleyball of the scene image with the identity of the woman from the portrait from the subject image.\n\n"
            "Transfer the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the woman from the portrait from the subject image to the woman playing volleyball of the scene image.\n\n"
            "Preserve the position, action, pose, role, interaction, clothing, and accessories of the woman playing volleyball from the scene image."
        )
        assert text == expected
        assert "Do not transfer the clothing" not in text
        assert "Keep every other person" not in text

        # Test presets in IDENTITY_TEST_PRESETS use the exact same Identity Transfer prompt template
        from ccc_krea2.easy_routing import IDENTITY_TEST_PRESETS

        for test_preset in IDENTITY_TEST_PRESETS:
            has_def, test_text, test_key = resolve_default_positive_prompt(
                preset=test_preset,
                has_s=True,
                has_sc=True,
                has_o=False,
                has_st=False,
                reference_subject="hero",
                subject_description="champion",
            )
            assert has_def is True
            assert test_key == test_preset
            assert test_text == (
                "Replace only the identity of the hero of the scene image with the identity of the champion from the subject image.\n\n"
                "Transfer the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the champion from the subject image to the hero of the scene image.\n\n"
                "Preserve the position, action, pose, role, interaction, clothing, and accessories of the hero from the scene image."
            )

        # Subject Transfer candidates use the BFS body-swap trigger and preserve Subject clothing.
        for preset in ("subject_transfer_1", "subject_transfer_2"):
            has_def, text, key = resolve_default_positive_prompt(
                preset=preset,
                has_s=True,
                has_sc=True,
                has_o=False,
                has_st=True,
                reference_subject="target model",
                subject_description="cyberpunk warrior",
            )
            assert has_def is True
            assert key == preset
            assert "body_swap" not in text
            assert "Replace only the target model of the scene image with the cyberpunk warrior from the subject image." in text
            assert "Transfer the exact facial identity, facial features, hair, anatomy, body shape, body proportions, clothing, and accessories of the cyberpunk warrior from the subject image." in text
            assert "Place the transferred cyberpunk warrior in the same position and pose as the target model." in text
            assert "perform the same action, fulfill the same role, and interact with every person and object in the same way as the target model" in text
            assert text.endswith("Keep every other person and the rest of the scene unchanged.")

        # Scene Reinterpretation with an explicit Outfit source
        has_def, text, key = resolve_default_positive_prompt(
            preset="scene_reinterpretation",
            has_s=True,
            has_sc=True,
            has_o=True,
            has_st=True,
            outfit_source="subject image",
            style_source="scene image",
            reference_subject="dancer",
            subject_description="portrait woman",
        )
        assert has_def is True
        assert key == "scene_reinterpretation_outfit"
        assert "Dress the portrait woman using the clothing, footwear, and accessories worn by the portrait woman in the subject image." in text

        # Test empty/blank fallback
        has_def, text, key = resolve_default_positive_prompt(
            preset="identity_transfer",
            has_s=True,
            has_sc=True,
            has_o=False,
            has_st=False,
            reference_subject="   ",
            subject_description="",
        )
        assert has_def is True
        assert "Replace only the identity of the main subject of the scene image with the identity of the main subject from the subject image." in text


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

        assert "Place the main subject from the subject image naturally into the scene image." in captured["positive_prompt"]
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

        assert "Transfer only the principal outfit identified in the outfit image to the main subject." in captured["positive_prompt"]
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

    def test_partial_scene_reinterpretation_empty_custom_prompt_raises(self, dummy_images):
        S, Sc, _, _ = dummy_images
        node = CcCKrea2EasyEdit()

        # Subject-only Scene Reinterpretation
        with pytest.raises(
            ValueError,
            match="Scene Reinterpretation requires a custom positive prompt when both Subject and Scene are not available.",
        ):
            node.process(
                model="model",
                clip="clip",
                vae="vae",
                positive_prompt="",
                use_default_prompt=True,
                preset="scene_reinterpretation",
                subject=S,
            )

        # Scene-only Scene Reinterpretation
        with pytest.raises(
            ValueError,
            match="Scene Reinterpretation requires a custom positive prompt when both Subject and Scene are not available.",
        ):
            node.process(
                model="model",
                clip="clip",
                vae="vae",
                positive_prompt="",
                use_default_prompt=True,
                preset="scene_reinterpretation",
                scene=Sc,
            )

    def test_partial_scene_reinterpretation_with_custom_prompt_succeeds(self, dummy_images, monkeypatch):
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

        # Subject-only + Custom
        _, _, _, _, report1 = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="My custom reinterpretation prompt",
            use_default_prompt=True,
            preset="scene_reinterpretation",
            subject=S,
        )
        assert captured["positive_prompt"] == "My custom reinterpretation prompt"
        assert "Prompt Source: custom" in report1
        assert "Scene Reinterpretation selected but Scene source is missing." in report1

        # Scene-only + Custom
        _, _, _, _, report2 = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="My custom scene-only reinterpretation prompt",
            use_default_prompt=True,
            preset="scene_reinterpretation",
            scene=Sc,
        )
        assert captured["positive_prompt"] == "My custom scene-only reinterpretation prompt"
        assert "Prompt Source: custom" in report2
        assert "Scene Reinterpretation selected but Subject source is missing." in report2


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


class TestEasyEditReportLatentSource:
    def test_report_resolved_latent_source_and_roles(self, dummy_images, monkeypatch):
        S, Sc, Ou, St = dummy_images
        node = CcCKrea2EasyEdit()

        class DummyVAE:
            def encode(self, x):
                return torch.zeros((1, 16, 8, 8), dtype=torch.float32)

        def mock_orchestrator(*args, **kwargs):
            return ("patched_model", "pos", "neg", "lat", "orchestrator_report")

        monkeypatch.setattr(
            "ccc_krea2.modular_nodes.easy_edit_node.run_krea2_edit_orchestrator",
            mock_orchestrator,
        )

        # Identity Transfer + Subject + Scene
        _, _, _, _, report_id = node.process(
            model="model",
            clip="clip",
            vae=DummyVAE(),
            positive_prompt="",
            use_default_prompt=True,
            preset="identity_transfer",
            subject=S,
            scene=Sc,
        )
        assert "Resolved Latent Source: scene image" in report_id
        assert "Target Content Role: scene" in report_id
        assert "Resolved Style Source: scene image (automatic)" in report_id
        assert "Appearance Ref 1: scene (boost=2.0, fit=contain)" in report_id
        assert "Appearance Ref 2: subject (boost=2.0, fit=contain)" in report_id
        assert "Target Content Fit: crop" in report_id

        # Scene Reinterpretation + Subject + Scene + optional Outfit; Style is locked to Scene
        _, _, _, _, report2 = node.process(
            model="model",
            clip="clip",
            vae="vae",
            positive_prompt="",
            use_default_prompt=True,
            preset="scene_reinterpretation",
            subject=S,
            scene=Sc,
            outfit=Ou,
            style=St,
        )
        assert "Resolved Latent Source: empty" in report2
        assert "Target Content Role: none" in report2
        assert "Appearance Ref 1: subject (boost=7.0, fit=contain)" in report2
        assert "Appearance Ref 2: outfit (boost=4.0, fit=contain)" in report2
        assert "Semantic-only Sources: none" in report2
        assert "Resolved Style Source: scene image (automatic)" in report2

        # Preserve Identity using Subject as target
        _, _, _, _, report3 = node.process(
            model="model",
            clip="clip",
            vae=DummyVAE(),
            positive_prompt="Custom prompt",
            use_default_prompt=False,
            preset="preserve_identity",
            subject=S,
        )
        assert "Resolved Latent Source: subject image" in report3
        assert "Target Content Role: subject" in report3

    def test_identity_transfer_all_presets_default_prompt_parity(self):
        """Verify prompt parity across all identity transfer test presets."""
        from ccc_krea2.easy_routing import IDENTITY_TEST_PRESETS

        assert len(IDENTITY_TEST_PRESETS) == 1

        expected_prompt_text = (
            "Replace only the identity of the volleyball player of the scene image with the identity of the portrait subject from the subject image.\n\n"
            "Transfer the exact facial identity, facial features, hair, anatomy, body shape, and body proportions of the portrait subject from the subject image to the volleyball player of the scene image.\n\n"
            "Preserve the position, action, pose, role, interaction, clothing, and accessories of the volleyball player from the scene image."
        )

        for preset in IDENTITY_TEST_PRESETS:
            has_def, text, key = resolve_default_positive_prompt(
                preset=preset,
                has_s=True,
                has_sc=True,
                has_o=False,
                has_st=False,
                reference_subject="volleyball player",
                subject_description="portrait subject",
            )
            assert has_def is True, f"Preset {preset} failed to resolve default prompt"
            assert key == preset, f"Preset {preset} resolved key '{key}' instead of '{preset}'"
            assert text == expected_prompt_text, f"Preset {preset} prompt text mismatch"
            assert "style" not in text.lower()
            assert "outfit" not in text.lower()
            assert "woman" not in text.lower()
