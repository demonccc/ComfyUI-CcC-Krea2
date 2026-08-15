"""Unit tests for CcCKrea2LoRAPromptSettings and CcCKrea2LoRAStack custom nodes."""

from dataclasses import FrozenInstanceError
import pytest

from ccc_krea2.prompt_augmentation import (
    CCC_KREA2_LORA_PROMPT_SETTINGS,
    CCC_KREA2_PROMPT_AUGMENTATION,
    LoRAPromptSettings,
    PromptAugmentation,
)
from ccc_krea2.lora import (
    CcCKrea2LoRAPromptSettings,
    CcCKrea2LoRAStack,
)


class FakeModel:
    def __init__(self, name="base_model"):
        self.name = name


# --- Tests: LoRA Prompt Settings ---


def test_lora_prompt_settings_node_contract():
    node = CcCKrea2LoRAPromptSettings()
    inp = node.INPUT_TYPES()
    req = inp["required"]

    assert "enabled" in req
    for i in range(1, 5):
        assert f"lora_{i}_prompt_enabled" in req
        assert f"lora_{i}_prompt_position" in req
        assert f"lora_{i}_positive_prompt" in req
        assert f"lora_{i}_negative_prompt" in req

    assert node.RETURN_TYPES == (CCC_KREA2_LORA_PROMPT_SETTINGS,)
    assert node.RETURN_NAMES == ("lora_prompt_settings",)


def test_lora_prompt_settings_creation_and_immutability():
    node = CcCKrea2LoRAPromptSettings()
    (settings,) = node.build_settings(
        enabled=True,
        lora_1_prompt_enabled=True,
        lora_1_prompt_position="prepend",
        lora_1_positive_prompt="pos 1",
        lora_1_negative_prompt="neg 1",
        lora_2_prompt_enabled=True,
        lora_2_prompt_position="append",
        lora_2_positive_prompt="pos 2",
        lora_2_negative_prompt="neg 2",
    )

    assert isinstance(settings, LoRAPromptSettings)
    assert len(settings.slots) == 4
    assert settings.enabled is True

    # Check slot 1
    s1 = settings.slots[0]
    assert s1.enabled is True
    assert s1.position == "prepend"
    assert s1.positive_prompt == "pos 1"
    assert s1.negative_prompt == "neg 1"

    # Check slot 2
    s2 = settings.slots[1]
    assert s2.enabled is True
    assert s2.position == "append"
    assert s2.positive_prompt == "pos 2"
    assert s2.negative_prompt == "neg 2"

    # Check immutability
    with pytest.raises(FrozenInstanceError):
        settings.enabled = False  # type: ignore

    with pytest.raises(FrozenInstanceError):
        s1.positive_prompt = "changed"  # type: ignore


def test_lora_prompt_settings_determinism():
    node = CcCKrea2LoRAPromptSettings()
    res1 = node.build_settings(enabled=True, lora_1_positive_prompt="test")
    res2 = node.build_settings(enabled=True, lora_1_positive_prompt="test")
    assert res1[0] == res2[0]


# --- Tests: LoRA Stack ---


def test_lora_stack_node_contract():
    node = CcCKrea2LoRAStack()
    inp = node.INPUT_TYPES()
    req = inp["required"]
    opt = inp["optional"]

    assert "model" in req
    assert "enabled" in req
    assert "global_strength" in req
    for i in range(1, 5):
        assert f"lora_{i}_enabled" in req
        assert f"lora_{i}_name" in req
        assert f"lora_{i}_strength" in req

    assert "prompt_augmentation" in opt
    assert "lora_prompt_settings" in opt

    assert node.RETURN_TYPES == ("MODEL", CCC_KREA2_PROMPT_AUGMENTATION)
    assert node.RETURN_NAMES == ("model", "prompt_augmentation")


def test_lora_stack_no_cache_invalidating_hook():
    assert not hasattr(CcCKrea2LoRAStack, "IS_CHANGED")
    assert not hasattr(CcCKrea2LoRAStack, "not_idempotent")


def test_lora_stack_disabled_stack_returns_input_model():
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("input")

    # Mock loaders to detect if load is called
    for loader in node.loaders:
        loader.load = lambda m, name, strn: FakeModel("should_not_be_called")  # type: ignore

    out_model, out_aug = node.apply_loras(input_model, enabled=False)
    assert out_model is input_model
    assert isinstance(out_aug, PromptAugmentation)


def test_lora_stack_disabled_stack_passes_incoming_augmentation():
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("input")
    inc_aug = PromptAugmentation(positive_prepend=("incoming",))

    out_model, out_aug = node.apply_loras(input_model, enabled=False, prompt_augmentation=inc_aug)
    assert out_model is input_model
    assert out_aug is inc_aug


def test_lora_stack_no_active_slots_returns_input_model():
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("input")

    # All slots disabled or None name
    out_model, out_aug = node.apply_loras(
        input_model,
        enabled=True,
        lora_1_enabled=False,
        lora_1_name="None",
    )
    assert out_model is input_model


def test_lora_stack_active_loras_applied_in_order_with_effective_strength(monkeypatch):
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("base")
    call_history = []

    def make_fake_load(slot_idx):
        def fake_load(m, lora_name, strength):
            call_history.append((slot_idx, lora_name, strength))
            return FakeModel(f"{m.name}_{lora_name}_{strength}")

        return fake_load

    for idx, loader in enumerate(node.loaders):
        monkeypatch.setattr(loader, "load", make_fake_load(idx + 1))

    out_model, _ = node.apply_loras(
        input_model,
        enabled=True,
        global_strength=0.5,
        lora_1_enabled=True,
        lora_1_name="lora_a.safetensors",
        lora_1_strength=1.0,
        lora_2_enabled=True,
        lora_2_name="lora_b.safetensors",
        lora_2_strength=-2.0,
        lora_3_enabled=False,  # disabled slot
        lora_3_name="lora_c.safetensors",
        lora_3_strength=1.0,
        lora_4_enabled=True,
        lora_4_name="None",  # None name
        lora_4_strength=1.0,
    )

    assert len(call_history) == 2
    assert call_history[0] == (1, "lora_a.safetensors", 0.5)
    assert call_history[1] == (2, "lora_b.safetensors", -1.0)
    assert out_model.name == "base_lora_a.safetensors_0.5_lora_b.safetensors_-1.0"


def test_lora_stack_zero_effective_strength_skips_loading(monkeypatch):
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("base")
    call_history = []

    for idx, loader in enumerate(node.loaders):
        monkeypatch.setattr(loader, "load", lambda m, name, s, idx=idx: call_history.append((idx + 1, name, s)))

    node.apply_loras(
        input_model,
        enabled=True,
        global_strength=0.0,
        lora_1_enabled=True,
        lora_1_name="lora_a.safetensors",
        lora_1_strength=1.0,
    )
    assert len(call_history) == 0


def test_lora_stack_duplicate_filenames_applied_independently(monkeypatch):
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("base")
    call_history = []

    for idx, loader in enumerate(node.loaders):
        monkeypatch.setattr(loader, "load", lambda m, name, s, idx=idx: call_history.append((idx + 1, name, s)) or m)

    node.apply_loras(
        input_model,
        enabled=True,
        global_strength=1.0,
        lora_1_enabled=True,
        lora_1_name="same_lora.safetensors",
        lora_1_strength=0.5,
        lora_2_enabled=True,
        lora_2_name="same_lora.safetensors",
        lora_2_strength=0.8,
    )

    assert len(call_history) == 2
    assert call_history[0] == (1, "same_lora.safetensors", 0.5)
    assert call_history[1] == (2, "same_lora.safetensors", 0.8)


def test_lora_stack_independent_loaders_per_slot():
    node = CcCKrea2LoRAStack()
    assert len(node.loaders) == 4
    assert len(set(id(ldr) for ldr in node.loaders)) == 4


def test_lora_prompt_settings_filter_prompts_for_active_loras_only(monkeypatch):
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("base")

    for loader in node.loaders:
        monkeypatch.setattr(loader, "load", lambda m, name, s: m)

    ps_node = CcCKrea2LoRAPromptSettings()
    (settings,) = ps_node.build_settings(
        enabled=True,
        lora_1_prompt_enabled=True,
        lora_1_prompt_position="prepend",
        lora_1_positive_prompt="active pos 1",
        lora_1_negative_prompt="active neg 1",
        lora_2_prompt_enabled=True,
        lora_2_prompt_position="append",
        lora_2_positive_prompt="disabled pos 2",
        lora_2_negative_prompt="disabled neg 2",
    )

    # Slot 1 is active (lora_1_enabled=True), Slot 2 is inactive (lora_2_enabled=False)
    _, out_aug = node.apply_loras(
        input_model,
        enabled=True,
        lora_prompt_settings=settings,
        lora_1_enabled=True,
        lora_1_name="lora_a.safetensors",
        lora_1_strength=1.0,
        lora_2_enabled=False,
        lora_2_name="lora_b.safetensors",
        lora_2_strength=1.0,
    )

    assert out_aug.positive_prepend == ("active pos 1",)
    assert out_aug.negative_prepend == ("active neg 1",)
    assert "disabled pos 2" not in out_aug.positive_append
    assert "disabled neg 2" not in out_aug.negative_append


def test_lora_stack_chaining_and_input_augmentation_immutability(monkeypatch):
    node = CcCKrea2LoRAStack()
    input_model = FakeModel("base")

    for loader in node.loaders:
        monkeypatch.setattr(loader, "load", lambda m, name, s: m)

    inc_aug = PromptAugmentation(
        positive_prepend=("inc_pre",),
        positive_append=("inc_app",),
    )

    ps_node = CcCKrea2LoRAPromptSettings()
    (settings,) = ps_node.build_settings(
        enabled=True,
        lora_1_prompt_enabled=True,
        lora_1_prompt_position="prepend",
        lora_1_positive_prompt="stack_pre_1",
        lora_1_prompt_position_2="append",
    )

    _, out_aug = node.apply_loras(
        input_model,
        enabled=True,
        prompt_augmentation=inc_aug,
        lora_prompt_settings=settings,
        lora_1_enabled=True,
        lora_1_name="lora_a.safetensors",
        lora_1_strength=1.0,
    )

    # Input augmentation must NOT be mutated
    assert inc_aug.positive_prepend == ("inc_pre",)
    assert inc_aug.positive_append == ("inc_app",)

    # Output augmentation extends input augmentation
    assert out_aug.positive_prepend == ("inc_pre", "stack_pre_1")
    assert out_aug.positive_append == ("inc_app",)


def test_lora_stack_identical_inputs_equal_output():
    node = CcCKrea2LoRAStack()
    m1 = FakeModel("base")
    m2 = FakeModel("base")

    node.loaders[0].load = lambda m, name, s: m  # type: ignore

    _, aug1 = node.apply_loras(m1, enabled=True, lora_1_enabled=True, lora_1_name="a.safetensors")
    _, aug2 = node.apply_loras(m2, enabled=True, lora_1_enabled=True, lora_1_name="a.safetensors")

    assert aug1 == aug2
