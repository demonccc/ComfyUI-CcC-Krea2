"""Unit tests for the current LoRA Stack nodes."""

from ccc_krea2.lora import CcCKrea2LoRAStack, CcCKrea2LoRAStackWithPrompt
from ccc_krea2.prompt_augmentation import CCC_KREA2_PROMPT_AUGMENTATION, PromptAugmentation


class FakeModel:
    def __init__(self, name="base_model"):
        self.name = name



def test_lora_stack_applies_enabled_slots_in_order(monkeypatch):
    node = CcCKrea2LoRAStack()
    model = FakeModel("base")
    calls = []

    for index, loader in enumerate(node.loaders, start=1):
        def fake_load(current, name, strength, index=index):
            calls.append((index, name, strength))
            return FakeModel(f"{current.name}:{name}:{strength}")
        monkeypatch.setattr(loader, "load", fake_load)

    (out_model,) = node.apply_loras(
        model,
        lora_1_enabled=True,
        lora_1_name="a.safetensors",
        lora_1_strength=0.5,
        lora_2_enabled=False,
        lora_2_name="ignored.safetensors",
        lora_3_enabled=True,
        lora_3_name="b.safetensors",
        lora_3_strength=-1.0,
    )

    assert calls == [
        (1, "a.safetensors", 0.5),
        (3, "b.safetensors", -1.0),
    ]
    assert out_model.name.endswith("b.safetensors:-1.0")


def test_lora_stack_skips_none_and_zero_strength(monkeypatch):
    node = CcCKrea2LoRAStack()
    model = FakeModel()
    calls = []
    for loader in node.loaders:
        monkeypatch.setattr(loader, "load", lambda *args: calls.append(args))

    (out_model,) = node.apply_loras(
        model,
        lora_1_enabled=True,
        lora_1_name="None",
        lora_1_strength=1.0,
        lora_2_enabled=True,
        lora_2_name="a.safetensors",
        lora_2_strength=0.0,
    )

    assert out_model is model
    assert calls == []



def test_lora_stack_with_prompt_applies_model_and_prompt(monkeypatch):
    node = CcCKrea2LoRAStackWithPrompt()
    model = FakeModel("base")
    calls = []

    for index, loader in enumerate(node.loaders, start=1):
        def fake_load(current, name, strength, index=index):
            calls.append((index, name, strength))
            return current
        monkeypatch.setattr(loader, "load", fake_load)

    out_model, augmentation = node.apply_loras(
        model,
        lora_1_enabled=True,
        lora_1_name="a.safetensors",
        lora_1_strength=0.75,
        lora_1_prompt_position="prepend",
        lora_1_prompt="trigger A",
        lora_2_enabled=True,
        lora_2_name="None",
        lora_2_strength=1.0,
        lora_2_prompt_position="append",
        lora_2_prompt="trigger B",
        lora_3_enabled=False,
        lora_3_prompt="ignored",
    )

    assert out_model is model
    assert calls == [(1, "a.safetensors", 0.75)]
    assert augmentation.positive_prepend == ("trigger A",)
    assert augmentation.positive_append == ("trigger B",)


def test_lora_stack_with_prompt_preserves_incoming_augmentation(monkeypatch):
    node = CcCKrea2LoRAStackWithPrompt()
    model = FakeModel()
    for loader in node.loaders:
        monkeypatch.setattr(loader, "load", lambda current, name, strength: current)

    incoming = PromptAugmentation(
        positive_prepend=("existing pre",),
        positive_append=("existing post",),
        negative_prepend=("negative pre",),
        negative_append=("negative post",),
    )

    _, augmentation = node.apply_loras(
        model,
        prompt_augmentation=incoming,
        lora_1_enabled=True,
        lora_1_name="None",
        lora_1_prompt_position="append",
        lora_1_prompt="new trigger",
    )

    assert augmentation.positive_prepend == ("existing pre",)
    assert augmentation.positive_append == ("existing post", "new trigger")
    assert augmentation.negative_prepend == ("negative pre",)
    assert augmentation.negative_append == ("negative post",)
    assert incoming.positive_append == ("existing post",)


def test_lora_nodes_have_no_cache_invalidating_hook():
    assert not hasattr(CcCKrea2LoRAStack, "IS_CHANGED")
    assert not hasattr(CcCKrea2LoRAStackWithPrompt, "IS_CHANGED")
