"""CPU-safe unit tests for ModelPatcher patch isolation."""

from ccc_krea2.patch import patch_krea2_model


class DummyModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        new_obj = DummyModel()
        new_obj.model_options = self.model_options.copy()
        return new_obj


def test_patch_isolation():
    original = DummyModel()

    patched = patch_krea2_model(original)

    # Verify original is completely unpatched and pristine
    assert "transformer_options" not in original.model_options

    # Verify patched model has transformer_options with wrapper registered
    assert "transformer_options" in patched.model_options
    to = patched.model_options["transformer_options"]
    assert "wrappers" in to


def test_reference_isolation_blocks_reference_cross_attention_not_target_references():
    import torch
    from ccc_krea2.patch import _compute_ref_attention_bias_patchified
    bias = _compute_ref_attention_bias_patchified(
        boosts=[1.0, 1.0], txt_len=2, ref_token_lens=[4, 4],
        tgt_len=4, ref_token_grids=[(2, 2), (2, 2)],
        target_grid=(2, 2), isolate_references=True,
    )
    assert bias is not None
    assert torch.isneginf(bias[0, 0, 2:6, 6:]).all()
    assert torch.isneginf(bias[0, 0, 6:10, 2:6]).all()
    assert (bias[0, 0, 2:6, 2:6] == 0).all()
    assert (bias[0, 0, 10:, 2:10] == 0).all()
    assert (bias[0, 0, :2, 2:10] == 0).all()


def test_edit_has_no_kv_mode_and_keeps_reference_isolation():
    from ccc_krea2.modular_nodes.edit_node import CcCKrea2Edit

    schema = CcCKrea2Edit.INPUT_TYPES()["optional"]
    assert "use_kv_reference" not in schema
    assert schema["isolate_references"][0] == "BOOLEAN"
    assert schema["isolate_references"][1]["default"] is False


def test_edit_joint_wrapper_propagates_reference_isolation(monkeypatch):
    import torch
    from ccc_krea2 import patch

    captured = {}

    def fake_joint(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(patch, "krea2_dit_incontext_forward", fake_joint)

    class FakeExecutor:
        class_obj = object()

    class WrappedModel:
        def clone(self):
            return WrappedModel()

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrapper = wrapper

    model = patch_krea2_model(WrappedModel())
    x = torch.zeros((1, 16, 8, 8))
    ref = torch.ones_like(x)
    model.wrapper(
        FakeExecutor(), x, torch.tensor([1.0]), torch.zeros((1, 2, 16)),
        ref_latents=[ref], ccc_ref_attention_isolation=True,
        transformer_options={},
    )
    assert captured["isolate_references"] is True
    assert len(captured["ref_latents"]) == 1
