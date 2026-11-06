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
