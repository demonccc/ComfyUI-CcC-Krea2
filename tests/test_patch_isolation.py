"""CPU-safe unit tests for ModelPatcher patch isolation."""

import pytest
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

    # Apply patch
    patched = patch_krea2_model(original, prepared_refs=[], target_h=1024, target_w=1024)

    # Verify original is completely unpatched and pristine
    assert "transformer_options" not in original.model_options

    # Verify patched model has transformer_options with wrapper registered
    assert "transformer_options" in patched.model_options
    to = patched.model_options["transformer_options"]
    assert "wrappers" in to
