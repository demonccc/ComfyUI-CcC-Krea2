"""Regression coverage for the split Edit node using the original orchestrator runtime."""

import inspect
from types import SimpleNamespace

import torch

from ccc_krea2.orchestrator_runtime import patch_krea2_orchestrated_model
from ccc_krea2.references import PreparedReference
from ccc_krea2.modular_nodes import edit_node


class _FakeModel:
    def __init__(self):
        self.model = SimpleNamespace(process_latent_in=lambda latent: latent)

    def clone(self):
        return _FakeModel()


def test_orchestrated_model_captures_prepared_references():
    refs = [
        PreparedReference(
            role="scene",
            grounding_image=None,
            vae_latent=torch.zeros((1, 16, 1, 176, 116)),
            spatial_attention_mask=None,
            boost=1.0,
            rope_position="inside:center:center",
        ),
        PreparedReference(
            role="subject",
            grounding_image=None,
            vae_latent=torch.zeros((1, 16, 1, 176, 82)),
            spatial_attention_mask=None,
            boost=4.0,
            rope_position="inside:center:center",
        ),
    ]

    patched = patch_krea2_orchestrated_model(_FakeModel(), refs)

    assert patched._ccc_orchestrated_reference_count == 2
    assert patched._ccc_orchestrated_reference_shapes == [
        (1, 16, 1, 176, 116),
        (1, 16, 1, 176, 82),
    ]
    assert patched._ccc_orchestrated_reference_boosts == [1.0, 4.0]
    assert patched._ccc_orchestrated_reference_rope == [
        "inside:center:center",
        "inside:center:center",
    ]


def test_split_edit_routes_through_orchestrator_without_direct_identity_bypass():
    assert edit_node.edit_engine_runtime.patch_krea2_model is patch_krea2_orchestrated_model

    source = inspect.getsource(edit_node.CcCKrea2Edit.process)
    assert "run_krea2_edit_orchestrator" in source
    assert "encode_visual_identity_direct" not in source
    assert "direct_identity" not in source
