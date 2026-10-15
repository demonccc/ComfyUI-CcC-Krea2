import pytest
import torch

from ccc_krea2.patch import _prepare_reference_latent_for_model


class _DummyKrea2Model:
    def __init__(self):
        self.seen_shapes = []

    def process_latent_in(self, latent):
        self.seen_shapes.append(tuple(int(v) for v in latent.shape))
        return latent.clone()


def test_4d_image_reference_gets_singleton_temporal_axis_before_model_normalization():
    model = _DummyKrea2Model()
    raw = torch.zeros((1, 16, 82, 176), dtype=torch.float32)

    processed = _prepare_reference_latent_for_model(model, raw)

    assert model.seen_shapes == [(1, 16, 1, 82, 176)]
    assert tuple(processed.shape) == (1, 16, 1, 82, 176)


def test_existing_5d_reference_is_not_reinterpreted():
    model = _DummyKrea2Model()
    raw = torch.zeros((1, 16, 1, 82, 176), dtype=torch.float32)

    processed = _prepare_reference_latent_for_model(model, raw)

    assert model.seen_shapes == [(1, 16, 1, 82, 176)]
    assert tuple(processed.shape) == tuple(raw.shape)


def test_invalid_reference_rank_is_rejected_before_model_normalization():
    model = _DummyKrea2Model()
    raw = torch.zeros((16, 82, 176), dtype=torch.float32)

    with pytest.raises(ValueError, match="B,C,H,W or B,C,T,H,W"):
        _prepare_reference_latent_for_model(model, raw)

    assert model.seen_shapes == []
