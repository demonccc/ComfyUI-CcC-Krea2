"""Regression checks for Moodboard-aligned Vision embedding behavior."""
import torch

from ccc_krea2.modular_nodes.edit_reference_types import VisionReferenceEntry
from ccc_krea2.vision_reference_runtime import build_vision_reference_spec
from ccc_krea2.style_processing import StyleSpanOperation, apply_statistical_style_fidelity


def _image():
    return torch.rand((1, 64, 64, 3))


def _op(extract, fidelity, rows=0):
    return StyleSpanOperation(
        logical_reference_id="ref", logical_vision_slot=1,
        physical_qwen_index=1, row_start=2, row_end=8,
        style_fidelity=fidelity, indirect_style_transfer=False,
        extract=extract, summary_rows=rows,
    )


def test_subject_matches_moodboard_whitening_formula():
    x = torch.arange(10 * 24, dtype=torch.float32).reshape(1, 10, 24)
    result, _, _ = apply_statistical_style_fidelity(x, [_op("subject", 0.25)])
    span = x.reshape(1, 10, 12, 2)[:, 2:8]
    mu = span.mean(dim=1, keepdim=True)
    sigma = span.std(dim=1, keepdim=True, correction=1 if span.shape[1] > 1 else 0) + 1e-6
    expected = 0.25 * span + 0.75 * ((span - mu) / sigma)
    assert torch.allclose(result.reshape(1, 10, 12, 2)[:, 2:8], expected)


def test_style_matches_moodboard_channel_statistics():
    x = torch.randn((1, 10, 24))
    result, _, _ = apply_statistical_style_fidelity(x, [_op("style", 0.0)])
    span = x.reshape(1, 10, 12, 2)[:, 2:8]
    mu = span.mean(dim=1, keepdim=True)
    sigma = span.std(dim=1, keepdim=True, correction=1 if span.shape[1] > 1 else 0) + 1e-6
    expected = torch.cat((mu, mu + sigma, mu - sigma), dim=1)[:, [0, 1, 2, 0, 1, 2]]
    assert torch.allclose(result.reshape(1, 10, 12, 2)[:, 2:8], expected)


def test_context_does_not_discard_spatial_vision_rows():
    spec = build_vision_reference_spec(
        VisionReferenceEntry(image=_image(), mode="context_indirect")
    )
    assert spec.vision_reference_mode == "context"
    assert spec.vision_summary_rows == 0
    assert spec.vision_indirect is False
    assert spec.indirect_vision_context is False


def test_visual_rows_are_kept_without_statistical_transform():
    x = torch.randn((1, 10, 24))
    result, changed, removed = apply_statistical_style_fidelity(x, [])
    assert torch.equal(result, x)
    assert changed is False
    assert removed == []


def test_single_row_style_transform_is_finite():
    x = torch.randn((1, 4, 24))
    op = StyleSpanOperation("ref", 1, 1, 2, 3, 0.0, False, "style", 0)
    result, _, _ = apply_statistical_style_fidelity(x, [op])
    assert torch.isfinite(result).all()
