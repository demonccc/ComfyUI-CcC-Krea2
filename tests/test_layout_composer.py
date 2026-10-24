"""Regression tests for the chainable Krea2 image grid composer."""
import pytest
import torch

from ccc_krea2.modular_nodes.layout_node import CcCKrea2LayoutCell, CcCKrea2LayoutComposer


def _img(v):
    return torch.full((1, 50, 30, 3), float(v))


def _sheet():
    chain = None
    for row, column, placement, value in [
        (1, 1, "cell", 0.1), (2, 1, "cell", 0.2),
        (1, 2, "cell", 0.3), (2, 2, "cell", 0.4),
        (1, 3, "full_column", 0.5), (1, 4, "full_column", 0.6),
    ]:
        chain, = CcCKrea2LayoutCell().add(
            _img(value), placement=placement, row=row, column=column,
            fit="stretch", previous_layout=chain,
        )
    return chain


def test_four_columns_two_rows_and_full_height_columns():
    chain = _sheet()
    output, info = CcCKrea2LayoutComposer().compose(
        chain, padding=0, gap=0, background_color="white"
    )
    assert output.shape == (1, 1024, 1024, 3)
    assert "2 rows x 4 columns" in info
    assert torch.allclose(output[:, 100, 100], torch.full((1, 3), 0.1))
    assert torch.allclose(output[:, 700, 100], torch.full((1, 3), 0.2))
    assert torch.allclose(output[:, 100, 300], torch.full((1, 3), 0.3))
    assert torch.allclose(output[:, 700, 300], torch.full((1, 3), 0.4))
    assert torch.allclose(output[:, 100, 600], torch.full((1, 3), 0.5))
    assert torch.allclose(output[:, 700, 600], torch.full((1, 3), 0.5))
    assert torch.allclose(output[:, 100, 900], torch.full((1, 3), 0.6))


def test_conflicting_cells_raise():
    first, = CcCKrea2LayoutCell().add(_img(0.1))
    second, = CcCKrea2LayoutCell().add(_img(0.2), previous_layout=first)
    with pytest.raises(ValueError, match="conflicts"):
        CcCKrea2LayoutComposer().compose(second)


def test_missing_cell_raises():
    first, = CcCKrea2LayoutCell().add(_img(0.1), row=2, column=2)
    with pytest.raises(ValueError, match="empty cell"):
        CcCKrea2LayoutComposer().compose(first)


def test_offsets_crop_relative_to_alignment():
    image = torch.zeros((1, 10, 20, 3))
    image[..., :10, :] = 1
    chain, = CcCKrea2LayoutCell().add(
        image, fit="crop", horizontal_position="left",
        vertical_position="up", offset_x=2, offset_y=3,
    )
    output, _ = CcCKrea2LayoutComposer().compose(chain, padding=0, gap=0, background_color="black")
    assert output[0, 0, 0, 0] == 0
    assert output[0, 3, 2, 0] == 1


def test_invalid_row_rejected():
    with pytest.raises(ValueError, match="start at 1"):
        CcCKrea2LayoutCell().add(_img(0.1), row=0)
