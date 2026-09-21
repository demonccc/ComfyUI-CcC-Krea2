"""Tests for tagged attention-region geometry and crop guardrails."""

import pytest

from ccc_krea2.attention_regions import (
    AttentionRegionChain,
    build_attention_region,
    resolve_attention_regions,
)


def _chain(*regions):
    chain = AttentionRegionChain()
    for region in regions:
        chain = chain.append(region)
    return chain


def test_region_chain_requires_unique_tags():
    woman = build_attention_region("woman", 10, 10, 30, 70)
    chain = _chain(woman)
    with pytest.raises(ValueError, match="Duplicate region tag 'woman'"):
        chain.append(build_attention_region("woman", 50, 10, 30, 70))


def test_empty_target_keeps_normalized_box_on_final_canvas():
    chain = _chain(build_attention_region("woman", 10, 20, 30, 40))
    resolved = resolve_attention_regions(chain, 1000, 500, {"mode": "empty"})
    assert resolved[0].target_box_px == (100.0, 100.0, 400.0, 300.0)
    assert resolved[0].target_box_normalized == (0.1, 0.2, 0.4, 0.6)


def test_contain_moves_box_with_scaled_image_and_padding():
    chain = _chain(build_attention_region("woman", 0, 0, 50, 100))
    placement = {
        "mode": "contain",
        "source_size": (1000, 500),
        "fitted_size": (800, 400),
        "crop": (0, 0, 800, 400),
        "padding": (0, 100, 0, 100),
    }
    resolved = resolve_attention_regions(chain, 800, 600, placement)
    assert resolved[0].target_box_px == (0.0, 100.0, 400.0, 500.0)


def test_crop_allows_region_only_when_box_is_fully_retained():
    chain = _chain(build_attention_region("center", 25, 0, 50, 100))
    placement = {
        "mode": "crop",
        "source_size": (1000, 500),
        "fitted_size": (1200, 600),
        "crop": (200, 0, 800, 600),
        "padding": (0, 0, 0, 0),
    }
    resolved = resolve_attention_regions(chain, 800, 600, placement)
    assert resolved[0].target_box_px == (100.0, 0.0, 700.0, 600.0)


def test_crop_raises_when_it_partially_cuts_region():
    chain = _chain(build_attention_region("woman", 0, 0, 30, 100))
    placement = {
        "mode": "crop",
        "source_size": (1000, 500),
        "fitted_size": (1200, 600),
        "crop": (200, 0, 800, 600),
        "padding": (0, 0, 0, 0),
    }
    with pytest.raises(ValueError, match="crop touches attention region 'woman'"):
        resolve_attention_regions(chain, 800, 600, placement)


def test_crop_raises_when_region_is_completely_outside():
    chain = _chain(build_attention_region("woman", 0, 0, 10, 100))
    placement = {
        "mode": "crop",
        "source_size": (1000, 500),
        "fitted_size": (1200, 600),
        "crop": (300, 0, 800, 600),
        "padding": (0, 0, 0, 0),
    }
    with pytest.raises(ValueError, match="crop touches attention region 'woman'"):
        resolve_attention_regions(chain, 800, 600, placement)
