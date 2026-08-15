"""Unit tests for output resolution calculation and megapixel aspect ratio math."""

import torch
import logging
from ccc_krea2.resolution import resolve_output_resolution
from ccc_krea2.t2i import calculate_t2i_resolution


def test_role_based_resolution_preserves_dimensions_and_aligns_16():
    # Image of size 1920x1080 [B, H, W, C] -> H=1080, W=1920
    img_1080p = torch.zeros((1, 1080, 1920, 3))
    node_images = {"subject": img_1080p}

    w, h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="off",
    )
    # 1920 is a multiple of 16 (120*16). 1080 / 16 = 67.5 -> round to 68 -> 68*16 = 1088.
    assert w == 1920
    assert h == 1088


def test_role_based_resolution_minimum_size():
    # Tiny image 50x30
    tiny_img = torch.zeros((1, 30, 50, 3))
    node_images = {"subject": tiny_img}

    w, h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
    )
    # Floor min size is 128
    assert w >= 128
    assert h >= 128


def test_custom_megapixel_resolution_aspect_ratio_math():
    # 16:9 aspect ratio image (1920x1080)
    aspect_img = torch.zeros((1, 1080, 1920, 3))
    node_images = {"subject": aspect_img}

    w, h = resolve_output_resolution(
        output_resolution="custom",
        megapixels=1.0,  # 1,000,000 pixels
        node_images=node_images,
        default_auto_role="subject",
    )
    # 16:9 at 1 MP -> target pixels 1e6 -> raw_w = sqrt(1e6 * (16/9)) = 1333.33 -> 1328 or 1344
    # raw_h = sqrt(1e6 / (16/9)) = 750.0 -> 752
    assert w % 16 == 0
    assert h % 16 == 0
    assert abs((w * h) - 1_000_000) < 50_000


def test_custom_megapixel_fallback_when_aspect_role_unavailable(caplog):
    caplog.set_level(logging.WARNING)
    img_subject = torch.zeros((1, 1000, 1000, 3))  # 1:1
    node_images = {"subject": img_subject, "scene": None}

    w, h = resolve_output_resolution(
        output_resolution="custom",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        custom_aspect_source="scene",  # scene is None
        node_name="CcC Krea2 - Subject + Scene",
    )

    # Should log warning and fall back to default_auto_role ('subject')
    assert "Selected custom_aspect_source 'scene' is not available" in caplog.text
    assert w == 1000 or w == 992 or w == 1008
    assert h == 1000 or h == 992 or h == 1008


def test_calculate_t2i_resolution_all_aspect_ratios():
    # 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3
    w, h = calculate_t2i_resolution("16:9", 1.0)
    assert w == 1328 and h == 752

    w_custom, h_custom = calculate_t2i_resolution("custom", 1.0, custom_aspect_width=21, custom_aspect_height=9)
    assert w_custom % 16 == 0 and h_custom % 16 == 0
    assert w_custom > h_custom
