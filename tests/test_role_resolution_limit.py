"""Focused unit tests for role-based output resolution limiting."""

import torch

from ccc_krea2.resolution import resolve_output_resolution


def test_role_resolution_limit_mode_off_preserves_previous_behavior():
    """1. role_resolution_limit_mode=off preserves the previous role-based behavior."""
    large_img = torch.rand((1, 3200, 4000, 3))  # 4000x3200 (both multiples of 16)
    node_images = {"subject": large_img}

    out_w, out_h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="off",
        role_resolution_max_megapixels=2.0,
    )

    assert out_w == 4000
    assert out_h == 3200


def test_role_resolution_limit_mode_max_megapixels_clamps_large_image():
    """2. role_resolution_limit_mode=max_megapixels clamps a large 4000x3000 role image."""
    large_img = torch.rand((1, 3000, 4000, 3))  # 12 MP
    node_images = {"subject": large_img}

    out_w, out_h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="max_megapixels",
        role_resolution_max_megapixels=2.0,  # 2 MP limit
    )

    area = out_w * out_h
    assert area <= 2_000_000
    assert out_w < 4000
    assert out_h < 3000


def test_limited_result_preserves_aspect_ratio_closely():
    """3. The limited result preserves aspect ratio closely."""
    large_img = torch.rand((1, 3000, 4000, 3))  # Aspect 4/3 = 1.333
    node_images = {"subject": large_img}

    out_w, out_h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="max_megapixels",
        role_resolution_max_megapixels=2.0,
    )

    orig_aspect = 4000.0 / 3000.0
    limited_aspect = float(out_w) / float(out_h)

    assert abs(limited_aspect - orig_aspect) < 0.05


def test_limited_result_aligned_to_16():
    """4. The limited result is aligned to multiples of 16."""
    large_img = torch.rand((1, 3000, 4000, 3))
    node_images = {"subject": large_img}

    out_w, out_h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="max_megapixels",
        role_resolution_max_megapixels=2.0,
    )

    assert out_w % 16 == 0
    assert out_h % 16 == 0


def test_limited_result_remains_at_least_128():
    """5. The limited result remains at least 128 in each dimension."""
    small_img = torch.rand((1, 100, 100, 3))
    node_images = {"subject": small_img}

    out_w, out_h = resolve_output_resolution(
        output_resolution="subject",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="max_megapixels",
        role_resolution_max_megapixels=0.25,
    )

    assert out_w >= 128
    assert out_h >= 128


def test_custom_output_resolution_ignores_role_resolution_limit_mode():
    """6. Custom output resolution ignores role_resolution_limit_mode."""
    large_img = torch.rand((1, 3000, 4000, 3))
    node_images = {"subject": large_img}

    out_w, out_h = resolve_output_resolution(
        output_resolution="custom",
        megapixels=4.0,  # Requests 4 MP in custom mode
        node_images=node_images,
        default_auto_role="subject",
        role_resolution_limit_mode="max_megapixels",
        role_resolution_max_megapixels=0.5,  # 0.5 MP role limit set
    )

    area = out_w * out_h
    # Custom mode should compute ~4 MP target area, ignoring the 0.5 MP role limit
    assert area > 3_000_000


def test_outfit_is_never_resolution_source():
    """7. Outfit is still never a valid resolution source."""
    outfit_img = torch.rand((1, 2000, 2000, 3))
    node_images = {"outfit": outfit_img}

    # Calling resolve_output_resolution with non-existent 'outfit' role key in output_resolution fallbacks to 1024x1024
    out_w, out_h = resolve_output_resolution(
        output_resolution="outfit",
        megapixels=1.0,
        node_images=node_images,
        default_auto_role="subject",
    )

    assert (out_w, out_h) == (1024, 1024)
