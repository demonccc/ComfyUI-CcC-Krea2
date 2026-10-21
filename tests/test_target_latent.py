"""Tests for active VAE latent normalization helpers."""

import pytest
import torch

from ccc_krea2.target_latent import get_image_dims, normalize_vae_output


def test_get_image_dims_supports_batched_and_unbatched_images():
    assert get_image_dims(torch.zeros((2, 160, 128, 3))) == (160, 128)
    assert get_image_dims(torch.zeros((160, 128, 3))) == (160, 128)
    with pytest.raises(ValueError, match="Unsupported image tensor shape"):
        get_image_dims(torch.zeros((160, 128)))


def test_normalize_vae_output_preserves_4d_and_5d_shapes():
    for latent in (torch.rand((1, 16, 18, 24)), torch.rand((1, 16, 1, 18, 24))):
        assert normalize_vae_output(latent, batch_size=1) is latent


def test_normalize_vae_output_repeats_only_the_batch_dimension():
    for latent in (torch.rand((1, 16, 18, 24)), torch.rand((1, 16, 1, 18, 24))):
        copied = normalize_vae_output({"samples": latent}, batch_size=3)
        assert copied.shape == (3, *latent.shape[1:])
        for item in copied:
            torch.testing.assert_close(item, latent[0])


def test_normalize_vae_output_accepts_sample_method():
    class Encoded:
        def sample(self):
            return torch.ones((1, 16, 2, 4))
    result = normalize_vae_output(Encoded(), batch_size=1)
    assert result.shape == (1, 16, 2, 4)


def test_normalize_vae_output_rejects_invalid_shape_and_batch():
    with pytest.raises(ValueError, match="4D or 5D"):
        normalize_vae_output(torch.ones((1, 16, 32)), batch_size=1)
    with pytest.raises(ValueError, match="batch size"):
        normalize_vae_output(torch.ones((2, 16, 8, 8)), batch_size=3)
    with pytest.raises(ValueError, match="Unsupported VAE return format"):
        normalize_vae_output(object(), batch_size=1)
