"""CPU-safe unit tests for mask processing and attention boost numerical stability."""

import torch
import math
from ccc_krea2.masks import process_inpaint_mask, process_attention_mask
from ccc_krea2.patch import _compute_ref_attention_bias_patchified
from ccc_krea2.latents import _process_inpaint_mask, generate_krea2_latent


class MockVAE:
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            b, c, h, w = image.shape
        else:
            b, h, w = 1, image.shape[0], image.shape[1]
        return torch.zeros((b, 16, max(1, h // 8), max(1, w // 8)))


def test_inpaint_mask_processing():
    mask = torch.zeros((100, 100))
    mask[40:60, 40:60] = 1.0

    # Invert
    inv = process_inpaint_mask(mask, invert=True)
    assert inv[0, 0, 0].item() == 1.0
    assert inv[0, 50, 50].item() == 0.0

    # Grow
    grown = process_inpaint_mask(mask, grow=5)
    assert grown.sum() > mask.sum()

    # Blur
    blurred = process_inpaint_mask(mask, blur=5)
    assert 0.0 < blurred[0, 38, 38].item() < 1.0


def test_binary_inpaint_mask_resizing_remains_binary_when_blur_is_zero():
    # 50x50 binary mask (values in {0.0, 1.0})
    mask = torch.zeros((50, 50))
    mask[10:40, 10:40] = 1.0

    # Process mask with resizing to (100, 100) and blur=0
    res_mask = _process_inpaint_mask(mask=mask, invert=False, grow=0, blur=0, target_h=100, target_w=100)

    # Unique values must be subset of {0.0, 1.0}
    unique_vals = set(res_mask.unique().tolist())
    for val in unique_vals:
        assert val in (0.0, 1.0) or abs(val - 0.0) < 1e-6 or abs(val - 1.0) < 1e-6


def test_generate_krea2_latent_inpaint_mask_binary_interpolation():
    vae = MockVAE()
    base_img = torch.rand((1, 200, 200, 3))
    inpaint_mask = torch.zeros((200, 200))
    inpaint_mask[50:150, 50:150] = 1.0

    latent_dict = generate_krea2_latent(
        model=None,
        vae=vae,
        width=400,
        height=400,
        batch_size=1,
        latent_source="image",
        base_image=base_img,
        inpaint_mask=inpaint_mask,
        inpaint_mask_blur=0,
        sampling_resize_mode="crop"
    )

    assert "noise_mask" in latent_dict
    noise_mask = latent_dict["noise_mask"]
    unique_vals = set(noise_mask.unique().tolist())
    for val in unique_vals:
        assert val in (0.0, 1.0) or abs(val - 0.0) < 1e-6 or abs(val - 1.0) < 1e-6


def test_inpaint_mask_blur_kernel_device_dtype_and_zero_blur_skip():
    # Test float32
    m_f32 = torch.ones((1, 20, 20), dtype=torch.float32)
    res_f32 = _process_inpaint_mask(m_f32, invert=False, grow=0, blur=2, target_h=20, target_w=20)
    assert res_f32.device == m_f32.device
    assert res_f32.dtype == torch.float32

    # Test float64
    m_f64 = torch.ones((1, 20, 20), dtype=torch.float64)
    res_f64 = _process_inpaint_mask(m_f64, invert=False, grow=0, blur=2, target_h=20, target_w=20)
    assert res_f64.device == m_f64.device
    assert res_f64.dtype == torch.float64

    # Test blur=0 avoids convolution (returns identical values for un-grew un-resized)
    m_zero = torch.rand((1, 10, 10))
    res_zero = _process_inpaint_mask(m_zero, invert=False, grow=0, blur=0, target_h=10, target_w=10)
    assert torch.equal(res_zero, m_zero)


def test_attention_mask_hard_vs_soft():
    mask = torch.tensor([[0.2, 0.7], [0.4, 0.9]])

    # Hard threshold
    hard = process_attention_mask(mask, mode="hard")
    assert torch.equal(hard[0], torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    # Soft mode
    soft = process_attention_mask(mask, mode="soft")
    assert torch.allclose(soft[0], mask)


def test_ref_attention_bias_numerical_safety():
    device = torch.device("cpu")
    dtype = torch.float32

    # Boost = 1.0 -> No bias added (returns None)
    bias_off = _compute_ref_attention_bias_patchified(
        boosts=[1.0], txt_len=10, ref_token_lens=[64], tgt_len=64,
        ref_masks=[None], ref_token_grids=[(8, 8)], mask_modes=["hard"],
        device=device, dtype=dtype
    )
    assert bias_off is None

    # Boost > 1.0 (e.g. 2.5) -> Positive log bias
    bias_high = _compute_ref_attention_bias_patchified(
        boosts=[2.5], txt_len=10, ref_token_lens=[64], tgt_len=64,
        ref_masks=[None], ref_token_grids=[(8, 8)], mask_modes=["hard"],
        device=device, dtype=dtype
    )
    assert bias_high is not None
    assert not torch.isnan(bias_high).any()
    assert not torch.isinf(bias_high).any()
    expected_val = math.log(2.5)
    assert abs(bias_high[0, 0, 74, 10].item() - expected_val) < 1e-4

    # Boost < 1.0 (e.g. 0.5) -> Negative log bias
    bias_low = _compute_ref_attention_bias_patchified(
        boosts=[0.5], txt_len=10, ref_token_lens=[64], tgt_len=64,
        ref_masks=[None], ref_token_grids=[(8, 8)], mask_modes=["hard"],
        device=device, dtype=dtype
    )
    assert bias_low is not None
    assert not torch.isnan(bias_low).any()
    assert not torch.isinf(bias_low).any()
    assert bias_low[0, 0, 74, 10].item() < 0.0

    # Zero boost (0.0) -> Clamped to 1e-4, no log(0) -inf exception or NaN
    bias_zero = _compute_ref_attention_bias_patchified(
        boosts=[0.0], txt_len=10, ref_token_lens=[64], tgt_len=64,
        ref_masks=[None], ref_token_grids=[(8, 8)], mask_modes=["hard"],
        device=device, dtype=dtype
    )
    assert bias_zero is not None
    assert not torch.isnan(bias_zero).any()
    assert not torch.isinf(bias_zero).any()
