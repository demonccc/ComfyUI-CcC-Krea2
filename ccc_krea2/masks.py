"""Mask processing utilities for attention masks and inpainting noise masks."""

import torch
import torch.nn.functional as F
import scipy.ndimage
import numpy as np
from typing import Tuple, Optional


def process_inpaint_mask(
    mask: torch.Tensor,
    invert: bool = False,
    grow: int = 0,
    blur: int = 0
) -> torch.Tensor:
    """Process an inpainting mask tensor."""
    if mask is None:
        raise ValueError("Inpaint mask cannot be None.")

    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m.squeeze(1)

    if invert:
        m = 1.0 - m

    m = m.clamp(0.0, 1.0)

    if grow > 0:
        device = m.device
        dtype = m.dtype
        m_np = m.cpu().numpy()
        kernel_size = grow * 2 + 1
        struct = np.ones((kernel_size, kernel_size), dtype=bool)

        grown_batch = []
        for i in range(m_np.shape[0]):
            binary = m_np[i] > 0.5
            dilated = scipy.ndimage.binary_dilation(binary, structure=struct)
            grown_batch.append(dilated.astype(np.float32))

        m = torch.from_numpy(np.stack(grown_batch)).to(device=device, dtype=dtype)

    if blur > 0:
        kernel_size = blur * 2 + 1
        sigma = blur / 2.0
        x = torch.arange(kernel_size, dtype=m.dtype, device=m.device) - blur
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        m_4d = m.unsqueeze(1)
        padding = blur
        blurred = F.conv2d(m_4d, kernel_2d, padding=padding)
        m = blurred.squeeze(1).clamp(0.0, 1.0)

    return m


def process_attention_mask(
    mask: Optional[torch.Tensor],
    invert: bool = False,
    mode: str = "hard",
    token_grid: Optional[Tuple[int, int]] = None
) -> Optional[torch.Tensor]:
    """Process a reference attention mask tensor."""
    if mask is None:
        return None

    m = mask.float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m.squeeze(1)

    if invert:
        m = 1.0 - m

    m = m.clamp(0.0, 1.0)

    if mode == "hard":
        m = (m > 0.5).float()

    if token_grid is not None:
        gh, gw = token_grid
        if (m.shape[-2], m.shape[-1]) != (gh, gw):
            m_4d = m.unsqueeze(1)
            if mode == "hard":
                m_res = F.interpolate(m_4d, size=(gh, gw), mode="nearest")
            else:
                m_res = F.interpolate(m_4d, size=(gh, gw), mode="bicubic", antialias=True)
            m = m_res.squeeze(1).clamp(0.0, 1.0)
            if mode == "hard":
                m = (m > 0.5).float()

    return m
