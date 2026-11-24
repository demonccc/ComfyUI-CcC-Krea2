"""Experimental Krea2 Edit reference K/V runtime.

Reuse Paint's isolated t=0 K/V machinery, while preserving Edit's reference
Boost/attention-mask semantics and configured RoPE positions. The joint path is
untouched when use_kv_reference=False.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .paint_patch import (
    _forward_with_registered_reference_kv,
    _precompute_registered_reference_kv,
    _reference_fingerprint,
)
from .patch import _normalize_runtime_list, _pad_to_patch_size


def _edit_kv_reference_bias(
    *,
    batch_size: int,
    txt_len: int,
    target_len: int,
    ref_grids: Sequence[Tuple[int, int]],
    boosts: Sequence[float],
    attention_maps: Sequence[Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build an additive attention bias for [text,target] queries and [text,target,refs] keys.

    Like joint Edit, only target -> reference attention is Boost-weighted.
    The isolated reference pass does not see the target or other references.
    """
    if len(ref_grids) != len(boosts) or len(ref_grids) != len(attention_maps):
        raise ValueError("[Krea2 CcC Edit K/V] Misaligned reference boosts or masks.")
    if not any(float(boost) != 0.0 or mask is not None for boost, mask in zip(boosts, attention_maps)):
        return None

    reference_len = sum(h * w for h, w in ref_grids)
    total_query_len = txt_len + target_len
    bias = torch.zeros(
        (batch_size, 1, total_query_len, total_query_len + reference_len),
        device=device, dtype=dtype,
    )
    offset = total_query_len
    for (height, width), boost, mask in zip(ref_grids, boosts, attention_maps):
        if not math.isfinite(float(boost)) or not -5.0 <= float(boost) <= 5.0:
            raise ValueError("[Krea2 CcC Edit K/V] Boost must be in [-5, 5].")
        span = height * width
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError("[Krea2 CcC Edit K/V] Attention mask must be [B,H,W].")
            weights = F.interpolate(
                mask.to(device=device, dtype=torch.float32).unsqueeze(1),
                size=(height, width), mode="area",
            )[:, 0].flatten(1)
            if weights.shape[0] == 1 and batch_size > 1:
                weights = weights.expand(batch_size, -1)
            if weights.shape[0] != batch_size:
                raise ValueError("[Krea2 CcC Edit K/V] Attention mask batch mismatch.")
            if not torch.isfinite(weights).all() or torch.any((weights < -5.0) | (weights > 5.0)):
                raise ValueError("[Krea2 CcC Edit K/V] Regional Boost must be in [-5, 5].")
            values = (weights * (math.log(10.0) / 5.0)).to(dtype=dtype)
            values = values.masked_fill(weights <= -5.0, float("-inf"))
        else:
            amount = float("-inf") if float(boost) == -5.0 else float(boost) * (math.log(10.0) / 5.0)
            values = torch.full((batch_size, span), amount, device=device, dtype=dtype)
        bias[:, :, txt_len:, offset:offset + span] = values[:, None, None, :]
        offset += span
    return bias


def krea2_dit_edit_kv_forward(
    *,
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: Sequence[torch.Tensor],
    ref_boosts: Optional[Sequence[float]] = None,
    ref_rope_positions: Optional[Sequence[str]] = None,
    ref_attention_maps: Optional[Sequence[Optional[torch.Tensor]]] = None,
    transformer_options: Optional[Dict[str, Any]] = None,
    kv_state: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Run Edit with isolated t=0 reference K/V, cached across denoising steps."""
    if not ref_latents:
        raise ValueError("[Krea2 CcC Edit K/V] Expected at least one visual reference.")
    transformer_options = dict(transformer_options or {})
    count = len(ref_latents)
    boosts = [float(v) for v in _normalize_runtime_list(ref_boosts, count, 0.0)]
    rope = [str(v) for v in _normalize_runtime_list(ref_rope_positions, count, "none")]
    maps = _normalize_runtime_list(ref_attention_maps, count, None)
    patch = getattr(dit_model, "patch", 2)
    if not isinstance(patch, int):
        patch = getattr(patch, "patch_size", 2)
    # Match the source shapes used by the K/V packing routine.
    grids: List[Tuple[int, int]] = []
    for ref in ref_latents:
        if ref.ndim not in (4, 5):
            raise ValueError("[Krea2 CcC Edit K/V] Reference must be 4D/5D.")
        grids.append((
            (ref.shape[-2] + patch - 1) // patch,
            (ref.shape[-1] + patch - 1) // patch,
        ))

    target_h = (x.shape[-2] + patch - 1) // patch
    target_w = (x.shape[-1] + patch - 1) // patch
    batch_size = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
    # Only determine token lengths here: the actual text sequence is unpacked
    # inside the Paint-style K/V forward, whose arguments are identical.
    unpacked_context = dit_model._unpack_context(context)
    txt_len = dit_model.txtmlp(
        dit_model.txtfusion(unpacked_context, mask=None, transformer_options=transformer_options)
    ).shape[1]
    bias = _edit_kv_reference_bias(
        batch_size=batch_size, txt_len=txt_len,
        target_len=target_h * target_w, ref_grids=grids,
        boosts=boosts, attention_maps=maps, device=x.device, dtype=x.dtype,
    )

    if kv_state is None:
        kv_state = {"last_timestep": None, "entries": {}}
    timestep = float(timesteps.max().detach().float().cpu())
    last = kv_state.get("last_timestep")
    if last is not None and timestep > float(last):
        kv_state["entries"].clear()
    kv_state["last_timestep"] = timestep
    key = (_reference_fingerprint(ref_latents, batch_size=batch_size, target_grid=(target_h, target_w)), tuple(rope))
    cached_kv = kv_state["entries"].get(key)
    if cached_kv is None:
        cached_kv = _precompute_registered_reference_kv(
            dit_model, x, timesteps, ref_latents, transformer_options,
            rope_positions=rope,
        )
        kv_state["entries"][key] = cached_kv

    return _forward_with_registered_reference_kv(
        dit_model, x, timesteps, context, cached_kv, transformer_options,
        reference_bias=bias,
    )
