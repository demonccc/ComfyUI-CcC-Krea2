"""Explicit ModelPatcher patch wrapper for SingleStreamDiT forward pass."""

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Any, List, Optional, Tuple, Dict

from .constants import LOGGER_PREFIX

try:
    import comfy.patcher_extension
    import comfy.ldm.common_dit
    from comfy.ldm.flux.layers import timestep_embedding
except ImportError:
    comfy = None


def _build_imgids_offset(
    bs: int,
    frame_idx: int,
    gh: int,
    gw: int,
    th: int,
    tw: int,
    device: torch.device
) -> torch.Tensor:
    """Builds a 3D position embedding grid (frame, row, col) for a reference latent."""
    off_h = max(0, (th - gh) // 2)
    off_w = max(0, (tw - gw) // 2)
    ids = torch.zeros((gh, gw, 3), device=device, dtype=torch.float32)
    ids[..., 0] = float(frame_idx)
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _compute_ref_attention_bias(
    boosts: List[float],
    txt_len: int,
    ref_lens: List[int],
    tgt_len: int,
    ref_masks: List[Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Computes an additive attention logit bias tensor for reference-guided attention steering.

    Numerical safety features:
    - Safe math.log: clamps boost to [1e-4, 100.0] preventing log(0) and -inf
    - Nan-to-num protection: eliminates any NaN/Inf logit artifacts
    - Handles boosts < 1.0 (loosening), = 1.0 (off), and > 1.0 (pulling towards reference)
    """
    if not boosts or (all(b == 1.0 for b in boosts) and not any(m is not None for m in ref_masks)):
        return None

    offsets = [txt_len]
    for rl in ref_lens:
        offsets.append(offsets[-1] + rl)

    tgt_start = offsets[-1]
    total_len = tgt_start + tgt_len

    bias = torch.zeros((1, 1, total_len, total_len), device=device, dtype=dtype)

    for i, boost in enumerate(boosts):
        mask = ref_masks[i] if i < len(ref_masks) else None
        # Clamp boost to prevent log(0) and numerical instability
        safe_b = max(1e-4, min(float(boost), 100.0))
        b_val = math.log(safe_b)

        ref_start = offsets[i]
        ref_end = offsets[i + 1]
        ref_len = ref_lens[i]

        if mask is not None:
            m_flat = mask.to(device=device, dtype=dtype).reshape(-1)
            if m_flat.numel() == ref_len:
                w_bias = m_flat.unsqueeze(0) * b_val
                bias[:, :, tgt_start:, ref_start:ref_end] += w_bias.unsqueeze(0).unsqueeze(0)
                continue

        if safe_b != 1.0:
            bias[:, :, tgt_start:, ref_start:ref_end] += b_val

    # Ensure no NaN or Inf values pass into attention calculation
    bias = torch.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
    return bias


def krea2_dit_incontext_forward(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: List[torch.Tensor],
    ref_boosts: List[float],
    ref_masks: List[Optional[torch.Tensor]],
    ref_fit: List[bool],
    transformer_options: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """In-context SingleStreamDiT forward pass logic."""
    if transformer_options is None:
        transformer_options = {}

    bs, c, H_orig, W_orig = x.shape
    patch = getattr(dit_model, "patch", 2)

    x_padded = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x_padded.shape[-2], x_padded.shape[-1]
    h_, w_ = H // patch, W // patch

    src_imgs = []
    src_lens = []
    src_grids = []

    for i, source in enumerate(ref_latents):
        src = source.to(device=x.device, dtype=x.dtype)
        if src.ndim == 5:
            sb, sc, st, sh, sw = src.shape
            src = src.reshape(sb * st, sc, sh, sw)
        if src.shape[0] != bs:
            src = src[:1].repeat(bs, *([1] * (src.ndim - 1)))

        src_padded = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))
        gh = src_padded.shape[-2] // patch
        gw = src_padded.shape[-1] // patch

        src_patch = rearrange(src_padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        src_emb = dit_model.first(src_patch)

        src_imgs.append(src_emb)
        src_lens.append(src_emb.shape[1])
        src_grids.append((gh, gw))

    if hasattr(dit_model, "_unpack_context"):
        context = dit_model._unpack_context(context)

    tgt_patch = rearrange(x_padded, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    tgt_emb = dit_model.first(tgt_patch)

    t = dit_model.tmlp(timestep_embedding(timesteps, dit_model.tdim).unsqueeze(1).to(tgt_emb.dtype))
    tvec = dit_model.tproj(t)

    context = dit_model.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = dit_model.txtmlp(context)

    txt_len = context.shape[1]
    tgt_len = tgt_emb.shape[1]

    combined = torch.cat([context] + src_imgs + [tgt_emb], dim=1)
    device = combined.device

    txtpos = torch.zeros((bs, txt_len, 3), device=device, dtype=torch.float32)
    srcpos = [
        _build_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
        for i, (gh, gw) in enumerate(src_grids)
    ]

    imgids = torch.zeros((h_, w_, 3), device=device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    tgtpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)

    pos = torch.cat([txtpos] + srcpos + [tgtpos], dim=1)
    freqs = dit_model.pe_embedder(pos)

    attn_bias = _compute_ref_attention_bias(
        boosts=ref_boosts,
        txt_len=txt_len,
        ref_lens=src_lens,
        tgt_len=tgt_len,
        ref_masks=ref_masks,
        device=device,
        dtype=combined.dtype
    )

    for block in dit_model.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = dit_model.last(combined, t)
    total_ref_len = sum(src_lens)

    out = final[:, txt_len + total_ref_len:txt_len + total_ref_len + tgt_len, :]
    out = rearrange(
        out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h_, w=w_, ph=patch, pw=patch, c=dit_model.channels
    )
    out = out[:, :, :H_orig, :W_orig]
    return out


def create_krea2_model_wrapper(apply_model: Any, args: Any) -> torch.Tensor:
    """Wrapper invoked by ComfyUI ModelPatcher during sampling."""
    input_x = args.input
    timestep = args.timestep
    c = args.c
    transformer_options = getattr(args, "transformer_options", {})

    ref_latents = c.get("reference_latents", None)
    if not ref_latents:
        return apply_model(*args)

    ref_boosts = c.get("reference_boosts", [1.0] * len(ref_latents))
    ref_masks = c.get("reference_masks", [None] * len(ref_latents))
    ref_fit = c.get("reference_fit", [True] * len(ref_latents))

    dit_model = transformer_options.get("inner_model", None)
    if dit_model is None:
        model_obj = getattr(apply_model, "__self__", None)
        if model_obj is not None:
            dit_model = getattr(model_obj, "diffusion_model", None)

    if dit_model is None or not hasattr(dit_model, "pe_embedder"):
        return apply_model(*args)

    context = c.get("c_crossattn", None)
    if context is None:
        context = c.get("context", None)

    return krea2_dit_incontext_forward(
        dit_model=dit_model,
        x=input_x,
        timesteps=timestep,
        context=context,
        ref_latents=ref_latents,
        ref_boosts=ref_boosts,
        ref_masks=ref_masks,
        ref_fit=ref_fit,
        transformer_options=transformer_options
    )


def patch_krea2_model(model: Any) -> Any:
    """Clones the input MODEL and attaches a per-instance DIFFUSION_MODEL wrapper."""
    if model is None:
        raise ValueError(f"{LOGGER_PREFIX} Input MODEL is None.")

    m = model.clone()
    to = m.model_options.setdefault("transformer_options", {})

    if comfy is not None and hasattr(comfy, "patcher_extension"):
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "ccc_krea2_edit",
            create_krea2_model_wrapper,
            to
        )
    else:
        wrappers = to.setdefault("wrappers", [])
        wrappers.append(create_krea2_model_wrapper)

    return m
