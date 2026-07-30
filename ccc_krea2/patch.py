"""Canonical ModelPatcher wrapper integration and DiT in-context forward execution."""

import math
from typing import List, Dict, Any, Optional
import torch

from .references import PreparedReference, _process_latent_in_if_available


def patch_krea2_model(model: Any, prepared_refs: List[PreparedReference]) -> Any:
    """Clone MODEL and register canonical DIFFUSION_MODEL wrapper with closure transport.

    Contract: patch_krea2_model(model, prepared_refs)
    """
    patched_model = model.clone()

    # Extract VAE latents and apply process_latent_in EXACTLY ONCE
    processed_ref_latents: List[torch.Tensor] = []
    ref_boosts: List[float] = []
    ref_masks: List[Optional[torch.Tensor]] = []
    ref_fit: List[Optional[Dict[str, Any]]] = []

    for ref in prepared_refs:
        if ref.vae_latent is not None:
            proc_lat = _process_latent_in_if_available(patched_model, ref.vae_latent)
            processed_ref_latents.append(proc_lat)
            ref_boosts.append(ref.boost)
            ref_masks.append(ref.token_attention_mask)
            ref_fit.append(ref.ref_fit_meta)

    # Per-instance wrapper closure
    def krea2_edit_wrapper(executor: Any, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, *wargs: Any, **kwargs: Any) -> torch.Tensor:
        """Canonical ComfyUI DIFFUSION_MODEL wrapper signature."""
        dit_model = getattr(executor, "class_obj", None)

        # Extract transformer_options from wargs or kwargs
        transformer_options = {}
        if wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        elif "transformer_options" in kwargs:
            transformer_options = kwargs["transformer_options"]

        if not processed_ref_latents or dit_model is None:
            return executor(x, timesteps, context, *wargs, **kwargs)

        return krea2_dit_incontext_forward(
            dit_model=dit_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=processed_ref_latents,
            ref_boosts=ref_boosts,
            ref_masks=ref_masks,
            ref_fit=ref_fit,
            transformer_options=transformer_options,
        )

    # Register closure using ModelPatcher API or transformer_options wrappers
    if hasattr(patched_model, "add_wrapper_with_key"):
        patched_model.add_wrapper_with_key("ccc_krea2_edit", krea2_edit_wrapper)
    else:
        if not hasattr(patched_model, "model_options"):
            patched_model.model_options = {}
        options = patched_model.model_options.setdefault("transformer_options", {})
        wrappers = options.setdefault("wrappers", [])
        wrappers.append(krea2_edit_wrapper)

    return patched_model


def krea2_dit_incontext_forward(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: List[torch.Tensor],
    ref_boosts: List[float],
    ref_masks: List[Optional[torch.Tensor]],
    ref_fit: List[Optional[Dict[str, Any]]],
    transformer_options: Dict[str, Any]
) -> torch.Tensor:
    """Execute Krea2 SingleStreamDiT in-context forward pass with attention steering."""
    ref_lens = [r.shape[-2] * r.shape[-1] for r in ref_latents]
    tgt_len = x.shape[-2] * x.shape[-1] if x.ndim == 4 else x.shape[1]
    txt_len = context.shape[1] if context is not None else 0

    attn_bias = _compute_ref_attention_bias(
        boosts=ref_boosts,
        txt_len=txt_len,
        ref_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=ref_masks,
        device=x.device,
        dtype=x.dtype
    )

    # If dit_model has native forward expecting ref_latents / attn_bias
    if hasattr(dit_model, "forward"):
        try:
            return dit_model.forward(
                x,
                timesteps,
                context,
                ref_latents=ref_latents,
                attn_bias=attn_bias,
                transformer_options=transformer_options
            )
        except TypeError:
            pass

    return dit_model(x, timesteps, context)


def _compute_ref_attention_bias(
    boosts: List[float],
    txt_len: int,
    ref_lens: List[int],
    tgt_len: int,
    ref_masks: List[Optional[torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Compute additive attention logit bias with numerical safety clamping."""
    if not boosts or all(b == 1.0 and m is None for b, m in zip(boosts, ref_masks)):
        return None

    total_ref_len = sum(ref_lens)
    seq_len = txt_len + total_ref_len + tgt_len

    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)

    ref_start = txt_len
    target_start = txt_len + total_ref_len

    for boost, ref_len, mask in zip(boosts, ref_lens, ref_masks):
        ref_end = ref_start + ref_len

        if boost != 1.0:
            safe_boost = max(1e-4, min(100.0, float(boost)))
            b_val = math.log(safe_boost)
            bias[0, 0, target_start:, ref_start:ref_end] += b_val

        if mask is not None:
            m_flat = mask.reshape(-1)
            if m_flat.numel() == ref_len:
                m_bias = (1.0 - m_flat.to(device=device, dtype=dtype)) * -1e4
                bias[0, 0, target_start:, ref_start:ref_end] += m_bias.unsqueeze(0)

        ref_start = ref_end

    return torch.nan_to_num(bias, nan=0.0, posinf=100.0, neginf=-100.0)
