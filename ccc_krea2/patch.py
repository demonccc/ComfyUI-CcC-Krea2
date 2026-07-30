"""ModelPatcher extensions and DIFFUSION_MODEL wrapper closure for Krea 2 editing."""

import inspect
import math
from typing import List, Dict, Any, Optional, Tuple
import torch

from .constants import ReferenceRole
from .references import PreparedReference


def patch_krea2_model(
    model: Any,
    prepared_refs: List[PreparedReference],
    target_h: int,
    target_w: int
) -> Any:
    """Clone MODEL instance and register DIFFUSION_MODEL wrapper capturing reference closure."""
    patched_model = model.clone()

    def krea2_edit_wrapper(executor: Any, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, *wargs: Any, **kwargs: Any) -> torch.Tensor:
        """ComfyUI DIFFUSION_MODEL wrapper compliant with executor signature."""
        dit_model = getattr(executor, "class_obj", None)

        if prepared_refs:
            ref_latents = [ref.vae_latent for ref in prepared_refs if ref.vae_latent is not None]
            boosts = [ref.boost for ref in prepared_refs]
            ref_masks = [ref.token_attention_mask for ref in prepared_refs]

            # Compute additive attention logit bias
            attn_bias = _compute_ref_attention_bias(
                boosts=boosts,
                txt_len=context.shape[1] if context is not None else 0,
                ref_lens=[r.shape[1] * r.shape[2] for r in ref_latents],
                tgt_len=x.shape[2] * x.shape[3] if x.ndim == 4 else x.shape[1],
                ref_masks=ref_masks,
                device=x.device,
                dtype=x.dtype
            )

            if attn_bias is not None:
                kwargs["attn_bias"] = attn_bias

            # Forward ref_latents via kwargs or positional arguments to dit_model
            if dit_model is not None and hasattr(dit_model, "forward"):
                sig = inspect.signature(dit_model.forward)
                if "ref_latents" in sig.parameters:
                    kwargs["ref_latents"] = ref_latents
                elif "references" in sig.parameters:
                    kwargs["references"] = ref_latents

        # Call executor with current ComfyUI wrapper contract
        return executor(x, timesteps, context, *wargs, **kwargs)

    # Register wrapper via transformer_options or set_model_patch
    if not hasattr(patched_model, "model_options"):
        patched_model.model_options = {}

    options = patched_model.model_options.setdefault("transformer_options", {})
    wrappers = options.setdefault("wrappers", [])
    wrappers.append(krea2_edit_wrapper)

    return patched_model


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
