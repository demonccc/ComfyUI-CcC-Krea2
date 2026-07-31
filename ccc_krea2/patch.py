"""Canonical ModelPatcher wrapper integration and DiT in-context forward execution."""

import math
from typing import List, Dict, Any, Optional, Tuple
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

    # Registration using ComfyUI patcher_extension.WrappersMP.DIFFUSION_MODEL
    _register_wrapper(patched_model, krea2_edit_wrapper)

    return patched_model


def _register_wrapper(patched_model: Any, wrapper: Any) -> None:
    """Register wrapper using ComfyUI WrappersMP.DIFFUSION_MODEL with API fallback."""
    registered = False

    wrapper_type = None
    try:
        import comfy.patcher_extension
        wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    except Exception:
        wrapper_type = "DIFFUSION_MODEL"

    if hasattr(patched_model, "add_wrapper_with_key"):
        try:
            patched_model.add_wrapper_with_key(wrapper_type, "ccc_krea2_edit", wrapper)
            registered = True
        except TypeError:
            try:
                patched_model.add_wrapper_with_key("ccc_krea2_edit", wrapper)
                registered = True
            except Exception:
                pass
        except Exception:
            pass

    if not registered:
        _fallback_options_register(patched_model, wrapper)


def _fallback_options_register(patched_model: Any, wrapper: Any) -> None:
    if not hasattr(patched_model, "model_options"):
        patched_model.model_options = {}
    options = patched_model.model_options.setdefault("transformer_options", {})
    wrappers = options.setdefault("wrappers", [])
    wrappers.append(wrapper)


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
    """Execute Krea2 SingleStreamDiT in-context forward pass with 3D RoPE position IDs and attention logit steering."""
    orig_ndim = x.ndim
    if orig_ndim == 5:
        x_4d = x.squeeze(2) if x.shape[2] == 1 else x[:, :, 0, :, :]
    else:
        x_4d = x

    bs, c, target_h, target_w = x_4d.shape
    txt_len = context.shape[1] if context is not None else 0

    ref_tokens_list: List[torch.Tensor] = []
    ref_lens: List[int] = []
    ref_pos_ids: List[torch.Tensor] = []

    for i, (ref_lat, meta) in enumerate(zip(ref_latents, ref_fit)):
        frame_idx = i + 1
        ref_h = ref_lat.shape[-2]
        ref_w = ref_lat.shape[-1]

        if hasattr(dit_model, "img_in"):
            ref_emb = dit_model.img_in(ref_lat)
        elif hasattr(dit_model, "x_embedder"):
            ref_emb = dit_model.x_embedder(ref_lat)
        else:
            ref_emb = ref_lat.flatten(2).transpose(1, 2)

        if ref_emb.ndim == 4:
            ref_emb = ref_emb.flatten(2).transpose(1, 2)

        ref_tokens_list.append(ref_emb)
        n_toks = ref_emb.shape[1]
        ref_lens.append(n_toks)

        y_off = meta.get("y_offset", 0.0) if meta else 0.0
        x_off = meta.get("x_offset", 0.0) if meta else 0.0
        pos_id = _build_ref_3d_rope_pos_ids(
            frame_idx=frame_idx,
            lat_h=ref_h,
            lat_w=ref_w,
            y_offset=y_off,
            x_offset=x_off,
            device=x.device
        )
        ref_pos_ids.append(pos_id)

    tgt_len = target_h * target_w

    attn_bias = _compute_ref_attention_bias(
        boosts=ref_boosts,
        txt_len=txt_len,
        ref_lens=ref_lens,
        tgt_len=tgt_len,
        ref_masks=ref_masks,
        device=x.device,
        dtype=x.dtype
    )

    if hasattr(dit_model, "forward"):
        try:
            out_tokens = dit_model.forward(
                x_4d,
                timesteps,
                context,
                ref_latents=ref_tokens_list,
                ref_pos_ids=ref_pos_ids,
                attn_bias=attn_bias,
                transformer_options=transformer_options
            )
            return _reshaped_target_output(out_tokens, x, orig_ndim, bs, c, target_h, target_w)
        except TypeError:
            pass

    out_tokens = dit_model(x_4d, timesteps, context)
    return _reshaped_target_output(out_tokens, x, orig_ndim, bs, c, target_h, target_w)


def _build_ref_3d_rope_pos_ids(
    frame_idx: int,
    lat_h: int,
    lat_w: int,
    y_offset: float,
    x_offset: float,
    device: torch.device
) -> torch.Tensor:
    """Build 3D RoPE position IDs [3, N] with fractional centering offsets."""
    grid_y = torch.arange(lat_h, device=device, dtype=torch.float32) + y_offset / 8.0
    grid_x = torch.arange(lat_w, device=device, dtype=torch.float32) + x_offset / 8.0

    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    mesh_t = torch.full_like(mesh_y, fill_value=float(frame_idx))

    pos_ids = torch.stack([mesh_t.flatten(), mesh_y.flatten(), mesh_x.flatten()], dim=0)
    return pos_ids


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


def _reshaped_target_output(
    out_tokens: torch.Tensor,
    orig_x: torch.Tensor,
    orig_ndim: int,
    bs: int,
    c: int,
    h: int,
    w: int
) -> torch.Tensor:
    """Return target token range reshaped back to 4D or 5D tensor matching original input shape."""
    if out_tokens.ndim == 3 and out_tokens.shape[1] != (h * w):
        out_tokens = out_tokens[:, - (h * w):, :]

    if out_tokens.ndim == 3:
        out_4d = out_tokens.transpose(1, 2).reshape(bs, c, h, w)
    elif out_tokens.ndim == 4:
        out_4d = out_tokens
    else:
        out_4d = orig_x

    if orig_ndim == 5:
        return out_4d.unsqueeze(2)

    return out_4d
