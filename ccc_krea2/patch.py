"""Canonical ModelPatcher wrapper integration and custom Krea 2 edit DiT forward execution."""

import math
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange

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
    mask_modes: List[str] = []

    for ref in prepared_refs:
        if ref.vae_latent is not None:
            proc_lat = _process_latent_in_if_available(patched_model, ref.vae_latent)
            processed_ref_latents.append(proc_lat)
            ref_boosts.append(ref.boost)
            ref_masks.append(ref.spatial_attention_mask)
            mask_modes.append(ref.mask_mode)

    # Per-instance wrapper closure
    def krea2_edit_wrapper(executor: Any, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, *wargs: Any, **kwargs: Any) -> torch.Tensor:
        """Canonical ComfyUI DIFFUSION_MODEL wrapper signature."""
        dit_model = getattr(executor, "class_obj", None)

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
            mask_modes=mask_modes,
            transformer_options=transformer_options,
        )

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
    mask_modes: List[str],
    transformer_options: Dict[str, Any]
) -> torch.Tensor:
    """Faithful custom Krea 2 edit forward pass (patchifying, 3D RoPE token offsets & attention logit steering)."""
    orig_ndim = x.ndim
    if orig_ndim == 5:
        x_4d = x.squeeze(2) if x.shape[2] == 1 else x[:, :, 0, :, :]
    else:
        x_4d = x

    bs, c, target_h, target_w = x_4d.shape

    # Read patch_size from model
    patch_size = getattr(dit_model, "patch_size", 2)
    if hasattr(dit_model, "patch") and hasattr(dit_model.patch, "patch_size"):
        patch_size = dit_model.patch.patch_size

    # Pad target to patch_size
    pad_h = (patch_size - (target_h % patch_size)) % patch_size
    pad_w = (patch_size - (target_w % patch_size)) % patch_size
    if pad_h > 0 or pad_w > 0:
        x_padded = F.pad(x_4d, (0, pad_w, 0, pad_h))
    else:
        x_padded = x_4d

    padded_h, padded_w = x_padded.shape[-2], x_padded.shape[-1]
    target_gh = padded_h // patch_size
    target_gw = padded_w // patch_size
    tgt_n_toks = target_gh * target_gw

    # Patchify target
    x_patchified = rearrange(x_padded, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size)

    # Process and patchify references
    ref_tokens_list: List[torch.Tensor] = []
    ref_token_grids: List[Tuple[int, int]] = []
    ref_token_lens: List[int] = []

    for ref_lat in ref_latents:
        # Match batch size to target
        ref_lat_b = _match_batch_size(ref_lat, bs)

        rh, rw = ref_lat_b.shape[-2], ref_lat_b.shape[-1]
        r_pad_h = (patch_size - (rh % patch_size)) % patch_size
        r_pad_w = (patch_size - (rw % patch_size)) % patch_size
        if r_pad_h > 0 or r_pad_w > 0:
            ref_padded = F.pad(ref_lat_b, (0, r_pad_w, 0, r_pad_h))
        else:
            ref_padded = ref_lat_b

        r_ph, r_pw = ref_padded.shape[-2], ref_padded.shape[-1]
        r_gh = r_ph // patch_size
        r_gw = r_pw // patch_size

        ref_patch = rearrange(ref_padded, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size)
        ref_tokens_list.append(ref_patch)
        ref_token_grids.append((r_gh, r_gw))
        ref_token_lens.append(r_gh * r_gw)

    # Embed patchified target & references using model's input projection
    if hasattr(dit_model, "img_in"):
        target_emb = dit_model.img_in(x_patchified)
        ref_embs = [dit_model.img_in(r) for r in ref_tokens_list]
    elif hasattr(dit_model, "x_embedder"):
        target_emb = dit_model.x_embedder(x_patchified)
        ref_embs = [dit_model.x_embedder(r) for r in ref_tokens_list]
    else:
        target_emb = x_patchified
        ref_embs = ref_tokens_list

    # Text context embedding
    if hasattr(dit_model, "txt_in"):
        context_emb = dit_model.txt_in(context)
    else:
        context_emb = context

    txt_len = context_emb.shape[1] if context_emb is not None else 0

    # Build sequence: [text | ref_1 | ... | ref_N | target]
    seq_components = []
    if context_emb is not None:
        seq_components.append(context_emb)
    seq_components.extend(ref_embs)
    seq_components.append(target_emb)

    full_seq = torch.cat(seq_components, dim=1)

    # Build 3D RoPE position IDs for text (frame 0), references (frames 1..N), target (frame 0)
    rope_pos_ids = _build_incontext_3d_rope_pos_ids(
        txt_len=txt_len,
        ref_token_grids=ref_token_grids,
        target_grid=(target_gh, target_gw),
        device=x.device
    )

    # Build attention logit bias covering full sequence
    attn_bias = _compute_ref_attention_bias_patchified(
        boosts=ref_boosts,
        txt_len=txt_len,
        ref_token_lens=ref_token_lens,
        tgt_len=tgt_n_toks,
        ref_masks=ref_masks,
        ref_token_grids=ref_token_grids,
        mask_modes=mask_modes,
        device=x.device,
        dtype=x.dtype
    )

    # Timestep and text fusion projections
    vec_emb = None
    if hasattr(dit_model, "time_in") and timesteps is not None:
        t_emb = dit_model.time_in(timesteps)
        if hasattr(dit_model, "vector_in"):
            vec_emb = dit_model.vector_in(t_emb)

    # Run transformer blocks directly
    h_seq = full_seq
    blocks = getattr(dit_model, "blocks", getattr(dit_model, "double_blocks", getattr(dit_model, "layers", [])))

    for block in blocks:
        h_seq = _call_transformer_block(block, h_seq, vec_emb, rope_pos_ids, attn_bias, transformer_options)

    # Single blocks if present
    single_blocks = getattr(dit_model, "single_blocks", [])
    for block in single_blocks:
        h_seq = _call_transformer_block(block, h_seq, vec_emb, rope_pos_ids, attn_bias, transformer_options)

    # Final projection layer
    if hasattr(dit_model, "final_layer"):
        out_seq = dit_model.final_layer(h_seq)
    elif hasattr(dit_model, "out_proj"):
        out_seq = dit_model.out_proj(h_seq)
    else:
        out_seq = h_seq

    # Slice target tokens only
    tgt_tokens = out_seq[:, -tgt_n_toks:, :]

    # Unpatchify back to 4D tensor
    out_4d = rearrange(tgt_tokens, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=target_gh, w=target_gw, p1=patch_size, p2=patch_size)

    # Crop to original unpadded target dimensions
    out_cropped = out_4d[:, :, :target_h, :target_w]

    if orig_ndim == 5:
        return out_cropped.unsqueeze(2)

    return out_cropped


def _build_incontext_3d_rope_pos_ids(
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device
) -> torch.Tensor:
    """Build 3D RoPE position IDs [3, seq_len] with patchified token grid centering offsets."""
    tgt_gh, tgt_gw = target_grid
    list_pos = []

    # Text tokens: frame 0, y=0, x=0
    if txt_len > 0:
        txt_pos = torch.zeros((3, txt_len), device=device, dtype=torch.float32)
        list_pos.append(txt_pos)

    # Reference tokens: frame 1..N, centered relative to target grid
    for i, (r_gh, r_gw) in enumerate(ref_token_grids):
        frame_idx = i + 1
        y_off = (tgt_gh - r_gh) / 2.0
        x_off = (tgt_gw - r_gw) / 2.0

        grid_y = torch.arange(r_gh, device=device, dtype=torch.float32) + y_off
        grid_x = torch.arange(r_gw, device=device, dtype=torch.float32) + x_off

        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        mesh_t = torch.full_like(mesh_y, fill_value=float(frame_idx))

        ref_pos = torch.stack([mesh_t.flatten(), mesh_y.flatten(), mesh_x.flatten()], dim=0)
        list_pos.append(ref_pos)

    # Target tokens: frame 0, y=0..tgt_gh, x=0..tgt_gw
    tgt_y = torch.arange(tgt_gh, device=device, dtype=torch.float32)
    tgt_x = torch.arange(tgt_gw, device=device, dtype=torch.float32)
    mesh_ty, mesh_tx = torch.meshgrid(tgt_y, tgt_x, indexing="ij")
    mesh_tt = torch.zeros_like(mesh_ty)

    tgt_pos = torch.stack([mesh_tt.flatten(), mesh_ty.flatten(), mesh_tx.flatten()], dim=0)
    list_pos.append(tgt_pos)

    return torch.cat(list_pos, dim=1)


def _compute_ref_attention_bias_patchified(
    boosts: List[float],
    txt_len: int,
    ref_token_lens: List[int],
    tgt_len: int,
    ref_masks: List[Optional[torch.Tensor]],
    ref_token_grids: List[Tuple[int, int]],
    mask_modes: List[str],
    device: torch.device,
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Compute additive attention logit bias covering full sequence with token-grid mask alignment."""
    if not boosts or all(b == 1.0 and m is None for b, m in zip(boosts, ref_masks)):
        return None

    total_ref_len = sum(ref_token_lens)
    seq_len = txt_len + total_ref_len + tgt_len

    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)

    ref_start = txt_len
    target_start = txt_len + total_ref_len

    for boost, ref_len, spatial_mask, (r_gh, r_gw), mask_mode in zip(
        boosts, ref_token_lens, ref_masks, ref_token_grids, mask_modes
    ):
        ref_end = ref_start + ref_len

        # Boost applied from target queries to reference keys
        if boost != 1.0:
            safe_boost = max(1e-4, min(100.0, float(boost)))
            b_val = math.log(safe_boost)
            bias[0, 0, target_start:, ref_start:ref_end] += b_val

        # Resize spatial mask directly to patchified token grid (r_gh, r_gw)
        if spatial_mask is not None:
            m_bchw = spatial_mask.float()
            if m_bchw.ndim == 2:
                m_bchw = m_bchw.unsqueeze(0).unsqueeze(0)
            elif m_bchw.ndim == 3:
                m_bchw = m_bchw.unsqueeze(1)

            m_resized = F.interpolate(m_bchw, size=(r_gh, r_gw), mode="bicubic", antialias=True).squeeze()

            if mask_mode == "hard":
                m_processed = (m_resized > 0.5).float()
            else:  # soft
                m_processed = m_resized.clamp(0.0, 1.0)

            m_flat = m_processed.reshape(-1).to(device=device, dtype=dtype)
            if m_flat.numel() == ref_len:
                m_bias = (1.0 - m_flat) * -1e4
                bias[0, 0, target_start:, ref_start:ref_end] += m_bias.unsqueeze(0)

        ref_start = ref_end

    return torch.nan_to_num(bias, nan=0.0, posinf=100.0, neginf=-100.0)


def _call_transformer_block(
    block: Any,
    h: torch.Tensor,
    vec_emb: Optional[torch.Tensor],
    rope_pos_ids: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    transformer_options: Dict[str, Any]
) -> torch.Tensor:
    """Execute a single transformer block attempting common block signatures."""
    try:
        return block(h, vec_emb=vec_emb, rope_pos_ids=rope_pos_ids, attn_bias=attn_bias, transformer_options=transformer_options)
    except TypeError:
        try:
            return block(h, vec_emb, attn_bias=attn_bias)
        except TypeError:
            try:
                return block(h, attn_bias=attn_bias)
            except TypeError:
                return block(h)


def _match_batch_size(tensor: torch.Tensor, target_bs: int) -> torch.Tensor:
    """Repeat or trim tensor along batch dimension to match target_bs."""
    curr_b = tensor.shape[0]
    if curr_b == target_bs:
        return tensor
    elif curr_b > target_bs:
        return tensor[:target_bs]
    else:
        repeats = (target_bs + curr_b - 1) // curr_b
        tiled = tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))
        return tiled[:target_bs]
