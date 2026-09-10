"""Canonical ModelPatcher wrapper integration and exact Krea 2 SingleStreamDiT edit forward execution."""

import math
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from .references import PreparedReference, _process_latent_in_if_available


def install_krea2_reference_boost_conditioning() -> None:
    """Allow per-conditioning reference boosts, matching the proven Krea2 Identity Edit contract.

    Positive and negative conditioning are evaluated separately by ComfyUI. Transporting boosts
    through conditioning metadata lets the positive pass use the configured reference fidelity
    while the grounded negative pass stays at 1.0, as used by the v1.2 Identity Edit workflow.
    """
    try:
        import comfy.conds
        import comfy.model_base
    except (ImportError, AttributeError):
        return

    krea2_cls = getattr(comfy.model_base, "Krea2", None)
    if krea2_cls is None or getattr(krea2_cls, "_ccc_krea2_reference_boost_patched", False):
        return

    original_extra_conds = krea2_cls.extra_conds

    def _ccc_krea2_extra_conds(self, **kwargs):
        out = original_extra_conds(self, **kwargs)

        ref_boosts = kwargs.get("reference_boosts")
        if ref_boosts is not None:
            out["ref_boosts"] = comfy.conds.CONDConstant(list(ref_boosts))

        masked_boosts = kwargs.get("reference_masked_boosts")
        if masked_boosts is not None:
            out["ccc_ref_masked_boosts"] = comfy.conds.CONDConstant(list(masked_boosts))

        return out

    krea2_cls.extra_conds = _ccc_krea2_extra_conds
    krea2_cls._ccc_krea2_reference_boost_patched = True


def attach_reference_boosts_to_conditioning(
    conditioning: List[Any],
    reference_boosts: List[float],
    reference_masked_boosts: Optional[List[float]] = None,
) -> List[Any]:
    """Attach pass-specific reference boost metadata to a conditioning object."""
    if not conditioning:
        return conditioning

    values = {"reference_boosts": [float(v) for v in reference_boosts]}
    if reference_masked_boosts is not None:
        values["reference_masked_boosts"] = [float(v) for v in reference_masked_boosts]

    try:
        import node_helpers

        return node_helpers.conditioning_set_values(conditioning, values)
    except (ImportError, AttributeError):
        updated = []
        for entry in conditioning:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                extras = dict(entry[1]) if isinstance(entry[1], dict) else {}
                extras.update(values)
                updated.append([entry[0], extras])
            else:
                updated.append(entry)
        return updated


def _normalize_runtime_boosts(value: Any, count: int, default: float = 1.0) -> List[float]:
    """Normalize condition-provided boost values to the active visual-reference count."""
    if count <= 0:
        return []
    if value is None:
        return [float(default)] * count

    if torch.is_tensor(value):
        raw = value.detach().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]

    resolved = [float(v) for v in raw]
    if len(resolved) < count:
        resolved = [float(default)] * (count - len(resolved)) + resolved
    elif len(resolved) > count:
        resolved = resolved[-count:]
    return resolved


def is_model_already_patched(model: Any, patch_key: str = "ccc_krea2_edit") -> bool:
    """Check if model object has already been patched with the given wrapper key."""
    if getattr(model, "_ccc_patch_key", None) == patch_key:
        return True

    if hasattr(model, "wrappers"):
        w_dict = getattr(model, "wrappers", {})
        if isinstance(w_dict, dict):
            for _, wrappers_list in w_dict.items():
                if isinstance(wrappers_list, dict) and patch_key in wrappers_list:
                    return True
                if isinstance(wrappers_list, list):
                    for w in wrappers_list:
                        if getattr(w, "__name__", "") == patch_key or getattr(w, "wrapper_key", "") == patch_key:
                            return True

    if hasattr(model, "model_options"):
        opts = getattr(model, "model_options", {})
        if isinstance(opts, dict):
            t_opts = opts.get("transformer_options", {})
            if isinstance(t_opts, dict):
                wrappers = t_opts.get("wrappers", {})
                if isinstance(wrappers, dict):
                    for _, diff_w in wrappers.items():
                        if isinstance(diff_w, dict) and patch_key in diff_w:
                            return True

    return False


def check_patch_safety(model: Any, target_patch: str) -> None:
    """Ensure model does not contain conflicting or incompatible patch wrappers."""
    if target_patch in ("krea2_edit", "ccc_krea2_edit"):
        if is_model_already_patched(model, "ccc_ostris_edit"):
            raise ValueError(
                "[CcC Krea2] Conflict detected: Cannot apply 'krea2_edit' patch to a model already patched with 'ccc_ostris_edit'."
            )
    elif target_patch in ("ostris_edit", "ccc_ostris_edit"):
        if is_model_already_patched(model, "ccc_krea2_edit"):
            raise ValueError(
                "[CcC Krea2] Conflict detected: Cannot apply 'ostris_edit' patch to a model already patched with 'ccc_krea2_edit'."
            )


def patch_krea2_model(model: Any, prepared_refs: List[PreparedReference]) -> Any:
    """Clone MODEL and register the Krea2 Edit DIFFUSION_MODEL wrapper.

    Reference geometry and latents are captured by the wrapper. Attention boosts are deliberately
    *not* captured as a single global value: they arrive per conditioning pass so the positive can
    use the configured boost while the grounded negative remains at 1.0.
    """
    if is_model_already_patched(model, "ccc_krea2_edit"):
        raise RuntimeError(
            "[CcC Krea2] Input MODEL is already patched by CcC Krea2 Edit. "
            "Chaining CcC Edit nodes is unsupported because each wrapper captures reference "
            "latents in a closure. Connect this Edit node to the unpatched upstream MODEL instead."
        )

    install_krea2_reference_boost_conditioning()

    patched_model = model.clone()
    patched_model._ccc_patch_key = "ccc_krea2_edit"

    processed_ref_latents: List[torch.Tensor] = []
    ref_masks: List[Optional[torch.Tensor]] = []
    mask_modes: List[str] = []
    ref_rope_positions: List[str] = []

    for ref in prepared_refs:
        if ref.vae_latent is not None:
            proc_lat = _process_latent_in_if_available(patched_model, ref.vae_latent)
            processed_ref_latents.append(proc_lat)
            ref_masks.append(ref.spatial_attention_mask)
            mask_modes.append(ref.mask_mode)
            ref_rope_positions.append(getattr(ref, "rope_position", "none"))

    def krea2_edit_wrapper(
        executor: Any,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *wargs: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Canonical ComfyUI DIFFUSION_MODEL wrapper signature."""
        dit_model = getattr(executor, "class_obj", None)

        transformer_options = {}
        if wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        elif "transformer_options" in kwargs:
            transformer_options = kwargs["transformer_options"]

        if not processed_ref_latents or dit_model is None:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("ref_boosts", None)
            fallback_kwargs.pop("ccc_ref_masked_boosts", None)
            return executor(x, timesteps, context, *wargs, **fallback_kwargs)

        count = len(processed_ref_latents)
        runtime_ref_boosts = _normalize_runtime_boosts(kwargs.get("ref_boosts"), count, default=1.0)
        runtime_masked_boosts = _normalize_runtime_boosts(
            kwargs.get("ccc_ref_masked_boosts"), count, default=1.0
        )

        return krea2_dit_incontext_forward(
            dit_model=dit_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=processed_ref_latents,
            ref_boosts=runtime_ref_boosts,
            ref_masked_boosts=runtime_masked_boosts,
            ref_masks=ref_masks,
            mask_modes=mask_modes,
            ref_rope_positions=ref_rope_positions,
            transformer_options=transformer_options,
        )

    _register_wrapper(patched_model, krea2_edit_wrapper)

    return patched_model


def _register_wrapper(patched_model: Any, wrapper: Any) -> None:
    """Register wrapper using ComfyUI WrappersMP.DIFFUSION_MODEL ("diffusion_model") with API fallback."""
    registered = False

    wrapper_type = "diffusion_model"
    try:
        import comfy.patcher_extension

        wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    except Exception:
        wrapper_type = "diffusion_model"

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
        _fallback_options_register(patched_model, wrapper, wrapper_type)


def _fallback_options_register(patched_model: Any, wrapper: Any, wrapper_type: Any = "diffusion_model") -> None:
    """Fallback options registration using nested dictionary structure."""
    if not hasattr(patched_model, "model_options"):
        patched_model.model_options = {}

    options = patched_model.model_options

    t_options = options.setdefault("transformer_options", {})
    wrappers = t_options.setdefault("wrappers", {})

    w_key = str(wrapper_type)
    if isinstance(wrappers, dict):
        diff_wrappers = wrappers.setdefault(w_key, {})
        if isinstance(diff_wrappers, dict):
            existing = diff_wrappers.get("ccc_krea2_edit")
            if existing is None:
                diff_wrappers["ccc_krea2_edit"] = [wrapper]
            elif isinstance(existing, list):
                existing.append(wrapper)
            else:
                diff_wrappers["ccc_krea2_edit"] = [existing, wrapper]
        elif isinstance(diff_wrappers, list):
            diff_wrappers.append(wrapper)


def _pad_to_patch_size(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad 4D tensor spatial dimensions to multiples of patch_size using replicate padding."""
    try:
        from comfy.ldm.common_dit import pad_to_patch_size

        return pad_to_patch_size(tensor, (patch_size, patch_size), padding_mode="replicate")
    except (ImportError, AttributeError):
        h, w = tensor.shape[-2], tensor.shape[-1]
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        if pad_h > 0 or pad_w > 0:
            return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
        return tensor


def _repeat_to_batch_size(tensor: torch.Tensor, target_bs: int) -> torch.Tensor:
    """Repeat or trim tensor along batch dimension to match target_bs."""
    try:
        from comfy.utils import repeat_to_batch_size

        return repeat_to_batch_size(tensor, target_bs)
    except (ImportError, AttributeError):
        curr_b = tensor.shape[0]
        if curr_b == target_bs:
            return tensor
        if curr_b > target_bs:
            return tensor[:target_bs]

        repeats = (target_bs + curr_b - 1) // curr_b
        tiled = tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))
        return tiled[:target_bs]


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Compute sinusoidal timestep embeddings with ComfyUI fallback."""
    try:
        from comfy.ldm.flux.layers import timestep_embedding

        return timestep_embedding(timesteps, dim, max_period=max_period)
    except (ImportError, AttributeError):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


def krea2_dit_incontext_forward(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: List[torch.Tensor],
    ref_boosts: List[float],
    ref_masks: List[Optional[torch.Tensor]],
    mask_modes: List[str],
    transformer_options: Dict[str, Any],
    ref_masked_boosts: Optional[List[float]] = None,
    ref_rope_positions: Optional[List[str]] = None,
) -> torch.Tensor:
    """Execute Krea 2 SingleStreamDiT edit forward using exact model member API and signatures."""
    orig_ndim = x.ndim
    if orig_ndim == 5:
        b_orig, c_orig, t_orig, h_orig, w_orig = x.shape
        x_4d = rearrange(x, "b c t h w -> (b t) c h w")
    else:
        x_4d = x
        t_orig = 1

    bs, c, target_h, target_w = x_4d.shape

    patch_size = getattr(dit_model, "patch", 2)
    if not isinstance(patch_size, int):
        patch_size = getattr(patch_size, "patch_size", 2)

    channels = getattr(dit_model, "channels", c)

    orig_tgt_h, orig_tgt_w = target_h, target_w

    x_padded = _pad_to_patch_size(x_4d, patch_size)
    padded_h, padded_w = x_padded.shape[-2], x_padded.shape[-1]

    target_gh = padded_h // patch_size
    target_gw = padded_w // patch_size
    tgt_n_toks = target_gh * target_gw

    x_patch = rearrange(
        x_padded,
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )

    ref_patches: List[torch.Tensor] = []
    ref_token_grids: List[Tuple[int, int]] = []
    ref_token_lens: List[int] = []

    for ref_lat in ref_latents:
        if ref_lat.ndim == 5:
            ref_lat = rearrange(ref_lat, "b c t h w -> (b t) c h w")

        ref_lat = ref_lat.to(device=x.device, dtype=x.dtype)
        ref_lat_b = _repeat_to_batch_size(ref_lat, bs)
        ref_padded = _pad_to_patch_size(ref_lat_b, patch_size)

        r_ph, r_pw = ref_padded.shape[-2], ref_padded.shape[-1]
        r_gh = r_ph // patch_size
        r_gw = r_pw // patch_size

        r_patch = rearrange(
            ref_padded,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=patch_size,
            p2=patch_size,
        )
        ref_patches.append(r_patch)
        ref_token_grids.append((r_gh, r_gw))
        ref_token_lens.append(r_gh * r_gw)

    ctx = dit_model._unpack_context(context)
    ctx = dit_model.txtfusion(ctx, mask=None, transformer_options=transformer_options)
    ctx = dit_model.txtmlp(ctx)

    txt_len = ctx.shape[1] if ctx is not None else 0

    target_emb = dit_model.first(x_patch)
    ref_embs = [dit_model.first(rp) for rp in ref_patches]

    seq_components = []
    if ctx is not None:
        seq_components.append(ctx)
    seq_components.extend(ref_embs)
    seq_components.append(target_emb)

    full_seq = torch.cat(seq_components, dim=1)

    rope_pos_ids = _build_incontext_3d_rope_pos_ids(
        batch_size=bs,
        txt_len=txt_len,
        ref_token_grids=ref_token_grids,
        target_grid=(target_gh, target_gw),
        ref_rope_positions=ref_rope_positions,
        device=x.device,
    )

    freqs = dit_model.pe_embedder(rope_pos_ids) if hasattr(dit_model, "pe_embedder") else None

    attn_bias = _compute_ref_attention_bias_patchified(
        boosts=ref_boosts,
        masked_boosts=ref_masked_boosts or [1.0] * len(ref_boosts),
        txt_len=txt_len,
        ref_token_lens=ref_token_lens,
        tgt_len=tgt_n_toks,
        ref_masks=ref_masks,
        ref_token_grids=ref_token_grids,
        mask_modes=mask_modes,
        device=x.device,
        dtype=x.dtype,
    )

    tdim = getattr(dit_model, "tdim", 256)
    t_emb_val = _timestep_embedding(timesteps, tdim).unsqueeze(1).to(x.dtype)
    t = dit_model.tmlp(t_emb_val)
    tvec = dit_model.tproj(t)

    h_seq = full_seq
    blocks = getattr(dit_model, "blocks", [])
    total_blocks = len(blocks)
    total_ref_len = sum(ref_token_lens)

    for i, block in enumerate(blocks):
        t_opts = transformer_options.copy()
        t_opts["total_blocks"] = total_blocks
        t_opts["block_type"] = "single"
        t_opts["img_slice"] = [slice(txt_len + total_ref_len, None)]
        t_opts["block_index"] = i

        h_seq = block(h_seq, tvec, freqs, attn_bias, transformer_options=t_opts)

    out_seq = dit_model.last(h_seq, t) if hasattr(dit_model, "last") else h_seq

    tgt_tokens = out_seq[:, -tgt_n_toks:, :]

    out_4d = rearrange(
        tgt_tokens,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=target_gh,
        w=target_gw,
        p1=patch_size,
        p2=patch_size,
        c=channels,
    )

    out_cropped = out_4d[:, :, :orig_tgt_h, :orig_tgt_w]

    if orig_ndim == 5:
        return rearrange(out_cropped, "(b t) c h w -> b c t h w", t=t_orig)

    return out_cropped


def _build_incontext_3d_rope_pos_ids(
    batch_size: int,
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device,
    ref_rope_positions: Optional[List[str]] = None,
) -> torch.Tensor:
    """Build legacy 3D RoPE position IDs with shape [batch_size, seq_len, 3].

    The modular Edit surface replaces this helper at import time with the three-axis
    implementation in modular_nodes.rope_position.
    """
    tgt_gh, tgt_gw = target_grid
    list_pos = []

    if txt_len > 0:
        txt_pos = torch.zeros((txt_len, 3), device=device, dtype=torch.float32)
        list_pos.append(txt_pos)

    for i, (r_gh, r_gw) in enumerate(ref_token_grids):
        frame_idx = i + 1
        position = (
            ref_rope_positions[i]
            if ref_rope_positions is not None and i < len(ref_rope_positions)
            else "none"
        )
        y_off = (tgt_gh - r_gh) / 2.0
        x_off = (tgt_gw - r_gw) / 2.0
        if position == "up":
            y_off = -float(r_gh)
        elif position == "down":
            y_off = float(tgt_gh)
        elif position == "left":
            x_off = -float(r_gw)
        elif position == "right":
            x_off = float(tgt_gw)
        elif position != "none":
            raise ValueError(
                f"Invalid reference RoPE position '{position}'. Expected none, up, down, left, or right."
            )

        grid_y = torch.arange(r_gh, device=device, dtype=torch.float32) + y_off
        grid_x = torch.arange(r_gw, device=device, dtype=torch.float32) + x_off

        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        mesh_t = torch.full_like(mesh_y, fill_value=float(frame_idx))

        ref_pos = torch.stack([mesh_t.flatten(), mesh_y.flatten(), mesh_x.flatten()], dim=-1)
        list_pos.append(ref_pos)

    tgt_y = torch.arange(tgt_gh, device=device, dtype=torch.float32)
    tgt_x = torch.arange(tgt_gw, device=device, dtype=torch.float32)
    mesh_ty, mesh_tx = torch.meshgrid(tgt_y, tgt_x, indexing="ij")
    mesh_tt = torch.zeros_like(mesh_ty)

    tgt_pos = torch.stack([mesh_tt.flatten(), mesh_ty.flatten(), mesh_tx.flatten()], dim=-1)
    list_pos.append(tgt_pos)

    seq_pos = torch.cat(list_pos, dim=0)
    return seq_pos.unsqueeze(0).repeat(batch_size, 1, 1)


def _compute_ref_attention_bias_patchified(
    boosts: List[float],
    txt_len: int,
    ref_token_lens: List[int],
    tgt_len: int,
    ref_masks: List[Optional[torch.Tensor]],
    ref_token_grids: List[Tuple[int, int]],
    mask_modes: List[str],
    device: torch.device,
    dtype: torch.dtype,
    masked_boosts: Optional[List[float]] = None,
) -> Optional[torch.Tensor]:
    """Compute additive attention logit bias covering full sequence."""
    resolved_base_boosts = []
    resolved_masked_boosts = []

    for i in range(len(boosts)):
        b = boosts[i]
        m = ref_masks[i] if i < len(ref_masks) else None

        if masked_boosts is not None and i < len(masked_boosts):
            resolved_base_boosts.append(b)
            resolved_masked_boosts.append(masked_boosts[i])
        else:
            if m is not None:
                resolved_base_boosts.append(1.0)
                resolved_masked_boosts.append(b)
            else:
                resolved_base_boosts.append(b)
                resolved_masked_boosts.append(1.0)

    if not boosts or all(
        b == 1.0 and mb == 1.0 and m is None
        for b, mb, m in zip(resolved_base_boosts, resolved_masked_boosts, ref_masks)
    ):
        return None

    total_ref_len = sum(ref_token_lens)
    seq_len = txt_len + total_ref_len + tgt_len

    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)

    ref_start = txt_len
    target_start = txt_len + total_ref_len

    for boost, masked_boost, ref_len, spatial_mask, (r_gh, r_gw), mask_mode in zip(
        resolved_base_boosts,
        resolved_masked_boosts,
        ref_token_lens,
        ref_masks,
        ref_token_grids,
        mask_modes,
    ):
        ref_end = ref_start + ref_len

        safe_base_boost = max(1e-4, min(100.0, float(boost)))
        base_bias = math.log(safe_base_boost)

        safe_masked_boost = max(1e-4, min(100.0, float(masked_boost)))
        masked_extra_bias = math.log(safe_masked_boost)

        if base_bias != 0.0:
            bias[0, 0, target_start:, ref_start:ref_end] += base_bias

        if spatial_mask is not None and masked_extra_bias != 0.0:
            m_bchw = spatial_mask[:1].float()
            if m_bchw.ndim == 2:
                m_bchw = m_bchw.unsqueeze(0).unsqueeze(0)
            elif m_bchw.ndim == 3:
                m_bchw = m_bchw.unsqueeze(0)

            m_resized = F.interpolate(m_bchw, size=(r_gh, r_gw), mode="nearest")
            m_2d = m_resized[0, 0]

            if mask_mode == "hard":
                m_processed = (m_2d > 0.5).float()
            else:
                m_processed = m_2d.clamp(0.0, 1.0)

            m_flat = m_processed.reshape(-1).to(device=device, dtype=dtype)
            if m_flat.numel() == ref_len:
                selected_bias = masked_extra_bias * m_flat
                bias[0, 0, target_start:, ref_start:ref_end] += selected_bias.unsqueeze(0)

        ref_start = ref_end

    return torch.nan_to_num(bias, nan=0.0, posinf=100.0, neginf=-100.0)
