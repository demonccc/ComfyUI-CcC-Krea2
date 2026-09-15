"""Krea 2 Identity Edit runtime using conditioning-transported reference latents.

The transport and forward semantics intentionally follow the proven RedNode / Krea2Moodboard
Identity Edit v1.2 contract: reference latents, fit flags and boosts travel with CONDITIONING,
so positive and negative passes can carry the same references while using different boosts.
CcC extends only the RoPE placement of already-fitted references.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def _conditioning_set_values(conditioning: List[Any], values: Dict[str, Any]) -> List[Any]:
    """Set conditioning metadata with a lightweight fallback for isolated tests."""
    if not conditioning:
        return conditioning
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


def attach_reference_boosts_to_conditioning(
    conditioning: List[Any],
    reference_boosts: List[float],
    reference_masked_boosts: Optional[List[float]] = None,
) -> List[Any]:
    """Attach pass-specific reference boost metadata."""
    values: Dict[str, Any] = {"reference_boosts": [float(v) for v in reference_boosts]}
    if reference_masked_boosts is not None:
        values["reference_masked_boosts"] = [float(v) for v in reference_masked_boosts]
    return _conditioning_set_values(conditioning, values)


def attach_reference_runtime_to_conditioning(
    conditioning: List[Any],
    reference_count: int,
    rope_positions: Optional[List[str]] = None,
    reference_boosts: Optional[List[float]] = None,
) -> List[Any]:
    """Attach RedNode-compatible fit metadata plus the CcC RoPE extension.

    `reference_latents` themselves are attached by the edit orchestrator. This helper adds the
    remaining per-pass controls without changing the public node surface.
    """
    if reference_count <= 0:
        return conditioning
    values: Dict[str, Any] = {
        "reference_fit": [True] * reference_count,
        "reference_rope_positions": list(rope_positions or ["none"] * reference_count),
    }
    if reference_boosts is not None and any(float(v) != 1.0 for v in reference_boosts):
        values["reference_boosts"] = [float(v) for v in reference_boosts]
    return _conditioning_set_values(conditioning, values)


def install_krea2_reference_conditioning() -> None:
    """Teach ComfyUI Krea2 to forward reference metadata from CONDITIONING to the DiT wrapper."""
    try:
        import comfy.conds
        import comfy.model_base
    except (ImportError, AttributeError):
        return

    krea2_cls = getattr(comfy.model_base, "Krea2", None)
    if krea2_cls is None or getattr(krea2_cls, "_ccc_krea2_reference_conditioning_patched", False):
        return

    original_extra_conds = krea2_cls.extra_conds
    original_extra_shapes = getattr(krea2_cls, "extra_conds_shapes", None)

    def _ccc_krea2_extra_conds(self, **kwargs):
        out = original_extra_conds(self, **kwargs)

        ref_latents = kwargs.get("reference_latents")
        if ref_latents is not None:
            out["ref_latents"] = comfy.conds.CONDList([self.process_latent_in(lat) for lat in ref_latents])

        ref_boosts = kwargs.get("reference_boosts")
        if ref_boosts is not None:
            out["ref_boosts"] = comfy.conds.CONDConstant(list(ref_boosts))

        ref_fit = kwargs.get("reference_fit")
        if ref_fit is not None:
            out["ref_fit"] = comfy.conds.CONDConstant(list(ref_fit))

        rope_positions = kwargs.get("reference_rope_positions")
        if rope_positions is not None:
            out["ccc_ref_rope_positions"] = comfy.conds.CONDConstant(list(rope_positions))

        return out

    krea2_cls.extra_conds = _ccc_krea2_extra_conds

    if original_extra_shapes is not None:

        def _ccc_krea2_extra_conds_shapes(self, **kwargs):
            out = original_extra_shapes(self, **kwargs)
            ref_latents = kwargs.get("reference_latents")
            if ref_latents is not None:
                out["ref_latents"] = [1, 16, sum(math.prod(lat.size()[2:]) for lat in ref_latents)]
            return out

        krea2_cls.extra_conds_shapes = _ccc_krea2_extra_conds_shapes

    krea2_cls._ccc_krea2_reference_conditioning_patched = True


def _normalize_runtime_list(value: Any, count: int, default: Any) -> List[Any]:
    if count <= 0:
        return []
    if value is None:
        return [default] * count
    if torch.is_tensor(value):
        raw = value.detach().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]
    if len(raw) < count:
        raw = [default] * (count - len(raw)) + raw
    elif len(raw) > count:
        raw = raw[-count:]
    return raw


def _normalize_runtime_boosts(value: Any, count: int, default: float = 1.0) -> List[float]:
    """Backward-compatible float specialization used by existing tests/helpers."""
    return [float(v) for v in _normalize_runtime_list(value, count, default)]


def is_model_already_patched(model: Any, patch_key: str = "ccc_krea2_edit") -> bool:
    """Check whether a model patcher already carries a CcC edit runtime marker/wrapper."""
    if getattr(model, "_ccc_patch_key", None) == patch_key:
        return True
    if hasattr(model, "wrappers"):
        wrappers = getattr(model, "wrappers", {})
        if isinstance(wrappers, dict):
            for value in wrappers.values():
                if isinstance(value, dict) and patch_key in value:
                    return True
                if isinstance(value, list):
                    for wrapper in value:
                        if getattr(wrapper, "wrapper_key", "") == patch_key:
                            return True
    if hasattr(model, "model_options"):
        options = getattr(model, "model_options", {})
        wrappers = options.get("transformer_options", {}).get("wrappers", {}) if isinstance(options, dict) else {}
        if isinstance(wrappers, dict):
            for value in wrappers.values():
                if isinstance(value, dict) and patch_key in value:
                    return True
    return False


def check_patch_safety(model: Any, target_patch: str) -> None:
    if target_patch in ("krea2_edit", "ccc_krea2_edit") and is_model_already_patched(model, "ccc_ostris_edit"):
        raise ValueError("[CcC Krea2] Cannot apply Krea2 Edit on a model already patched for Ostris Edit.")
    if target_patch in ("ostris_edit", "ccc_ostris_edit") and is_model_already_patched(model, "ccc_krea2_edit"):
        raise ValueError("[CcC Krea2] Cannot apply Ostris Edit on a model already patched for Krea2 Edit.")


def patch_krea2_model(model: Any, prepared_refs: Optional[List[Any]] = None) -> Any:
    """Clone MODEL and register a runtime wrapper that consumes refs from CONDITIONING.

    `prepared_refs` is retained only for call compatibility. References are deliberately not
    captured in the MODEL anymore; this is the key RedNode-compatible behavior.
    """
    if is_model_already_patched(model, "ccc_krea2_edit"):
        raise RuntimeError(
            "[CcC Krea2] Input MODEL is already patched by CcC Krea2 Edit. Connect Edit to the unpatched upstream MODEL."
        )

    install_krea2_reference_conditioning()
    patched_model = model.clone()
    patched_model._ccc_patch_key = "ccc_krea2_edit"

    def krea2_edit_wrapper(
        executor: Any,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *wargs: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        dit_model = getattr(executor, "class_obj", None)

        # ComfyUI signature drift: older cores call
        #   (..., attention_mask, transformer_options)
        # while newer Krea2 cores may call
        #   (..., attention_mask, ref_latents, transformer_options).
        # Prefer explicit conditioning kwargs, otherwise recover the positional ref list.
        ref_latents = kwargs.get("ref_latents")
        if ref_latents is None:
            drift = list(wargs)
            if drift and isinstance(drift[-1], dict):
                drift.pop()
            if len(drift) >= 2 and isinstance(drift[1], (list, tuple)):
                candidate = drift[1]
                if all(torch.is_tensor(item) for item in candidate):
                    ref_latents = list(candidate)
        ref_latents = list(ref_latents or [])

        if dit_model is None or not ref_latents:
            fallback_kwargs = dict(kwargs)
            for key in ("ref_latents", "ref_boosts", "ref_fit", "ccc_ref_rope_positions"):
                fallback_kwargs.pop(key, None)
            return executor(x, timesteps, context, *wargs, **fallback_kwargs)

        transformer_options = kwargs.get("transformer_options")
        if transformer_options is None and wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        if transformer_options is None:
            transformer_options = {}

        count = len(ref_latents)
        boosts = [float(v) for v in _normalize_runtime_list(kwargs.get("ref_boosts"), count, 1.0)]
        fit_flags = [bool(v) for v in _normalize_runtime_list(kwargs.get("ref_fit"), count, False)]
        rope_positions = [
            str(v)
            for v in _normalize_runtime_list(kwargs.get("ccc_ref_rope_positions"), count, "none")
        ]

        return krea2_dit_incontext_forward(
            dit_model=dit_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=ref_latents,
            ref_boosts=boosts,
            ref_fit=fit_flags,
            ref_rope_positions=rope_positions,
            transformer_options=transformer_options,
        )

    _register_wrapper(patched_model, krea2_edit_wrapper)
    return patched_model


def _register_wrapper(patched_model: Any, wrapper: Any) -> None:
    registered = False
    wrapper_type: Any = "diffusion_model"
    try:
        import comfy.patcher_extension

        wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    except Exception:
        pass

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
        if not hasattr(patched_model, "model_options"):
            patched_model.model_options = {}
        wrappers = patched_model.model_options.setdefault("transformer_options", {}).setdefault("wrappers", {})
        key = str(wrapper_type)
        value = wrappers.setdefault(key, {})
        if isinstance(value, dict):
            value["ccc_krea2_edit"] = [wrapper]
        elif isinstance(value, list):
            value.append(wrapper)


def _pad_to_patch_size(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    try:
        from comfy.ldm.common_dit import pad_to_patch_size

        return pad_to_patch_size(tensor, (patch_size, patch_size), padding_mode="replicate")
    except (ImportError, AttributeError):
        h, w = tensor.shape[-2:]
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else tensor


def _repeat_to_batch_size(tensor: torch.Tensor, target_bs: int) -> torch.Tensor:
    try:
        from comfy.utils import repeat_to_batch_size

        return repeat_to_batch_size(tensor, target_bs)
    except (ImportError, AttributeError):
        if tensor.shape[0] == target_bs:
            return tensor
        if tensor.shape[0] > target_bs:
            return tensor[:target_bs]
        return tensor[:1].expand(target_bs, *tensor.shape[1:])


def _fit_latent(src: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """RedNode legacy fallback: crop to target AR then resize in latent space."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (height, width):
        return src
    scale = max(height / sh, width / sw)
    crop_h = min(sh, int(round(height / scale)))
    crop_w = min(sw, int(round(width / scale)))
    y0 = (sh - crop_h) // 2
    x0 = (sw - crop_w) // 2
    src = src[..., y0:y0 + crop_h, x0:x0 + crop_w]
    return F.interpolate(src.float(), size=(height, width), mode="bilinear")


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    try:
        from comfy.ldm.flux.layers import timestep_embedding

        return timestep_embedding(timesteps, dim, max_period=max_period)
    except (ImportError, AttributeError):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) if dim % 2 else emb


def _compute_ref_attention_bias_patchified(
    boosts: List[float],
    txt_len: int,
    ref_token_lens: List[int],
    tgt_len: int,
    ref_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ref_token_grids: Optional[List[Tuple[int, int]]] = None,
    mask_modes: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    masked_boosts: Optional[List[float]] = None,
) -> Optional[torch.Tensor]:
    """Target->reference attention bias with legacy optional mask compatibility.

    The split public Edit surface currently supplies unmasked refs, so its runtime path reduces
    exactly to RedNode's per-reference log(boost) bias. The mask branch is retained for internal
    compatibility and tests.
    """
    if not boosts:
        return None
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    ref_masks = list(ref_masks or [None] * len(boosts))
    ref_token_grids = list(ref_token_grids or [(1, n) for n in ref_token_lens])
    mask_modes = list(mask_modes or ["hard"] * len(boosts))

    offsets = [txt_len]
    for length in ref_token_lens:
        offsets.append(offsets[-1] + length)
    target_start = offsets[-1]
    seq_len = target_start + tgt_len
    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)

    any_effect = False
    for index, boost in enumerate(boosts):
        spatial_mask = ref_masks[index] if index < len(ref_masks) else None
        ref_start = offsets[index]
        ref_end = ref_start + ref_token_lens[index]
        boost_log = math.log(max(float(boost), 1e-4))

        if spatial_mask is None:
            if float(boost) != 1.0:
                bias[:, :, target_start:, ref_start:ref_end] = boost_log
                any_effect = True
            continue

        if boost_log == 0.0:
            continue
        gh, gw = ref_token_grids[index]
        mask = spatial_mask[:1].float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=(gh, gw), mode="nearest")[0, 0]
        if mask_modes[index] == "hard":
            mask = (mask > 0.5).float()
        else:
            mask = mask.clamp(0.0, 1.0)
        flat = mask.reshape(-1).to(device=device, dtype=dtype)
        if flat.numel() == ref_token_lens[index]:
            bias[:, :, target_start:, ref_start:ref_end] += boost_log * flat.view(1, 1, 1, -1)
            any_effect = True

    return bias if any_effect else None


def _build_incontext_3d_rope_pos_ids(
    batch_size: int,
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device,
    ref_rope_positions: Optional[List[str]] = None,
) -> torch.Tensor:
    """Default RedNode-compatible centered positions; modular Edit can replace this helper."""
    tgt_h, tgt_w = target_grid
    parts = [torch.zeros((txt_len, 3), device=device, dtype=torch.float32)] if txt_len else []
    for index, (ref_h, ref_w) in enumerate(ref_token_grids):
        y0 = float(max(0, (tgt_h - ref_h) // 2))
        x0 = float(max(0, (tgt_w - ref_w) // 2))
        ys = torch.arange(ref_h, device=device, dtype=torch.float32) + y0
        xs = torch.arange(ref_w, device=device, dtype=torch.float32) + x0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        tt = torch.full_like(yy, float(index + 1))
        parts.append(torch.stack([tt.flatten(), yy.flatten(), xx.flatten()], dim=-1))
    ys = torch.arange(tgt_h, device=device, dtype=torch.float32)
    xs = torch.arange(tgt_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    parts.append(torch.stack([torch.zeros_like(yy).flatten(), yy.flatten(), xx.flatten()], dim=-1))
    return torch.cat(parts, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)


def krea2_dit_incontext_forward(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: List[torch.Tensor],
    ref_boosts: Optional[List[float]] = None,
    transformer_options: Optional[Dict[str, Any]] = None,
    ref_fit: Optional[List[bool]] = None,
    ref_rope_positions: Optional[List[str]] = None,
    **_legacy: Any,
) -> torch.Tensor:
    """RedNode-compatible in-context Krea2 forward with optional CcC RoPE displacement."""
    transformer_options = transformer_options or {}
    n_refs = len(ref_latents)
    boosts = [float(v) for v in _normalize_runtime_list(ref_boosts, n_refs, 1.0)]
    fit_flags = [bool(v) for v in _normalize_runtime_list(ref_fit, n_refs, True)]
    rope_positions = [str(v) for v in _normalize_runtime_list(ref_rope_positions, n_refs, "none")]

    temporal = x.ndim == 5
    if temporal:
        batch5, channels5, frames5, height5, width5 = x.shape
        x = x.reshape(batch5 * frames5, channels5, height5, width5)

    batch, _, original_h, original_w = x.shape
    patch_size = getattr(dit_model, "patch", 2)
    if not isinstance(patch_size, int):
        patch_size = getattr(patch_size, "patch_size", 2)

    x = _pad_to_patch_size(x, patch_size)
    height, width = x.shape[-2:]
    target_gh, target_gw = height // patch_size, width // patch_size

    sources = []
    for index, source in enumerate(ref_latents):
        src = source.to(device=x.device, dtype=x.dtype)
        if src.ndim == 5:
            sb, sc, st, sh, sw = src.shape
            src = src.reshape(sb * st, sc, sh, sw)
        src = _repeat_to_batch_size(src, batch)
        keep_own_grid = fit_flags[index] and src.shape[-2] <= height and src.shape[-1] <= width
        if src.shape[-2:] != (height, width) and not keep_own_grid:
            src = _fit_latent(src, height, width).to(x.dtype)
        sources.append(_pad_to_patch_size(src, patch_size))

    context = dit_model._unpack_context(context)
    target_tokens = rearrange(
        x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size
    )
    target_tokens = dit_model.first(target_tokens)
    source_tokens = [
        dit_model.first(
            rearrange(src, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        )
        for src in sources
    ]

    t_emb = _timestep_embedding(timesteps, getattr(dit_model, "tdim", 256)).unsqueeze(1).to(target_tokens.dtype)
    t = dit_model.tmlp(t_emb)
    tvec = dit_model.tproj(t)

    context = dit_model.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = dit_model.txtmlp(context)

    txt_len = context.shape[1]
    target_len = target_tokens.shape[1]
    source_lens = [tokens.shape[1] for tokens in source_tokens]
    source_len = sum(source_lens)
    combined = torch.cat([context] + source_tokens + [target_tokens], dim=1)

    source_grids = [(src.shape[-2] // patch_size, src.shape[-1] // patch_size) for src in sources]
    pos = _build_incontext_3d_rope_pos_ids(
        batch_size=batch,
        txt_len=txt_len,
        ref_token_grids=source_grids,
        target_grid=(target_gh, target_gw),
        ref_rope_positions=rope_positions,
        device=combined.device,
    )
    freqs = dit_model.pe_embedder(pos)

    attn_bias = _compute_ref_attention_bias_patchified(
        boosts=boosts,
        txt_len=txt_len,
        ref_token_lens=source_lens,
        tgt_len=target_len,
        device=combined.device,
        dtype=combined.dtype,
    )

    total_blocks = len(dit_model.blocks)
    total_ref_len = source_len
    for block_index, block in enumerate(dit_model.blocks):
        # Preserve the current ComfyUI Krea2 transformer metadata contract while keeping the
        # Identity Edit sequence order [text | refs | target]. This matters for attention patches
        # that consume img_slice/block_index but does not change RedNode reference semantics.
        block_options = transformer_options.copy()
        block_options["total_blocks"] = total_blocks
        block_options["block_type"] = "single"
        block_options["img_slice"] = [slice(txt_len + total_ref_len, None)]
        block_options["block_index"] = block_index
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=block_options)

    final = dit_model.last(combined, t)
    output = final[:, txt_len + source_len:txt_len + source_len + target_len]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=target_gh,
        w=target_gw,
        ph=patch_size,
        pw=patch_size,
        c=dit_model.channels,
    )
    output = output[:, :, :original_h, :original_w]

    if temporal:
        output = output.reshape(batch5, frames5, dit_model.channels, original_h, original_w).movedim(1, 2)
    return output
