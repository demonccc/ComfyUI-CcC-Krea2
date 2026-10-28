"""Krea2 CcC Edit runtime using conditioning-transported reference latents.

Prepared reference latents, source/target attention masks, boosts and RoPE placement travel with CONDITIONING.
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


def attach_reference_runtime_to_conditioning(
    conditioning: List[Any],
    reference_count: int,
    rope_positions: Optional[List[str]] = None,
    reference_boosts: Optional[List[float]] = None,
    reference_target_masks: Optional[List[Any]] = None,
    reference_attention_scopes: Optional[List[str]] = None,
) -> List[Any]:
    """Attach RoPE, boosts and target masks for ordered visual references."""
    if reference_count <= 0:
        return conditioning
    values: Dict[str, Any] = {
        "reference_rope_positions": list(rope_positions or ["none"] * reference_count),
    }
    if reference_boosts is not None and any(float(v) != 1.0 for v in reference_boosts):
        values["reference_boosts"] = [float(v) for v in reference_boosts]
    if reference_target_masks is not None and any(mask is not None for mask in reference_target_masks):
        values["reference_target_masks"] = list(reference_target_masks)
    if reference_attention_scopes is not None:
        values["reference_attention_scopes"] = list(reference_attention_scopes)
    return _conditioning_set_values(conditioning, values)


def _prepare_reference_latent_for_model(model: Any, latent: torch.Tensor) -> torch.Tensor:
    """Normalize an image reference to Krea2/Wan's 5D latent contract before process_latent_in.

    VAE image encoders commonly return B,C,H,W. Krea2 uses a Wan latent format whose channel
    normalization tensors are B,C,T,H,W. Passing a 4D image latent directly into that
    normalization can broadcast the channel axis into a bogus temporal dimension. Image
    references therefore get an explicit singleton T axis before model latent scaling.
    Existing 5D references are preserved unchanged.
    """
    if not torch.is_tensor(latent):
        raise TypeError(
            f"[Krea2 CcC Edit] Reference latent must be a torch.Tensor, got {type(latent).__name__}."
        )
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    elif latent.ndim != 5:
        raise ValueError(
            "[Krea2 CcC Edit] Reference latent must be B,C,H,W or B,C,T,H,W; "
            f"received shape {tuple(latent.shape)}."
        )
    return model.process_latent_in(latent)


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
            out["ref_latents"] = comfy.conds.CONDList(
                [_prepare_reference_latent_for_model(self, lat) for lat in ref_latents]
            )

        ref_boosts = kwargs.get("reference_boosts")
        if ref_boosts is not None:
            out["ref_boosts"] = comfy.conds.CONDConstant(list(ref_boosts))

        rope_positions = kwargs.get("reference_rope_positions")
        if rope_positions is not None:
            out["ccc_ref_rope_positions"] = comfy.conds.CONDConstant(list(rope_positions))

        target_masks = kwargs.get("reference_target_masks")
        if target_masks is not None:
            out["ccc_ref_target_masks"] = comfy.conds.CONDConstant(list(target_masks))

        source_masks = kwargs.get("reference_boost_masks")
        if source_masks is not None:
            out["ccc_ref_boost_masks"] = comfy.conds.CONDConstant(list(source_masks))

        attention_maps = kwargs.get("reference_attention_maps")
        if attention_maps is not None:
            out["ccc_ref_attention_maps"] = comfy.conds.CONDConstant(list(attention_maps))

        attention_scopes = kwargs.get("reference_attention_scopes")
        if attention_scopes is not None:
            out["ccc_ref_attention_scopes"] = comfy.conds.CONDConstant(list(attention_scopes))

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


def is_model_already_patched(model: Any, patch_key: str = "ccc_krea2_edit") -> bool:
    """Check whether a model patcher already carries a Krea2 CcC Edit runtime marker/wrapper."""
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


def patch_krea2_model(model: Any) -> Any:
    """Clone MODEL and register the Krea2 CcC Edit runtime wrapper that consumes references from CONDITIONING."""
    if is_model_already_patched(model, "ccc_krea2_edit"):
        raise RuntimeError(
            "[Krea2 CcC Edit] Input MODEL is already patched by Krea2 CcC Edit. Connect Edit to the unpatched upstream MODEL."
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
            for key in (
                "ref_latents", "ref_boosts", "ccc_ref_rope_positions",
                "ccc_ref_target_masks", "ccc_ref_attention_scopes", "ccc_ref_boost_masks",
            ):
                fallback_kwargs.pop(key, None)
            return executor(x, timesteps, context, *wargs, **fallback_kwargs)

        transformer_options = kwargs.get("transformer_options")
        if transformer_options is None and wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        if transformer_options is None:
            transformer_options = {}

        count = len(ref_latents)
        boosts = [float(v) for v in _normalize_runtime_list(kwargs.get("ref_boosts"), count, 1.0)]
        rope_positions = [
            str(v)
            for v in _normalize_runtime_list(kwargs.get("ccc_ref_rope_positions"), count, "none")
        ]
        target_masks = _normalize_runtime_list(kwargs.get("ccc_ref_target_masks"), count, None)
        source_masks = _normalize_runtime_list(kwargs.get("ccc_ref_boost_masks"), count, None)
        attention_maps = _normalize_runtime_list(kwargs.get("ccc_ref_attention_maps"), count, None)
        attention_scopes = [
            str(v)
            for v in _normalize_runtime_list(kwargs.get("ccc_ref_attention_scopes"), count, "global")
        ]

        return krea2_dit_incontext_forward(
            dit_model=dit_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=ref_latents,
            ref_boosts=boosts,
            ref_rope_positions=rope_positions,
            ref_target_masks=target_masks,
            ref_masks=source_masks,
            ref_attention_maps=attention_maps,
            ref_attention_scopes=attention_scopes,
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


def _mask_to_flat(
    spatial_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Project an IMAGE-space mask onto its actual DiT token grid, per batch item."""
    if spatial_mask.ndim == 2:
        spatial_mask = spatial_mask.unsqueeze(0)
    if spatial_mask.ndim != 3:
        raise ValueError(f"[Krea2 CcC Edit] Attention mask must be H,W or B,H,W, got {tuple(spatial_mask.shape)}.")
    if spatial_mask.shape[0] not in (1, batch_size):
        raise ValueError(
            f"[Krea2 CcC Edit] Attention mask batch {spatial_mask.shape[0]} "
            f"does not match target batch {batch_size}."
        )
    # Area projection conserves how much of each token is selected. Original
    # Krea2Edit uses a hard > 0.5 threshold for source ref_boost_mask.
    values = F.interpolate(
        spatial_mask.to(device=device, dtype=torch.float32).unsqueeze(1),
        size=(grid_h, grid_w),
        mode="area",
    )[:, 0]
    return (values > 0.5).to(dtype=dtype).reshape(values.shape[0], -1).expand(batch_size, -1)


def _compute_ref_attention_bias_patchified(
    boosts: List[float],
    txt_len: int,
    ref_token_lens: List[int],
    tgt_len: int,
    ref_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ref_token_grids: Optional[List[Tuple[int, int]]] = None,
    ref_attention_maps: Optional[List[Optional[torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    ref_target_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ref_attention_scopes: Optional[List[str]] = None,
    target_grid: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
) -> Optional[torch.Tensor]:
    """Boost selected reference keys for queries inside the selected target mask.

    No mask means the complete reference (or, for target masks, the complete
    canvas). Selection is independent of RoPE positions. Each batch member
    keeps its own masks and token attention scores.
    """
    if not boosts:
        return None
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    ref_masks = list(ref_masks or [None] * len(boosts))
    ref_attention_maps = list(ref_attention_maps or [None] * len(boosts))
    ref_target_masks = list(ref_target_masks or [None] * len(boosts))
    ref_token_grids = list(ref_token_grids or [(1, n) for n in ref_token_lens])
    ref_attention_scopes = list(ref_attention_scopes or ["global"] * len(boosts))
    if not (len(ref_token_lens) == len(ref_token_grids) == len(boosts)):
        raise ValueError("[Krea2 CcC Edit] Reference tokens, grids and boosts must be aligned.")

    active = any(
        float(boost) != 1.0 or scope == "only in mask" or ref_attention_maps[i] is not None
        for i, (boost, scope) in enumerate(zip(boosts, ref_attention_scopes))
    )
    if not active:
        return None

    offsets = [txt_len]
    for length in ref_token_lens:
        offsets.append(offsets[-1] + length)
    target_start = offsets[-1]
    seq_len = target_start + tgt_len
    bias = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype)

    for index, boost in enumerate(boosts):
        scope = ref_attention_scopes[index]
        if scope not in ("global", "boost in mask", "only in mask"):
            raise ValueError(f"[Krea2 CcC Edit] Invalid reference attention scope '{scope}'.")
        source_mask = ref_masks[index] if index < len(ref_masks) else None
        target_mask = ref_target_masks[index] if index < len(ref_target_masks) else None
        if scope != "global" and target_mask is None:
            raise ValueError(f"[Krea2 CcC Edit] Scope '{scope}' requires a target_attention_mask.")

        ref_start = offsets[index]
        ref_end = offsets[index + 1]
        ref_len = ref_token_lens[index]
        gh, gw = ref_token_grids[index]
        if gh * gw != ref_len:
            raise ValueError("[Krea2 CcC Edit] Reference mask token grid does not match reference tokens.")

        ref_weight = (
            _mask_to_flat(source_mask, gh, gw, batch_size, device, dtype)
            if source_mask is not None
            else torch.ones((batch_size, ref_len), device=device, dtype=dtype)
        )
        if scope == "global":
            tgt_weight = torch.ones((batch_size, tgt_len), device=device, dtype=dtype)
        else:
            if target_grid is None or target_grid[0] * target_grid[1] != tgt_len:
                raise ValueError("[Krea2 CcC Edit] Target mask requires a valid target token grid.")
            tgt_weight = _mask_to_flat(
                target_mask, target_grid[0], target_grid[1],
                batch_size, device, dtype
            )

        block = bias[:, :, target_start:, ref_start:ref_end]
        regional = ref_attention_maps[index] if index < len(ref_attention_maps) else None
        if regional is not None:
            if regional.ndim == 2:
                regional = regional.unsqueeze(0)
            if regional.ndim != 3:
                raise ValueError("Reference weight map must be [B,H,W].")
            weights = F.interpolate(
                regional.to(device=device, dtype=torch.float32).unsqueeze(1),
                size=(gh, gw), mode="area"
            )[:, 0].flatten(1).to(dtype=dtype)
            weights = weights.expand(batch_size, -1)
            if torch.any(weights < 0):
                raise ValueError("Reference attention weights must be nonnegative.")
            block += (weights.clamp_min(1e-12).log().unsqueeze(1)
                      * tgt_weight.unsqueeze(-1)).unsqueeze(1)
            zero = (weights <= 0).unsqueeze(1) & (tgt_weight > 0).unsqueeze(-1)
            block.masked_fill_(zero.unsqueeze(1), float("-inf"))
        elif float(boost) != 1.0:
            score = math.log(max(float(boost), 1e-4))
            block += (score * tgt_weight.unsqueeze(-1) * ref_weight.unsqueeze(1)).unsqueeze(1)
        if scope == "only in mask":
            block.masked_fill_((tgt_weight <= 0.5).view(batch_size, 1, tgt_len, 1), float("-inf"))

    return bias


def _build_incontext_3d_rope_pos_ids(
    batch_size: int,
    txt_len: int,
    ref_token_grids: List[Tuple[int, int]],
    target_grid: Tuple[int, int],
    device: torch.device,
    ref_rope_positions: Optional[List[str]] = None,
) -> torch.Tensor:
    """Default Identity Edit centered positions; modular Edit can replace this helper."""
    tgt_h, tgt_w = target_grid
    parts = [torch.zeros((txt_len, 3), device=device, dtype=torch.float32)] if txt_len else []
    for index, (ref_h, ref_w) in enumerate(ref_token_grids):
        y0 = max(0.0, (tgt_h - ref_h) / 2.0)
        x0 = max(0.0, (tgt_w - ref_w) / 2.0)
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
    ref_rope_positions: Optional[List[str]] = None,
    ref_target_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ref_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ref_attention_maps: Optional[List[Optional[torch.Tensor]]] = None,
    ref_attention_scopes: Optional[List[str]] = None,
) -> torch.Tensor:
    """In-context Krea2 forward with optional Krea2 CcC Edit RoPE displacement."""
    transformer_options = transformer_options or {}
    n_refs = len(ref_latents)
    boosts = [float(v) for v in _normalize_runtime_list(ref_boosts, n_refs, 1.0)]
    rope_positions = [str(v) for v in _normalize_runtime_list(ref_rope_positions, n_refs, "none")]
    target_masks = _normalize_runtime_list(ref_target_masks, n_refs, None)
    source_masks = _normalize_runtime_list(ref_masks, n_refs, None)
    attention_maps = _normalize_runtime_list(ref_attention_maps, n_refs, None)
    attention_scopes = [str(v) for v in _normalize_runtime_list(ref_attention_scopes, n_refs, "global")]

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
    for source in ref_latents:
        src = source.to(device=x.device, dtype=x.dtype)
        if src.ndim == 5:
            sb, sc, st, sh, sw = src.shape
            src = src.reshape(sb * st, sc, sh, sw)
        src = _repeat_to_batch_size(src, batch)
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
        ref_token_grids=source_grids,
        ref_masks=source_masks,
        ref_attention_maps=attention_maps,
        ref_target_masks=target_masks,
        ref_attention_scopes=attention_scopes,
        target_grid=(target_gh, target_gw),
        batch_size=batch,
        device=combined.device,
        dtype=combined.dtype,
    )

    total_blocks = len(dit_model.blocks)
    total_ref_len = source_len
    for block_index, block in enumerate(dit_model.blocks):
        # Preserve the current ComfyUI Krea2 transformer metadata contract while keeping the
        # Identity Edit sequence order [text | refs | target]. This matters for attention patches
        # that consume img_slice/block_index without changing reference semantics.
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
