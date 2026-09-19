"""Krea2 CcC Paint runtime with registered t=0 reference K/V."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from einops import rearrange

from .patch import (
    _pad_to_patch_size,
    _repeat_to_batch_size,
    _timestep_embedding,
    install_krea2_reference_conditioning,
)


def _register_paint_wrapper(patched_model: Any, wrapper: Any) -> None:
    """Register a diffusion-model wrapper under a Paint-specific key."""
    registered = False
    wrapper_type: Any = "diffusion_model"
    try:
        import comfy.patcher_extension

        wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    except Exception:
        pass

    if hasattr(patched_model, "add_wrapper_with_key"):
        try:
            patched_model.add_wrapper_with_key(wrapper_type, "ccc_krea2_paint", wrapper)
            registered = True
        except TypeError:
            try:
                patched_model.add_wrapper_with_key("ccc_krea2_paint", wrapper)
                registered = True
            except Exception:
                pass
        except Exception:
            pass

    if not registered:
        if not hasattr(patched_model, "model_options"):
            patched_model.model_options = {}
        wrappers = patched_model.model_options.setdefault(
            "transformer_options", {}
        ).setdefault("wrappers", {})
        key = str(wrapper_type)
        value = wrappers.setdefault(key, {})
        if isinstance(value, dict):
            value["ccc_krea2_paint"] = [wrapper]
        elif isinstance(value, list):
            value.append(wrapper)


def _reference_fingerprint(
    refs: Sequence[torch.Tensor],
    *,
    batch_size: int,
    target_grid: Tuple[int, int],
) -> Tuple[Any, ...]:
    key: List[Any] = [int(batch_size), tuple(int(v) for v in target_grid)]
    for ref in refs:
        value = ref.detach()
        reduced = value.float()
        key.append(
            (
                tuple(int(v) for v in value.shape),
                round(float(reduced.sum().item()), 5),
                round(float(reduced.square().sum().item()), 5),
            )
        )
    return tuple(key)


def _registered_reference_tokens(
    dit_model: Any,
    refs: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    target_grid: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patchify refs and register their x/y RoPE coordinates over the complete target grid."""
    patch_size = getattr(dit_model, "patch", 2)
    if not isinstance(patch_size, int):
        patch_size = getattr(patch_size, "patch_size", 2)

    target_h, target_w = target_grid
    token_parts: List[torch.Tensor] = []
    position_parts: List[torch.Tensor] = []

    for index, ref in enumerate(refs):
        source = ref.to(device=device, dtype=dtype)
        if source.ndim == 5:
            rb, rc, rt, rh, rw = source.shape
            source = source.reshape(rb * rt, rc, rh, rw)
        if source.ndim != 4:
            raise ValueError(
                "[Krea2 CcC Paint] Reference latent must be 4D or 5D, "
                f"got {tuple(source.shape)}."
            )
        source = _pad_to_patch_size(source, patch_size)
        source = _repeat_to_batch_size(source, batch_size)
        ref_h = source.shape[-2] // patch_size
        ref_w = source.shape[-1] // patch_size

        token_parts.append(
            rearrange(
                source,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch_size,
                pw=patch_size,
            )
        )

        ys = (
            (torch.arange(ref_h, device=device, dtype=torch.float32) + 0.5)
            * (float(target_h) / float(ref_h))
            - 0.5
        )
        xs = (
            (torch.arange(ref_w, device=device, dtype=torch.float32) + 0.5)
            * (float(target_w) / float(ref_w))
            - 0.5
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        tt = torch.full_like(yy, float(index + 1))
        position_parts.append(
            torch.stack((tt.flatten(), yy.flatten(), xx.flatten()), dim=-1)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

    if not token_parts:
        raise ValueError("[Krea2 CcC Paint] At least one reference latent is required.")
    return torch.cat(token_parts, dim=1), torch.cat(position_parts, dim=1)


def _attention_with_optional_reference_kv(
    attn: Any,
    x: torch.Tensor,
    freqs: Any,
    *,
    kv_capture: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    reference_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    transformer_options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Krea2 attention with post-RoPE reference K/V capture/injection."""
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import optimized_attention_masked

    transformer_options = transformer_options or {}
    q = attn.wq(x)
    k = attn.wk(x)
    v = attn.wv(x)
    gate = attn.gate(x)

    q = rearrange(q, "b l (h d) -> b h l d", h=attn.heads)
    k = rearrange(k, "b l (h d) -> b h l d", h=attn.kvheads)
    v = rearrange(v, "b l (h d) -> b h l d", h=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)

    if kv_capture is not None:
        kv_capture.append((k, v))

    if reference_kv is not None:
        ref_k, ref_v = reference_kv
        k = torch.cat((k, ref_k.to(device=k.device, dtype=k.dtype)), dim=2)
        v = torch.cat((v, ref_v.to(device=v.device, dtype=v.dtype)), dim=2)

    if attn.kvheads != attn.heads:
        repeat = attn.heads // attn.kvheads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    attended = optimized_attention_masked(
        q,
        k,
        v,
        attn.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return attn.wo(attended * torch.sigmoid(gate))


def _block_with_optional_reference_kv(
    block: Any,
    x: torch.Tensor,
    vec: torch.Tensor,
    freqs: Any,
    *,
    kv_capture: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    reference_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    transformer_options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """SingleStreamBlock execution with optional isolated reference K/V."""
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)

    pre = (1 + prescale) * block.prenorm(x) + preshift
    x = x + pregate * _attention_with_optional_reference_kv(
        block.attn,
        pre,
        freqs,
        kv_capture=kv_capture,
        reference_kv=reference_kv,
        transformer_options=transformer_options,
    )

    post = (1 + postscale) * block.postnorm(x) + postshift
    x = x + postgate * block.mlp(post)
    return x


def _precompute_registered_reference_kv(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    refs: Sequence[torch.Tensor],
    transformer_options: Dict[str, Any],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    temporal = x.ndim == 5
    batch_size = x.shape[0] * (x.shape[2] if temporal else 1)
    patch_size = getattr(dit_model, "patch", 2)
    if not isinstance(patch_size, int):
        patch_size = getattr(patch_size, "patch_size", 2)

    target_h = int(math.ceil(x.shape[-2] / patch_size))
    target_w = int(math.ceil(x.shape[-1] / patch_size))
    ref_tokens, ref_positions = _registered_reference_tokens(
        dit_model,
        refs,
        batch_size=batch_size,
        device=x.device,
        dtype=x.dtype,
        target_grid=(target_h, target_w),
    )
    hidden = dit_model.first(ref_tokens)

    zero = torch.zeros_like(timesteps)
    t0 = _timestep_embedding(zero, getattr(dit_model, "tdim", 256)).unsqueeze(1).to(hidden.dtype)
    t0 = dit_model.tmlp(t0)
    ref_vec = dit_model.tproj(t0)
    freqs = dit_model.pe_embedder(ref_positions)

    caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    total_blocks = len(dit_model.blocks)
    for block_index, block in enumerate(dit_model.blocks):
        capture: List[Tuple[torch.Tensor, torch.Tensor]] = []
        block_options = dict(transformer_options)
        block_options["total_blocks"] = total_blocks
        block_options["block_type"] = "single"
        block_options["block_index"] = block_index
        hidden = _block_with_optional_reference_kv(
            block,
            hidden,
            ref_vec,
            freqs,
            kv_capture=capture,
            transformer_options=block_options,
        )
        caches.append(capture[0])
    return caches


def _forward_with_registered_reference_kv(
    dit_model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    reference_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    transformer_options: Dict[str, Any],
) -> torch.Tensor:
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)

    batch, _, original_h, original_w = x.shape
    patch_size = getattr(dit_model, "patch", 2)
    if not isinstance(patch_size, int):
        patch_size = getattr(patch_size, "patch_size", 2)

    x = _pad_to_patch_size(x, patch_size)
    height, width = x.shape[-2:]
    grid_h, grid_w = height // patch_size, width // patch_size

    context = dit_model._unpack_context(context)
    image_tokens = rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )
    image_tokens = dit_model.first(image_tokens)

    t = _timestep_embedding(timesteps, getattr(dit_model, "tdim", 256)).unsqueeze(1).to(image_tokens.dtype)
    t = dit_model.tmlp(t)
    tvec = dit_model.tproj(t)

    context = dit_model.txtfusion(
        context,
        mask=None,
        transformer_options=transformer_options,
    )
    context = dit_model.txtmlp(context)

    txt_len = context.shape[1]
    image_len = image_tokens.shape[1]
    combined = torch.cat((context, image_tokens), dim=1)

    text_positions = torch.zeros(
        (batch, txt_len, 3),
        device=combined.device,
        dtype=torch.float32,
    )
    ys = torch.arange(grid_h, device=combined.device, dtype=torch.float32)
    xs = torch.arange(grid_w, device=combined.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    image_positions = torch.stack(
        (torch.zeros_like(yy).flatten(), yy.flatten(), xx.flatten()),
        dim=-1,
    ).unsqueeze(0).repeat(batch, 1, 1)
    freqs = dit_model.pe_embedder(torch.cat((text_positions, image_positions), dim=1))

    total_blocks = len(dit_model.blocks)
    if len(reference_kv) != total_blocks:
        raise ValueError(
            "[Krea2 CcC Paint] Reference K/V cache block count does not match the DiT."
        )

    for block_index, (block, cached_kv) in enumerate(zip(dit_model.blocks, reference_kv)):
        block_options = dict(transformer_options)
        block_options["total_blocks"] = total_blocks
        block_options["block_type"] = "single"
        block_options["img_slice"] = [slice(txt_len, None)]
        block_options["block_index"] = block_index
        combined = _block_with_optional_reference_kv(
            block,
            combined,
            tvec,
            freqs,
            reference_kv=cached_kv,
            transformer_options=block_options,
        )

    final = dit_model.last(combined, t)
    output = final[:, txt_len : txt_len + image_len]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=grid_h,
        w=grid_w,
        ph=patch_size,
        pw=patch_size,
        c=dit_model.channels,
    )
    output = output[:, :, :original_h, :original_w]
    if temporal:
        output = output.reshape(b5, t5, dit_model.channels, original_h, original_w).movedim(1, 2)
    return output


def patch_krea2_paint_model(model: Any, *, kv_cache: bool = True) -> Any:
    """Clone MODEL and install the Krea2 CcC Paint registered-reference runtime."""
    existing = getattr(model, "_ccc_patch_key", None)
    if existing is not None:
        raise RuntimeError(
            f"[Krea2 CcC Paint] Input MODEL is already patched by '{existing}'. "
            "Connect Paint to the unpatched model after LoRA loading."
        )

    install_krea2_reference_conditioning()
    patched = model.clone()
    patched._ccc_patch_key = "ccc_krea2_paint"
    cache_state: Dict[str, Any] = {"last_timestep": None, "entries": {}}

    def paint_wrapper(
        executor: Any,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        *wargs: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        dit_model = getattr(executor, "class_obj", None)
        refs = kwargs.get("ref_latents")
        if refs is None:
            drift = list(wargs)
            if drift and isinstance(drift[-1], dict):
                drift.pop()
            if len(drift) >= 2 and isinstance(drift[1], (list, tuple)):
                candidate = drift[1]
                if all(torch.is_tensor(item) for item in candidate):
                    refs = list(candidate)
        refs = list(refs or [])

        if dit_model is None or not refs:
            fallback_kwargs = dict(kwargs)
            for key in ("ref_latents", "ref_boosts", "ref_fit", "ccc_ref_rope_positions"):
                fallback_kwargs.pop(key, None)
            return executor(x, timesteps, context, *wargs, **fallback_kwargs)

        transformer_options = kwargs.get("transformer_options")
        if transformer_options is None and wargs and isinstance(wargs[-1], dict):
            transformer_options = wargs[-1]
        transformer_options = dict(transformer_options or {})

        patch_size = getattr(dit_model, "patch", 2)
        if not isinstance(patch_size, int):
            patch_size = getattr(patch_size, "patch_size", 2)
        target_grid = (
            int(math.ceil(x.shape[-2] / patch_size)),
            int(math.ceil(x.shape[-1] / patch_size)),
        )
        batch_size = x.shape[0] * (x.shape[2] if x.ndim == 5 else 1)
        current_timestep = float(timesteps.max().detach().float().cpu())

        last_timestep = cache_state["last_timestep"]
        if last_timestep is not None and current_timestep > float(last_timestep):
            cache_state["entries"].clear()
        cache_state["last_timestep"] = current_timestep

        key = _reference_fingerprint(
            refs,
            batch_size=batch_size,
            target_grid=target_grid,
        )
        reference_kv = cache_state["entries"].get(key) if kv_cache else None
        if reference_kv is None:
            reference_kv = _precompute_registered_reference_kv(
                dit_model,
                x,
                timesteps,
                refs,
                transformer_options,
            )
            if kv_cache:
                cache_state["entries"][key] = reference_kv

        return _forward_with_registered_reference_kv(
            dit_model,
            x,
            timesteps,
            context,
            reference_kv,
            transformer_options,
        )

    _register_paint_wrapper(patched, paint_wrapper)
    return patched
