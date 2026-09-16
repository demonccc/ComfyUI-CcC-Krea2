"""CcC Krea 2 Identity Edit runtime with per-reference RoPE placement.

Visual-only Identity Edit stores prepared appearance references in a per-MODEL diffusion
wrapper. This mirrors the source-patch architecture: Qwen conditioning carries semantics,
while the MODEL wrapper owns clean VAE source latents, fidelity boosts and spatial placement.
No global SingleStreamDiT monkey-patch is installed by CcC.
"""

import math
from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange

from .patch import install_krea2_reference_conditioning


def _fit_latent(src: torch.Tensor, height: int, width: int) -> torch.Tensor:
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


def _ref_attn_bias(
    boosts: List[float],
    txt_len: int,
    source_lens: List[int],
    target_len: int,
    device: torch.device,
    dtype: torch.dtype,
):
    offsets = [txt_len]
    for source_len in source_lens:
        offsets.append(offsets[-1] + source_len)
    target_start = offsets[-1]
    total_len = target_start + target_len
    bias = torch.zeros((1, 1, total_len, total_len), device=device, dtype=dtype)
    for index, boost in enumerate(boosts):
        if boost == 1.0:
            continue
        bias[:, :, target_start:, offsets[index]:offsets[index] + source_lens[index]] = math.log(
            max(float(boost), 1e-4)
        )
    return bias


def _align_runtime_list(value: Any, count: int, default: Any):
    if count <= 0:
        return []
    if value is None:
        raw = []
    elif torch.is_tensor(value):
        raw = value.detach().flatten().tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [value]
    return [default] * max(0, count - len(raw)) + raw[-count:]


def _extract_transformer_options(wargs, kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        for item in reversed(wargs):
            if isinstance(item, dict):
                transformer_options = item
                break
    return transformer_options or {}


def _extract_conditioning_refs(wargs, kwargs):
    ref_latents = kwargs.get("ref_latents")
    if ref_latents is None:
        for item in wargs:
            if isinstance(item, (list, tuple)) and item and all(torch.is_tensor(v) for v in item):
                ref_latents = list(item)
                break
    return list(ref_latents or [])


def _identity_edit_forward(
    model: Any,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    ref_latents: List[torch.Tensor],
    ref_boosts: List[float],
    ref_fit: List[bool],
    ref_rope_positions: List[str],
    transformer_options: dict,
) -> torch.Tensor:
    """Run the training-style Krea2 sequence [text | refs | target]."""
    try:
        import comfy.ldm.common_dit
        from comfy.ldm.flux.layers import timestep_embedding
        from .modular_nodes.rope_position import build_incontext_3d_rope_pos_ids
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("[CcC Krea2] Required Krea2 runtime modules are unavailable.") from exc

    count = len(ref_latents)
    ref_boosts = [float(v) for v in _align_runtime_list(ref_boosts, count, 1.0)]
    ref_fit = [bool(v) for v in _align_runtime_list(ref_fit, count, False)]
    ref_rope_positions = [
        str(v)
        for v in _align_runtime_list(ref_rope_positions, count, "inside:center:center")
    ]

    temporal = x.ndim == 5
    if temporal:
        batch5, channels5, frames5, height5, width5 = x.shape
        x = x.reshape(batch5 * frames5, channels5, height5, width5)

    batch, _, original_h, original_w = x.shape
    patch = model.patch

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    height, width = x.shape[-2:]
    target_gh, target_gw = height // patch, width // patch

    sources = []
    for index, source in enumerate(ref_latents):
        src = source.to(device=x.device, dtype=x.dtype)
        if src.ndim == 5:
            sb, sc, st, sh, sw = src.shape
            src = src.reshape(sb * st, sc, sh, sw)
        if src.ndim != 4:
            raise ValueError(
                f"[CcC Krea2] Runtime reference must resolve to 4D image latent data, got {tuple(src.shape)}."
            )
        if src.shape[0] != batch:
            src = src[:1].expand(batch, *src.shape[1:])

        native = ref_fit[index] and src.shape[-2] <= height and src.shape[-1] <= width
        if src.shape[-2:] != (height, width) and not native:
            src = _fit_latent(src, height, width).to(x.dtype)
        sources.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch)))

    context = model._unpack_context(context)

    target = rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch,
        pw=patch,
    )
    target = model.first(target)
    source_tokens = [
        model.first(
            rearrange(
                source,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                ph=patch,
                pw=patch,
            )
        )
        for source in sources
    ]

    t = model.tmlp(timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(target.dtype))
    tvec = model.tproj(t)

    context = model.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = model.txtmlp(context)

    txt_len = context.shape[1]
    target_len = target.shape[1]
    source_lens = [tokens.shape[1] for tokens in source_tokens]
    source_len = sum(source_lens)
    combined = torch.cat([context] + source_tokens + [target], dim=1)

    source_grids = [(source.shape[-2] // patch, source.shape[-1] // patch) for source in sources]
    pos = build_incontext_3d_rope_pos_ids(
        batch_size=batch,
        txt_len=txt_len,
        ref_token_grids=source_grids,
        target_grid=(target_gh, target_gw),
        ref_rope_positions=ref_rope_positions,
        device=combined.device,
    )
    freqs = model.pe_embedder(pos)

    attn_bias = None
    if source_tokens and any(boost != 1.0 for boost in ref_boosts):
        attn_bias = _ref_attn_bias(
            ref_boosts,
            txt_len,
            source_lens,
            target_len,
            combined.device,
            combined.dtype,
        )

    for block in model.blocks:
        combined = block(
            combined,
            tvec,
            freqs,
            attn_bias,
            transformer_options=transformer_options,
        )

    final = model.last(combined, t)
    output = final[:, txt_len + source_len:txt_len + source_len + target_len, :]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=target_gh,
        w=target_gw,
        ph=patch,
        pw=patch,
        c=model.channels,
    )
    output = output[:, :, :original_h, :original_w]

    if temporal:
        output = output.reshape(
            batch5, frames5, model.channels, original_h, original_w
        ).movedim(1, 2)
    return output


def _register_model_wrapper(model: Any, wrapper: Any) -> None:
    """Install a per-model diffusion wrapper without mutating SingleStreamDiT globally."""
    try:
        import comfy.patcher_extension

        options = model.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "ccc_krea2_edit",
            wrapper,
            options,
        )
        return
    except (ImportError, AttributeError, TypeError):
        pass

    if hasattr(model, "add_wrapper_with_key"):
        try:
            model.add_wrapper_with_key("diffusion_model", "ccc_krea2_edit", wrapper)
            return
        except TypeError:
            model.add_wrapper_with_key("ccc_krea2_edit", wrapper)
            return

    raise RuntimeError("[CcC Krea2] This ComfyUI version does not expose a compatible model wrapper API.")


def install_identity_krea2_forward() -> bool:
    """Compatibility shim: CcC no longer patches SingleStreamDiT globally."""
    return True


def _process_model_latents(model: Any, refs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Scale raw VAE refs exactly once using the same BaseModel latent transform as the target."""
    base_model = getattr(model, "model", None)
    if base_model is None or not hasattr(base_model, "process_latent_in"):
        raise RuntimeError("[CcC Krea2] Connected MODEL does not expose process_latent_in().")
    return [base_model.process_latent_in(ref) for ref in refs]


def patch_krea2_identity_model(
    model: Any,
    reference_latents: Optional[Sequence[torch.Tensor]] = None,
    reference_boosts: Optional[Sequence[float]] = None,
    reference_fit: Optional[Sequence[bool]] = None,
    reference_rope_positions: Optional[Sequence[str]] = None,
) -> Any:
    """Return a model clone with an isolated Identity Edit appearance wrapper.

    When ``reference_latents`` are provided, they are captured by the MODEL wrapper and are
    never transported through CONDITIONING. This is the preferred visual-only path and keeps
    third-party ``Krea2.extra_conds`` patches out of appearance transport. With no captured
    refs, the compatibility path can still consume conditioning-owned refs.
    """
    captured_refs = list(reference_latents or [])
    if captured_refs:
        runtime_refs = _process_model_latents(model, captured_refs)
    else:
        install_krea2_reference_conditioning()
        runtime_refs = []

    patched = model.clone()
    patched._ccc_patch_key = "ccc_krea2_edit"
    patched._ccc_identity_reference_count = len(runtime_refs)
    patched._ccc_identity_reference_shapes = [tuple(int(v) for v in ref.shape) for ref in runtime_refs]

    captured_boosts = [float(v) for v in _align_runtime_list(reference_boosts, len(runtime_refs), 1.0)]
    captured_fit = [bool(v) for v in _align_runtime_list(reference_fit, len(runtime_refs), True)]
    captured_rope = [
        str(v)
        for v in _align_runtime_list(
            reference_rope_positions, len(runtime_refs), "inside:center:center"
        )
    ]
    patched._ccc_identity_reference_boosts = list(captured_boosts)
    patched._ccc_identity_reference_fit = list(captured_fit)
    patched._ccc_identity_reference_rope = list(captured_rope)

    def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
        transformer_options = _extract_transformer_options(wargs, kwargs)

        if runtime_refs:
            refs = runtime_refs
            boosts = captured_boosts
            fit_flags = captured_fit
            rope_positions = captured_rope
        else:
            refs = _extract_conditioning_refs(wargs, kwargs)
            if not refs:
                return executor(x, timesteps, context, *wargs, **kwargs)
            count = len(refs)
            boosts = [float(v) for v in _align_runtime_list(kwargs.get("ref_boosts"), count, 1.0)]
            fit_flags = [bool(v) for v in _align_runtime_list(kwargs.get("ref_fit"), count, False)]
            rope_positions = [
                str(v)
                for v in _align_runtime_list(
                    kwargs.get("ccc_ref_rope_positions"), count, "inside:center:center"
                )
            ]

        diffusion_model = getattr(executor, "class_obj", None)
        if diffusion_model is None:
            raise RuntimeError("[CcC Krea2] Could not resolve the Krea2 diffusion model from the wrapper executor.")

        return _identity_edit_forward(
            model=diffusion_model,
            x=x,
            timesteps=timesteps,
            context=context,
            ref_latents=refs,
            ref_boosts=boosts,
            ref_fit=fit_flags,
            ref_rope_positions=rope_positions,
            transformer_options=transformer_options,
        )

    _register_model_wrapper(patched, wrapper)
    return patched
