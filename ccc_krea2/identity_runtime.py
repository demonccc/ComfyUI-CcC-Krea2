"""CcC Krea2 Identity Edit runtime with per-reference RoPE placement.

This module owns the SingleStreamDiT in-context reference forward used by the split Edit
surface. Target geometry, pixel-space reference fit and conditioning transport stay outside
this module; CcC adds per-reference RoPE placement at runtime.
"""

import math
from typing import Any, List

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


def install_identity_krea2_forward() -> bool:
    """Install the CcC Identity Edit SingleStreamDiT forward once per ComfyUI process."""
    try:
        import comfy.ldm.common_dit
        from comfy.ldm.flux.layers import timestep_embedding
        from comfy.ldm.krea2.model import SingleStreamDiT
        from .modular_nodes.rope_position import build_incontext_3d_rope_pos_ids
    except (ImportError, AttributeError):
        return False

    if getattr(SingleStreamDiT, "_ccc_identity_runtime_patched", False):
        return True

    def _ccc_krea2_forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        *_drift,
        transformer_options=None,
        **kwargs,
    ):
        # Handle ComfyUI signature drift while preserving conditioning-owned references.
        native_ref = None
        drift = list(_drift)
        if drift and isinstance(drift[-1], dict) and transformer_options is None:
            transformer_options = drift.pop()
        if drift:
            native_ref = drift.pop(0)
        if transformer_options is None:
            transformer_options = {}

        ref_latents = kwargs.get("ref_latents", None) or native_ref or []
        ref_boosts = list(kwargs.get("ref_boosts", None) or [])
        ref_fit = list(kwargs.get("ref_fit", None) or [])
        ref_rope_positions = list(kwargs.get("ccc_ref_rope_positions", None) or [])

        count = len(ref_latents)
        ref_boosts = [float(v) for v in _align_runtime_list(ref_boosts, count, 1.0)]
        ref_fit = [bool(v) for v in _align_runtime_list(ref_fit, count, False)]
        ref_rope_positions = [
            str(v)
            for v in _align_runtime_list(
                ref_rope_positions, count, "inside:center:center"
            )
        ]

        temporal = x.ndim == 5
        if temporal:
            batch5, channels5, frames5, height5, width5 = x.shape
            x = x.reshape(batch5 * frames5, channels5, height5, width5)

        batch, _, original_h, original_w = x.shape
        patch = self.patch

        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
        height, width = x.shape[-2:]
        target_gh, target_gw = height // patch, width // patch

        sources = []
        for index, source in enumerate(ref_latents):
            src = source.to(device=x.device, dtype=x.dtype)
            if src.ndim == 5:
                sb, sc, st, sh, sw = src.shape
                src = src.reshape(sb * st, sc, sh, sw)
            if src.shape[0] != batch:
                src = src[:1].expand(batch, *src.shape[1:])

            # Pixel-fit-prepared refs keep their own grid when they fit inside the target.
            native = ref_fit[index] and src.shape[-2] <= height and src.shape[-1] <= width
            if src.shape[-2:] != (height, width) and not native:
                src = _fit_latent(src, height, width).to(x.dtype)
            sources.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch)))

        context = self._unpack_context(context)

        target = rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch,
            pw=patch,
        )
        target = self.first(target)
        source_tokens = [
            self.first(
                rearrange(
                    source,
                    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=patch,
                    pw=patch,
                )
            )
            for source in sources
        ]

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(target.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

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
        freqs = self.pe_embedder(pos)

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

        # Keep the native block call contract; CcC only changes reference placement and boosts.
        for block in self.blocks:
            combined = block(
                combined,
                tvec,
                freqs,
                attn_bias,
                transformer_options=transformer_options,
            )

        final = self.last(combined, t)
        output = final[:, txt_len + source_len:txt_len + source_len + target_len, :]
        output = rearrange(
            output,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=target_gh,
            w=target_gw,
            ph=patch,
            pw=patch,
            c=self.channels,
        )
        output = output[:, :, :original_h, :original_w]

        if temporal:
            output = output.reshape(
                batch5, frames5, self.channels, original_h, original_w
            ).movedim(1, 2)
        return output

    SingleStreamDiT._forward = _ccc_krea2_forward
    SingleStreamDiT._ccc_identity_runtime_patched = True
    return True


def patch_krea2_identity_model(model: Any) -> Any:
    """Return a model clone using the CcC Identity Edit runtime."""
    install_krea2_reference_conditioning()
    if not install_identity_krea2_forward():
        raise RuntimeError(
            "[CcC Krea2] Could not install the Krea2 Identity Edit SingleStreamDiT runtime."
        )

    patched = model.clone()
    patched._ccc_patch_key = "ccc_krea2_edit"
    return patched
