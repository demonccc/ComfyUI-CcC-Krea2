# ComfyUI-CcC-Krea2

> [!WARNING]
> **Status: Experimental Alpha**  
> This package is currently in **Experimental Alpha**. End-to-end validation with an active ComfyUI installation, Krea 2 checkpoint, and compatible identity edit LoRA is ongoing.

**CcC Krea2** is an explicit reference-guided image editing custom node suite for ComfyUI. It enables Krea 2 identity editing, scene transfer, and inpainting with zero global monkey patching, strict `ModelPatcher` instance isolation, dual-path image conditioning (Qwen3-VL grounding and VAE reference latents), and attention mask steering.

License: GPL-3.0  
Category in ComfyUI: `CcC/Krea2`  
Initial Version: `0.1.0`

---

## Architectural Principles

- **Explicit ModelPatcher Isolation**: Uses standard ComfyUI `DIFFUSION_MODEL` wrapper registration `(executor, x, timesteps, context, *wargs, **kwargs)`. Each node clones the `MODEL` and patches only that model instance.
- **Closure Transport**: Reference latents, masks, and attention boosts are captured directly in the per-instance wrapper closure.
- **Identical Latent Scaling (`process_latent_in`)**: Every VAE reference latent passes through `model.model.process_latent_in(...)` to ensure target and reference latents share the exact same latent scaling space.
- **Native Reference Pixel Geometry**: `reference_fit_mode="fit"` preserves native reference grid aligned to multiples of 16 without black canvas padding.
- **Masked Latent Img2Img Inpainting**: Inpainting nodes attach `noise_mask` metadata for latent img2img sampling. For pixel-exact outside-mask preservation, combine with a downstream VAE decode + pixel compositing node.
- **Independent Sampling vs Reference Geometry**:
  - `sampling_resize_mode` (`fit`, `crop`, `stretch`) controls ONLY the output `LATENT` returned to KSampler.
  - `reference_fit_mode` (`fit`, `crop`) controls VAE reference token geometry (`stretch` is excluded).
- **Dynamic Qwen3-VL Prompt Templates**: Automatically constructs system and vision prompt templates matching the exact number of active reference images.
- **Pure PyTorch Processing**: Zero reliance on external binary dilation dependencies. Mask growth uses PyTorch `max_pool2d`.

---

## Workflow Classification & LoRA Contracts

| Feature / Workflow | Status | Requirements & Notes |
| :--- | :---: | :--- |
| **CcC Krea2 - Subject** | **Standard** | Single-reference subject identity editing. Verified for standard Krea 2 Identity Edit LoRA contract. |
| **CcC Krea2 - Inpaint** | **Standard** | Masked latent img2img inpainting on source image. |
| **CcC Krea2 - Subject + Scene** | **Standard** | Dual-reference editing `[scene, subject]`. Primary subject placed last in sequence. |
| **CcC Krea2 - Inpaint Subject + Scene** | **Standard** | Dual-reference inpainting on scene image base. |
| **CcC Krea2 - Subject + Outfit** | *Experimental* | Requires specialized outfit-trained edit LoRA. |
| **CcC Krea2 - Subject + Scene + Outfit** | *Experimental* | Requires specialized 3-reference LoRA trained for `[scene, outfit, subject]`. |
| **CcC Krea2 - Inpaint Subject + Outfit** | *Experimental* | Requires specialized outfit inpainting LoRA. |
| **Hard Attention Masks** | **Stable** | Binary threshold at 0.5. |
| **Soft Attention Masks** | *Experimental* | Continuous weight scaling `[0.0, 1.0]`. |

---

## Node Reference

### General Editing Nodes
1. **CcC Krea2 - Subject** (`CcCKrea2Subject`)
2. **CcC Krea2 - Subject + Scene** (`CcCKrea2SubjectScene`)
3. **CcC Krea2 - Subject + Outfit** (`CcCKrea2SubjectOutfit`) *(Experimental)*
4. **CcC Krea2 - Subject + Scene + Outfit** (`CcCKrea2SubjectSceneOutfit`) *(Experimental)*

### Inpainting Nodes
5. **CcC Krea2 - Inpaint** (`CcCKrea2Inpaint`) — Base image explicitly bound to `source_image`.
6. **CcC Krea2 - Inpaint Subject + Scene** (`CcCKrea2InpaintSubjectScene`) — Base image explicitly bound to `scene_image`.
7. **CcC Krea2 - Inpaint Subject + Outfit** (`CcCKrea2InpaintSubjectOutfit`) *(Experimental)* — Base image explicitly bound to `subject_image`.

---

## Unit Testing

Run unit tests using `pytest`:

```bash
PYTHONPATH=. pytest tests/
```

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0). See [LICENSE](LICENSE) for details.
