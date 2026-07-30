# ComfyUI-CcC-Krea2

**CcC Krea2** is a production-ready, explicit reference-guided image editing custom node suite for ComfyUI. It enables bit-identical Krea 2 identity editing, scene transfer, and inpainting with zero global monkey patching, strict ModelPatcher instance isolation, dual-path image conditioning (Qwen3-VL grounding and VAE reference latents), and comprehensive attention mask steering.

License: GPL-3.0  
Category in ComfyUI: `CcC/Krea2`  
Initial Version: `0.1.0`

---

## Key Features

- **Explicit ModelPatcher Isolation**: Uses ComfyUI `ModelPatcher` wrapper registration. Each node clones the `MODEL` and patches only that model instance. Core ComfyUI execution logic remains pristine.
- **Dual-Path Reference Pipeline**:
  - **Qwen3-VL Grounding Path**: Resizes reference images preserving aspect ratio with flexible mode (`normalize`, `downscale_only`, `clamp`, `none`) and resolution presets (`balanced` = 768px, `max_identity` = 1024px, `custom`).
  - **VAE Reference Latent Path**: Resizes reference images into output pixel grid using `reference_fit_mode` (`fit` or `crop`) and encodes latents with VAE.
- **Independent Sampling vs Reference Geometry**:
  - `sampling_resize_mode` (`fit`, `crop`, `stretch`) controls ONLY the output `LATENT` returned to KSampler.
  - `reference_fit_mode` (`fit`, `crop`) controls VAE reference token geometry. `stretch` is excluded for reference tokens to prevent deformation.
- **Dynamic Qwen3-VL Prompt Templates**: Automatically constructs system and vision prompt templates matching the exact number of active reference images.
- **Attention Steering & Numerical Safety**: Additive logit bias steering (`b_val = math.log(boost)`). Protected against `log(0)`, `NaN`, and `Inf` with automatic clamping and value normalization.
- **Single VAE Encoding Execution**: Reference images are prepared and VAE-encoded once per node execution and shared across positive and negative conditionings.

---

## LoRA Compatibility & Feature Classification

> [!IMPORTANT]
> To achieve optimal results, please review the contract matrix below:

| Feature / Workflow | Status | Verification & Contract Notes |
| :--- | :---: | :--- |
| **CcC Krea2 - Subject** | **Verified** | Standard single-reference identity editing contract. Verified with standard Krea 2 Identity Edit LoRA. |
| **CcC Krea2 - Inpaint** | **Verified** | Standard single-image inpainting with noise mask metadata. |
| **CcC Krea2 - Subject + Scene** | **Verified** | Verified dual-reference workflow `[scene, subject]`. Primary subject is placed last in sequence. |
| **CcC Krea2 - Inpaint Subject + Scene** | **Verified** | Verified dual-reference inpainting on scene image base. |
| **CcC Krea2 - Subject + Outfit** | *Experimental* | Requires specialized outfit-trained edit LoRA. Standard identity LoRAs do not natively recognize the `outfit` role. |
| **CcC Krea2 - Subject + Scene + Outfit** | *Experimental* | Requires specialized 3-reference LoRA trained for `[scene, outfit, subject]` sequence order. |
| **CcC Krea2 - Inpaint Subject + Outfit** | *Experimental* | Requires specialized outfit inpainting LoRA. |
| **Hard Attention Masks** | **Stable** | Binary threshold at 0.5. Recommended stable mask steering path. |
| **Soft Attention Masks** | *Experimental* | Continuous grayscale weight scaling `[0.0, 1.0]`. |

---

## Provided Custom Nodes

### General Editing Nodes

1. **CcC Krea2 - Subject** (`CcCKrea2Subject`)
   - Standard single reference subject editing.
   - Reference role: `[subject]`
   - Latent options: `empty`, `subject`

2. **CcC Krea2 - Subject + Scene** (`CcCKrea2SubjectScene`)
   - Dual reference editing: background scene and primary subject.
   - Reference roles: `[scene, subject]`
   - Latent options: `empty`, `subject`, `scene`

3. **CcC Krea2 - Subject + Outfit** (`CcCKrea2SubjectOutfit`) *(Experimental)*
   - Subject and outfit composition.
   - Reference roles: `[outfit, subject]`
   - Latent options: `empty`, `subject` (Outfit is strictly excluded as a sampling latent)

4. **CcC Krea2 - Subject + Scene + Outfit** (`CcCKrea2SubjectSceneOutfit`) *(Experimental)*
   - 3-reference composition.
   - Reference roles: `[scene, outfit, subject]`
   - Latent options: `empty`, `subject`, `scene`

### Inpainting Nodes

5. **CcC Krea2 - Inpaint** (`CcCKrea2Inpaint`)
   - Single source image inpainting.
   - Reference role: `[source]`

6. **CcC Krea2 - Inpaint Subject + Scene** (`CcCKrea2InpaintSubjectScene`)
   - Inpainting on scene image using subject identity reference.
   - Reference roles: `[scene, subject]`

7. **CcC Krea2 - Inpaint Subject + Outfit** (`CcCKrea2InpaintSubjectOutfit`) *(Experimental)*
   - Inpainting on subject image using outfit reference.
   - Reference roles: `[outfit, subject]`

---

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/demonccc/ComfyUI-CcC-Krea2.git
   ```

2. Install dependencies:
   ```bash
   pip install -r ComfyUI-CcC-Krea2/requirements.txt
   ```

3. Restart ComfyUI. The nodes will appear under the category `CcC/Krea2`.

---

## Running Unit Tests

Run the CPU-safe unit test suite using `pytest`:

```bash
cd custom_nodes/ComfyUI-CcC-Krea2
pytest tests/
```

---

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0). See [LICENSE](LICENSE) for details.
