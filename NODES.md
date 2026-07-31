# CcC Krea2 Node Reference

Complete user-facing documentation for all 7 nodes in the **CcC Krea2** suite.

---

## Shared Concepts & Pipeline Architecture

### Dual-Path Reference Pipeline
Every reference image passed into a CcC Krea2 node is processed through two distinct parallel paths:

1. **Qwen3-VL Grounding Path**:
   - Resizes reference images according to `grounding_resize_mode` and `grounding_px` (default 768px).
   - Generates vision tokens for Qwen3-VL dynamic prompt template formatting inside `CONDITIONING`.

2. **VAE Reference Path**:
   - Fits reference images using `reference_fit_mode` (`fit` preserving aspect ratio aligned to `/16` floor grid, or `crop`).
   - Encodes reference images into VAE latents, applies `process_latent_in`, and captures them inside a per-instance model patch closure.

### Model Patching (`patched_model`)
- Each node clones the input `MODEL` and applies a model patch exclusively to that returned instance.
- No global state or class modification is performed on ComfyUI.
- The model patch injects in-context reference sequence tokens and 3D RoPE position IDs during DiT execution.

### Grounding Controls
- **`grounding_resize_mode`**: Controls pixel sizing for Qwen3-VL vision tokens (`none`, `downscale_only`, `normalize`, `clamp`).
- **`grounding_preset`**: Presets (`balanced`: 768px, `max_identity`: 1024px).
- **`grounding_px`**: Target pixel length for vision tokens (default 768).

### Geometry Modes
- **`reference_fit_mode`**: Controls spatial transformation for VAE reference tokens (`fit` or `crop`). `fit` preserves native aspect ratio aligned to `/16` grid without black canvas padding.
- **`sampling_resize_mode`**: Controls spatial transformation for the output `LATENT` returned to KSampler (`fit`, `crop`, `stretch`).

### Attention Steering & Masks
- **`boost`**: Per-reference attention multiplier (default `subject`: 2.5, `scene`/`outfit`/`source`: 1.0). Higher values force stronger attention alignment to that reference's tokens.
- **Attention Mask**: An attention mask does not remove or block reference tokens. It limits where the reference boost is applied. Outside the mask, the reference remains available with normal attention bias equal to zero.
- **Inpaint Mask**: Defines the spatial edit region on the target latent during sampling.
- **Attention Masks vs Inpaint Masks**: Attention masks control where the per-reference attention boost is applied, whereas inpaint masks specify where generation/editing occurs on the output image.

### Sampling Latent & Inpainting Behavior
- **`latent_source`**: Determines whether KSampler begins from an empty latent or a VAE-encoded reference image latent (`empty`, `subject`, `scene`).
- **Masked Latent img2img**: Inpainting nodes construct a masked latent img2img target latent structure (`samples` + `noise_mask`) for KSampler.

---

## Shared Compatibility Note

> [!NOTE]
> Reference interpretation depends on the loaded Krea 2 edit LoRA, reference order and prompt. Identity Edit v1.2 is the recommended starting point.

---

## Node Reference

### 1. CcC Krea2 - Subject

Single-reference subject identity editing node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[subject]`
- **Inputs**:
  - `model` (Required): Input Krea 2 `MODEL`.
  - `clip` (Required): Input `CLIP` text/vision encoder.
  - `vae` (Required): Input `VAE`.
  - `prompt` (Required): Text prompt describing the target edit.
  - `subject_image` (Required): Primary subject reference image.
- **Latent Source**: `empty` | `subject`
- **Key Parameters**: `subject_boost` (default 2.5), `subject_attention_mask`, `sampling_resize_mode`, `reference_fit_mode`.
- **Example Use Case**: Changing the hair color, age, or expression of a subject while preserving identity.

---

### 2. CcC Krea2 - Subject + Outfit

Dual-reference subject identity and clothing/outfit editing node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[outfit, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image` (Required): Primary subject reference image.
  - `outfit_image` (Required): Reference garment or outfit image.
- **Latent Source**: `empty` | `subject`
- **Key Parameters**: `subject_boost`, `outfit_boost`, `outfit_attention_mask`.
- **Example Use Case**: Transferring a specific outfit onto a target subject.

---

### 3. CcC Krea2 - Subject + Scene

Dual-reference subject identity and background scene composition node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[scene, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image` (Required): Primary subject reference image.
  - `scene_image` (Required): Target background scene image.
- **Latent Source**: `empty` | `subject` | `scene`
- **Key Parameters**: `subject_boost`, `scene_boost`, `scene_attention_mask`.
- **Example Use Case**: Placing a subject into a new environment or background scene.

---

### 4. CcC Krea2 - Subject + Scene + Outfit

Triple-reference composition node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[scene, outfit, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image` (Required): Primary subject reference image.
  - `scene_image` (Required): Target background scene image.
  - `outfit_image` (Required): Reference garment or outfit image.
- **Latent Source**: `empty` | `subject` | `scene`
- **Key Parameters**: `subject_boost`, `scene_boost`, `outfit_boost`.
- **Example Use Case**: Composing a subject wearing a specific outfit inside a complex custom scene.

---

### 5. CcC Krea2 - Inpaint

Localized inpainting node using a single source image and mask.

- **Category**: `CcC/Krea2`
- **Base Image**: `source_image`
- **Reference Order**: `[source]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `source_image` (Required): Base image to modify.
  - `inpaint_mask` (Required): Mask defining the edit region.
- **Latent Source**: Fixed to `source_image` VAE latent with noise mask.
- **Key Parameters**: `inpaint_mask_grow`, `inpaint_mask_blur`, `source_boost`.
- **Example Use Case**: Replacing an object or repairing a specific region within a single image.

---

### 6. CcC Krea2 - Inpaint Subject + Outfit

Localized inpainting node for subject and outfit transfer.

- **Category**: `CcC/Krea2`
- **Base Image**: `subject_image`
- **Reference Order**: `[outfit, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image` (Required): Base subject image to modify.
  - `outfit_image` (Required): Reference outfit image.
  - `inpaint_mask` (Required): Mask defining the clothing region to modify.
- **Latent Source**: Fixed to `subject_image` VAE latent with noise mask.
- **Key Parameters**: `outfit_boost`, `inpaint_mask_grow`, `inpaint_mask_blur`.
- **Example Use Case**: Inpainting a new outfit directly onto a masked portion of a subject photo.

---

### 7. CcC Krea2 - Inpaint Subject + Scene

Dual-reference localized inpainting node for subject placement into a scene.

- **Category**: `CcC/Krea2`
- **Base Image**: `scene_image`
- **Reference Order**: `[scene, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `scene_image` (Required): Base background scene image.
  - `subject_image` (Required): Subject reference image.
  - `inpaint_mask` (Required): Mask defining where the subject should be inserted into the scene.
- **Latent Source**: Fixed to `scene_image` VAE latent with noise mask.
- **Key Parameters**: `subject_boost`, `scene_boost`, `inpaint_mask_grow`, `inpaint_mask_blur`.
- **Example Use Case**: Seamlessly inpainting a subject into a specific masked region of a background scene.
