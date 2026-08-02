# CcC Krea2 Node Reference

Complete user-facing documentation for all 11 nodes in the **CcC Krea2** suite.

---

## Shared Concepts & Pipeline Architecture

### Dual-Path Reference Pipeline
Every reference image passed into a CcC Krea2 node is processed through two distinct parallel paths:

1. **Qwen3-VL Grounding Path**:
   - Resizes reference images according to `grounding_resize_mode` and `grounding_px` (preset default 768px).
   - Generates vision tokens for Qwen3-VL dynamic prompt template formatting inside `CONDITIONING`.

2. **VAE Reference Path**:
   - Fits reference images using `reference_fit_mode` (`fit` preserving aspect ratio aligned to `/16` floor grid, or `crop`).
   - Encodes reference images into VAE latents, applies `process_latent_in`, and captures them inside a per-instance model patch closure.

### Preset System & Precedence Rules
Node configuration follows a hierarchical precedence resolver:

$$\text{Internal Defaults} \rightarrow \text{Selected Main Preset} \rightarrow \text{Edit Advanced Settings} \rightarrow \text{Image Advanced Settings per Role}$$

#### Presets
- **`balanced`** (Default): Standard identity editing (`subject boost`: 2.5, `grounding_px`: 768). Recommended for most identity transfers.
- **`max_identity`**: Maximum identity lock (`subject boost`: 4.0, `grounding_px`: 1024; `source boost`: 2.5). Recommended for subtle facial edits.
- **`flexible`**: High prompt adherence (`subject boost`: 1.5, `grounding_px`: 512). Recommended when heavy pose or artistic style changes are requested.

| Preset | Subject Boost | Grounding Px | Source Boost | Grounding Px | Scene / Outfit Boost | Target Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `balanced` | 2.5 | 768 | 1.0 | 768 | 1.0 | Standard identity editing, balanced identity & prompt responsiveness |
| `max_identity` | 4.0 | 1024 | 2.5 | 1024 | 1.0 | Maximum identity preservation, high-detail facial/source lock |
| `flexible` | 1.5 | 512 | 1.5 | 512 | 1.0 | High prompt adherence, flexible stylization and pose changes |

### Output Resolution & Megapixel Math
Output resolution can be selected directly on each main node:
- **Role-based resolution** (`subject`, `scene`, `source`): Reads the original image's width and height, preserves aspect ratio, and aligns dimensions to the nearest valid multiple of 16 (min 128x128).
  - **Role Resolution Limiting**: Controlled via `role_resolution_limit_mode` (`max_megapixels` default, or `off`) and `role_resolution_max_megapixels` (default `2.0` MP) in `CcC Krea2 - Edit Advanced Settings`. When an original role image (e.g. 4000x3000 = 12 MP) exceeds the megapixel limit, it is scaled down proportionally to remain at or below the threshold while preserving aspect ratio and 16-pixel grid alignment. Custom resolution mode ignores role resolution limiting. `outfit` is never a resolution source.
- **`custom` resolution**: Calculates output dimensions from `megapixels * 1,000,000` using the aspect ratio of the selected aspect source image (configured via `custom_aspect_source` in Edit Advanced Settings, or auto-selected based on active roles), aligned to 16-pixel multiples.

### Resize Modes vs Resize Methods
- **Resize Modes**: Spatial transform geometry (`fit` preserving AR with padding or `/16` alignment, `crop` center-cropped, `stretch` direct resize).
- **Resize Methods**: Downscaling / upscaling algorithm (`auto`, `nearest-exact`, `bilinear`, `bicubic`, `area`, `lanczos`). `auto` automatically chooses `area` for downscaling and `bicubic` (with antialiasing) for upscaling.

### Automatic Qwen3-VL Role Instructions
The system automatically formats Qwen3-VL role instructions based on active reference images, providing standard reference terms:
- `subject image`: "Use the subject image for identity, facial features, hair, anatomy, body shape and body proportions."
- `scene image`: "Use the scene image for composition, pose, environment, interactions and lighting."
- `outfit image`: "Use the outfit image for the outfit and garment details. Do not use the wearer of the outfit image as the subject identity."
- `source image`: "The source image is the base image being edited."

---

## Example Workflows

The `workflows/` directory contains six production-ready ComfyUI workflow JSON files:
- `subject_edit.json`: Single-reference subject identity editing workflow.
- `subject_scene.json`: Dual-reference subject placement into a scene.
- `subject_outfit.json`: Dual-reference subject identity and outfit transfer.
- `subject_outfit_scene.json`: Triple-reference subject, scene, and outfit composition.
- `subject_scene_qwen_simple.json`: Subject + scene workflow integrated with Simple Qwen-VL Vision Language Model prompt builder.
- `subject_scene_outfit_qwen_simple.json`: Subject + scene + outfit workflow integrated with Simple Qwen-VL VLM.

Each workflow is organized into distinct visual node groups (`Images`, `Models`, `Advanced Settings (disabled by default)`, `LoRA Stack + Edit + KSampler`, `Output`, and optional `Qwen Prompt Builder`).

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
  - `preset`: `balanced` | `max_identity` | `flexible`.
  - `output_resolution`: `subject` | `custom`.
  - `megapixels`: Float (default `1.0`).
  - `prompt_augmentation` (Optional): Socket input from `CcC Krea2 - LoRA Stack`.
  - `image_advanced_settings` (Optional): Socket input from `CcC Krea2 - Image Advanced Settings`.
  - `edit_advanced_settings` (Optional): Socket input from `CcC Krea2 - Edit Advanced Settings`.

---

### 2. CcC Krea2 - Subject + Outfit

Dual-reference subject identity and clothing/outfit editing node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[outfit, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image` (Required): Primary subject reference image.
  - `outfit_image` (Required): Reference garment or outfit image.
  - `preset`, `output_resolution`, `megapixels`
  - `prompt_augmentation` (Optional)

---

### 3. CcC Krea2 - Subject + Scene

Dual-reference subject identity and background scene composition node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[scene, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image` (Required): Primary subject reference image.
  - `scene_image` (Required): Target background scene image.
  - `preset`, `output_resolution`, `megapixels`
  - `prompt_augmentation` (Optional)

---

### 4. CcC Krea2 - Subject + Scene + Outfit

Triple-reference composition node.

- **Category**: `CcC/Krea2`
- **Reference Order**: `[scene, outfit, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image`, `scene_image`, `outfit_image` (Required)
  - `preset`, `output_resolution`, `megapixels`
  - `prompt_augmentation` (Optional)

---

### 5. CcC Krea2 - Inpaint

Localized inpainting node using a single source image and mask.

- **Category**: `CcC/Krea2`
- **Base Image**: `source_image`
- **Reference Order**: `[source]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `source_image` (Required): Base image to modify.
  - `inpaint_mask` (Optional): Mask defining edit region.
  - `preset`, `output_resolution`, `megapixels`
  - `prompt_augmentation` (Optional)

---

### 6. CcC Krea2 - Inpaint Subject + Outfit

Localized inpainting node for subject and outfit transfer.

- **Category**: `CcC/Krea2`
- **Base Image**: `subject_image`
- **Reference Order**: `[outfit, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `subject_image`, `outfit_image` (Required)
  - `inpaint_mask` (Optional)
  - `preset`, `output_resolution`, `megapixels`
  - `prompt_augmentation` (Optional)

---

### 7. CcC Krea2 - Inpaint Subject + Scene

Dual-reference localized inpainting node for subject placement into a scene.

- **Category**: `CcC/Krea2`
- **Base Image**: `scene_image`
- **Reference Order**: `[scene, subject]`
- **Inputs**:
  - `model`, `clip`, `vae`, `prompt` (Required)
  - `scene_image`, `subject_image` (Required)
  - `inpaint_mask` (Optional)
  - `preset`, `output_resolution`, `megapixels`
  - `prompt_augmentation` (Optional)

---

### 8. CcC Krea2 - Image Advanced Settings

Advanced settings node for fine-grained per-role image settings configuration.

Connecting Image Advanced Settings fully overrides the selected preset for that image role. To change only one value, keep the other visible values configured as desired because the entire role configuration is applied.

- **Category**: `CcC/Krea2`
- **Output**: `image_advanced_settings` (Custom Socket Type `CCC_KREA2_IMAGE_ADVANCED_SETTINGS`)
- **Chaining**: Connect an existing `image_advanced_settings` output into the optional `image_advanced_settings` input to chain multiple role overrides sequentially (e.g. `Subject` -> `Scene`). The last configuration for the same role wins.
- **Inputs**:
  - `role`: `subject` | `scene` | `outfit` | `source`
  - `boost`: Conditioning strength multiplier for this role.
  - `mask_invert`: Invert attention mask for this role.
  - `grounding_resize_mode`: `none` | `downscale_only` | `normalize` | `clamp`.
  - `grounding_px`: Grounding target pixel dimension.
  - `grounding_min_px`: Grounding minimum dimension floor.
  - `grounding_max_px`: Grounding maximum dimension ceiling.
  - `grounding_resize_method`: Resampling algorithm (`auto`, `nearest-exact`, `bilinear`, `bicubic`, `area`, `lanczos`).
  - `reference_fit_mode`: `fit` | `crop`.
  - `reference_resize_method`: Resampling algorithm (`auto`, `nearest-exact`, `bilinear`, `bicubic`, `area`, `lanczos`).

---

### 9. CcC Krea2 - Edit Advanced Settings

Advanced settings node for sampling parameters, attention mask modes, aspect ratio source controls, and role-based resolution limits.

- **Category**: `CcC/Krea2`
- **Output**: `edit_advanced_settings` (Custom Socket Type `CCC_KREA2_EDIT_ADVANCED_SETTINGS`)
- **Inputs**:
  - `batch_size`: Latent sampling batch size (1 - 64).
  - `sampling_resize_mode`: `fit` | `crop` | `stretch`.
  - `sampling_resize_method`: Resampling algorithm (`auto`, `nearest-exact`, `bilinear`, `bicubic`, `area`, `lanczos`).
  - `attention_mask_mode`: `hard` (binary threshold) | `soft` (continuous values).
  - `role_resolution_limit_mode`: `max_megapixels` (default) | `off`. Enables megapixel clamping for role-based output resolution.
  - `role_resolution_max_megapixels`: Float (default `2.0`, min `0.25`, max `12.0`, step `0.25`). Maximum megapixel threshold for role-based output resolution.
  - `custom_aspect_source`: `auto` | `subject` | `scene` | `source`.
  - `prompt_instructions_mode`: `automatic` | `append`.
  - `prompt_instructions`: Custom text instructions appended to system prompt.
  - `inpaint_mask_invert`, `inpaint_mask_grow`, `inpaint_mask_blur`: Mask preprocessing controls.

---

### 10. CcC Krea2 - LoRA Prompt Settings

Configures positive and negative prompt text augmentations associated with up to four LoRA slots.

- **Category**: `CcC/Krea2`
- **Output**: `lora_prompt_settings` (Custom Socket Type `CCC_KREA2_LORA_PROMPT_SETTINGS`)
- **Inputs**:
  - `enabled`: Boolean toggle to enable/disable prompt settings bundle.
  - `lora_1_prompt_enabled` .. `lora_4_prompt_enabled`: Per-slot prompt enablement toggles.
  - `lora_1_prompt_position` .. `lora_4_prompt_position`: Position for positive/negative text (`prepend` | `append`).
  - `lora_1_positive_prompt` .. `lora_4_positive_prompt`: Positive prompt text fragments.
  - `lora_1_negative_prompt` .. `lora_4_negative_prompt`: Negative prompt text fragments.

---

### 11. CcC Krea2 - LoRA Stack

Model-only LoRA loader supporting up to 4 chained model-only LoRAs and accumulated prompt augmentations.

- **Category**: `CcC/Krea2`
- **Outputs**:
  - `model`: Patched Krea 2 `MODEL`.
  - `prompt_augmentation` (Custom Socket Type `CCC_KREA2_PROMPT_AUGMENTATION`): Immutable prompt augmentation object containing accumulated positive/negative prepends and appends.
- **Inputs**:
  - `model` (Required): Input Krea 2 `MODEL`.
  - `enabled`: Global stack enablement toggle.
  - `global_strength`: Global strength multiplier applied to all slot strengths.
  - `lora_1_enabled` .. `lora_4_enabled`: Per-slot enablement toggles.
  - `lora_1_name` .. `lora_4_name`: LoRA filenames selected from ComfyUI `loras` folder.
  - `lora_1_strength` .. `lora_4_strength`: Per-slot strength values.
  - `prompt_augmentation` (Optional): Incoming accumulated `prompt_augmentation` from a preceding LoRA stack.
  - `lora_prompt_settings` (Optional): `lora_prompt_settings` bundle from `CcC Krea2 - LoRA Prompt Settings`.

---

### 12. CcC Krea2 - Text to Image

Native helper node for standard Krea 2 Text-to-Image generation using native CLIP tokenization and Empty SD3 Latent generation.

- **Category**: `CcC/Krea2`
- **Outputs**:
  - `model`: Input `MODEL` passed through unchanged.
  - `positive`: `CONDITIONING` generated via native CLIP text tokenization and scheduled encoding.
  - `negative`: `CONDITIONING` generated via native CLIP text tokenization and scheduled encoding.
  - `latent`: Empty SD3 `LATENT` dict aligned to 16-pixel multiples (min 128x128) for the selected aspect ratio and megapixel count.
- **Inputs**:
  - `model` (Required): Krea 2 `MODEL`.
  - `clip` (Required): Krea 2 `CLIP` text encoder.
  - `prompt` (Required): Multiline positive generation prompt.
  - `aspect_ratio` (Required): Output aspect ratio (`1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `3:2`, `2:3`, `custom`).
  - `megapixels` (Required): Target megapixel area (0.25 to 2.0 MP).
  - `batch_size` (Required): Batch size for empty latent creation (1 to 64).
  - `custom_aspect_width` / `custom_aspect_height` (Required): Ratio numerator and denominator used when `aspect_ratio` is set to `custom`.
  - `negative_prompt` (Optional): Multiline negative generation prompt.
  - `prompt_augmentation` (Optional): Incoming `CCC_KREA2_PROMPT_AUGMENTATION` socket from `CcC Krea2 - LoRA Stack`.

