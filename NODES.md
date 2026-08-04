# CcC Krea2 Node Reference

Complete user-facing documentation for all nodes in the **CcC Krea2** suite, including both classic single-node workflows and the decoupled 5-Layer Modular Pipeline.

---

## 5-Layer Decoupled Modular Architecture

CcC Krea2 provides a modern, 5-layer decoupled architecture for precise image editing and reference composition:

1. **Layer 1: Input & Vision Preparation (`CcCKrea2QwenVisionImagePrep`)**
   Processes input images via Qwen-VL tokenization, yielding reusable, pre-computed `CCC_KREA2_PREPARED_IMAGE` tokens and dual-path geometry bounds.

2. **Layer 2: Declarative Reference Chain (`CcCKrea2SubjectImage`, `CcCKrea2SceneImage`, `CcCKrea2OutfitImage`, `CcCKrea2StyleImage`)**
   Defines atomic reference roles with specific weights, fit strategies (`auto`, `fit`, `crop`), prompt alias templates, anchor ranges (0.0–1.0), and attention masks, chaining them into an immutable `CCC_KREA2_REFERENCE_CHAIN`.

3. **Layer 3: Target Latent (`CcCKrea2TargetLatent`)**
   Computes or extracts the target latent canvas (`empty`, `subject`, `scene`, `inpaint`, `custom`) using an automated 4-way visual reference fit resolver (`fixed`, `fit_subject`, `fit_scene`, `crop_subject`).

4. **Layer 4: Styling & LoRA Stack (`CcCKrea2LoRAStack`, `CcCKrea2LoRAPromptSettings`)**
   Applies model-only LoRAs and prompt augmentations cleanly decoupled from reference role declarations.

5. **Layer 5: Orchestration Engine (`CcCKrea2Edit`)**
   Consolidates the reference chain, target latent, and styling inputs, executing model patching and conditioning generation in a single atomic step.

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
---

## Modular Pipeline Node Reference

### 13. CcC Krea2 - Qwen Vision Image Prep

Vision preparation layer node for processing input images into reusable vision embeddings via Qwen-VL.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `prepared_image`: `CCC_KREA2_PREPARED_IMAGE` socket containing VAE latents and Qwen vision tokens.
  - `vision_image`: `IMAGE` tensor after vision resize processing.
  - `vision_info`: Human-readable `STRING` summary of preparation parameters.
- **Inputs**:
  - `clip` (Required): `CLIP` text/vision encoder.
  - `image` (Required): `IMAGE` input tensor.
  - `preset`: `native` | `balanced` | `max_identity` | `flexible`.
  - `target_megapixels`: Target area scaling factor (float, default `1.0`).
  - `fit_mode`: `auto` | `fit` | `crop` | `stretch`.
  - `resize_method`: Resampling algorithm (`auto`, `nearest-exact`, `bilinear`, `bicubic`, `area`, `lanczos`).

---

### 14. CcC Krea2 - Subject Image Reference

Declarative reference node for defining subject identity inputs in the reference chain.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `references`: `CCC_KREA2_REFERENCE_CHAIN` immutable reference stack output.
- **Inputs**:
  - `prepared_image` (Required): Input `CCC_KREA2_PREPARED_IMAGE` from Vision Prep.
  - `attention_mask` (Optional): Spatial `MASK` for subject isolating attention.
  - `previous_references` (Optional): Incoming `CCC_KREA2_REFERENCE_CHAIN` to append onto.
  - `subject_preset`: `auto` | `balanced` | `max_identity` | `flexible`.
  - `aliases`: Prompt replacement tokens (e.g. `subject_image, Image {slot}`).
  - `vision_directive`: Optional text instruction for Qwen vision reasoning.
  - `subject_boost`, `grounding_boost`, `reference_boost`: Fine-grained attention multipliers.
  - `fit_mode`: `auto` | `fit` | `crop` | `stretch`.

---

### 15. CcC Krea2 - Scene Image Reference

Declarative reference node for scene composition, environment, and lighting inputs.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `references`: `CCC_KREA2_REFERENCE_CHAIN` immutable reference stack output.
- **Inputs**:
  - `prepared_image` (Required): Input `CCC_KREA2_PREPARED_IMAGE` from Vision Prep.
  - `attention_mask` (Optional): Spatial `MASK` for scene masking.
  - `previous_references` (Optional): Incoming `CCC_KREA2_REFERENCE_CHAIN`.
  - `scene_preset`: `auto` | `balanced` | `max_identity` | `flexible`.
  - `aliases`: Prompt replacement tokens (e.g. `scene_image, Image {slot}`).
  - `vision_directive`: Text instruction for scene reasoning.
  - `scene_boost`, `grounding_boost`, `reference_boost`: Attention multipliers.
  - `fit_mode`: `auto` | `fit` | `crop` | `stretch`.

---

### 16. CcC Krea2 - Outfit Image Reference

Declarative reference node for outfit and garment details.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `references`: `CCC_KREA2_REFERENCE_CHAIN` immutable reference stack output.
- **Inputs**:
  - `prepared_image` (Required): Input `CCC_KREA2_PREPARED_IMAGE` from Vision Prep.
  - `attention_mask` (Optional): Garment spatial `MASK`.
  - `previous_references` (Optional): Incoming `CCC_KREA2_REFERENCE_CHAIN`.
  - `outfit_preset`: `auto` | `balanced` | `max_identity` | `flexible`.
  - `aliases`: Prompt replacement tokens (e.g. `outfit_image, Image {slot}`).
  - `vision_directive`: Text instruction for garment details.
  - `outfit_boost`, `grounding_boost`, `reference_boost`: Attention multipliers.
  - `fit_mode`: `auto` | `fit` | `crop` | `stretch`.

---

### 17. CcC Krea2 - Style Image Reference

Declarative reference node for artistic style, color grade, or moodboard transfer.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `references`: `CCC_KREA2_REFERENCE_CHAIN` immutable reference stack output.
- **Inputs**:
  - `prepared_image` (Required): Input `CCC_KREA2_PREPARED_IMAGE` from Vision Prep.
  - `previous_references` (Optional): Incoming `CCC_KREA2_REFERENCE_CHAIN`.
  - `aliases`: Prompt replacement tokens (`style_image`).
  - `vision_directive`: Style directive prompt.
  - `style_boost`: Global style strength multiplier.
  - `moodboard_layout`: Grid placement layout (`single`, `2x2`, `3x3`).
  - `enable_color_transfer`, `enable_texture_transfer`: Stylization toggles.

---

### 18. CcC Krea2 - Target Latent

Target latent canvas creation and reference geometry resolver node.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `target_latent`: Formatted target `LATENT` dict.
  - `latent_info`: Human-readable `STRING` summary of latent dimensions.
- **Inputs**:
  - `vae` (Required): `VAE` encoder/decoder.
  - `subject_image` / `scene_image` (Optional): Prepared image reference sources for automatic aspect ratio and geometry extraction.
  - `target_content`: `empty` | `subject` | `scene` | `inpaint` | `custom`.
  - `geometry_mode`: `fixed` | `fit_subject` | `fit_scene` | `crop_subject`.
  - `target_megapixels`: Output canvas megapixels (float, default `1.0`).
  - `aspect_ratio`: Target aspect ratio string (`1:1`, `4:3`, `16:9`, etc.).
  - `batch_size`: Latent batch count.

---

### 19. CcC Krea2 - Edit (Modular Orchestrator)

Modular edit orchestrator node that combines reference chains, target latents, and model patches to produce sampling conditioning and latents.

- **Category**: `CcC/Krea2/Modular`
- **Outputs**:
  - `model`: Patched Krea 2 `MODEL`.
  - `positive`: Target positive `CONDITIONING` with Qwen vision tokens.
  - `negative`: Target negative `CONDITIONING`.
  - `latent`: Target `LATENT` passed through to KSampler.
  - `edit_info`: `STRING` execution report.
- **Inputs**:
  - `model` (Required): Krea 2 `MODEL`.
  - `clip` (Required): Krea 2 `CLIP` text encoder.
  - `vae` (Required): Krea 2 `VAE`.
  - `references` (Optional): `CCC_KREA2_REFERENCE_CHAIN` from Layer 2.
  - `target_latent` (Optional): Target `LATENT` from Layer 3 (`CcCKrea2TargetLatent`).
  - `prompt` (Required): Text prompt describing the target edit.
  - `negative_prompt` (Optional): Negative prompt text.
  - `global_vision_directive` (Optional): Multiline global vision directive text.
  - `prompt_augmentation` (Optional): Socket input from `CcC Krea2 - LoRA Stack`.


