# CcC Krea2 Node Reference

This document provides the complete specification of all public nodes in the `ComfyUI-CcC-Krea2` package, matching the Python input and output class definitions.

---

## 1. Socket Types

| Socket Type | Connected Between | Description |
|---|---|---|
| `PREPARED_VISION_IMAGE` | Layer 1 $\rightarrow$ Layer 3 | Contains untouched original image tensor, prepared vision image tensor, and preparation spec metadata. |
| `LATENT` | Layer 2 $\rightarrow$ Layer 5 | Standard ComfyUI latent dictionary containing `"samples"` tensor and geometry metadata. |
| `REFERENCE_CHAIN` | Layer 3 $\rightarrow$ Layer 4 $\rightarrow$ Layer 5 | Immutable linked chain storing ordered reference specifications. |
| `CCC_KREA2_PROMPT_AUGMENTATION` | LoRA Stack $\rightarrow$ Main Nodes | Immutable stack of loaded LoRAs and accumulated prompt augmentations. |

---

## 2. Modular 5-Layer Nodes

### 2.1 CcC Krea2 - Qwen Vision Image Prep
- **Class Name**: `CcCKrea2QwenVisionImagePrep`
- **Category**: `CcC/Krea2`
- **Description**: Prepares a derivative vision image (`vision_image`) optimized for Qwen Vision tokenization while preserving the untouched raw source image (`original_image`) for VAE processing.
- **Required Inputs**:
  - `clip` (`CLIP`): The Qwen3-VL text encoder instance.
  - `image` (`IMAGE`): Input image tensor (`[B, H, W, C]`).
  - `mode` (`["native", "adaptive", "fixed"]`, default: `"native"`): Sizing strategy.
  - `min_mp` (`FLOAT`, default: `0.0`, min: `0.0`, max: `12.0`, step: `0.01`): Minimum megapixels for adaptive mode.
  - `max_mp` (`FLOAT`, default: `1.0`, min: `0.0`, max: `12.0`, step: `0.01`): Maximum megapixels for adaptive mode.
  - `fixed_mp` (`FLOAT`, default: `1.0`, min: `0.1`, max: `12.0`, step: `0.01`): Target megapixels for fixed mode.
  - `downscale_method` (`["auto", "area", "bicubic", "bilinear", "lanczos", "nearest-exact"]`, default: `"auto"`): Interpolation method for downscaling.
  - `upscale_method` (`["auto", "bicubic", "bilinear", "lanczos", "nearest-exact"]`, default: `"auto"`): Interpolation method for upscaling.
- **Outputs**:
  - `prepared_image` (`PREPARED_VISION_IMAGE`): Prepared vision data container.
  - `vision_image` (`IMAGE`): Derivative image tensor.
  - `vision_info` (`STRING`): Text summary of dimensions and megapixels.

---

### 2.2 CcC Krea2 - Target Latent
- **Class Name**: `CcCKrea2TargetLatent`
- **Category**: `CcC/Krea2`
- **Description**: Creates the target latent container, independently specifying Latent Content Source and Target Geometry strategy.
- **Required Inputs**:
  - `target_content` (`["empty", "subject", "scene"]`, default: `"empty"`): Latent content source.
  - `geometry_mode` (`["favor_subject", "favor_scene", "fixed"]`, default: `"fixed"`): Latent output resolution geometry strategy.
  - `target_megapixels` (`FLOAT`, default: `2.0`, min: `0.1`, max: `12.0`, step: `0.01`): Target megapixels for geometry resolution when favoring subject or scene.
  - `fixed_megapixels` (`FLOAT`, default: `2.0`, min: `0.1`, max: `12.0`, step: `0.01`): Output megapixels when target geometry is `fixed`.
  - `aspect_ratio` (`["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"]`, default: `"1:1"`): Output aspect ratio when target geometry is `fixed`.
  - `batch_size` (`INT`, default: `1`, min: `1`, max: `64`, step: `1`): Latent batch size.
- **Optional Inputs**:
  - `vae` (`VAE`): VAE model for encoding content image if content is `subject` or `scene`.
  - `subject_image` (`PREPARED_VISION_IMAGE`): Image used when content is `subject` or geometry is `favor_subject`.
  - `scene_image` (`PREPARED_VISION_IMAGE`): Image used when content is `scene` or geometry is `favor_scene`.
- **Runtime Requirements**:
  - `vae` is required at execution time whenever `target_content` is `subject` or `scene`.
  - `subject_image` is required at execution time whenever `target_content` is `subject` or `geometry_mode` is `favor_subject`.
  - `scene_image` is required at execution time whenever `target_content` is `scene` or `geometry_mode` is `favor_scene`.
- **Outputs**:
  - `target_latent` (`LATENT`): Standard ComfyUI latent dictionary format.
  - `latent_info` (`STRING`): Text summary of target latent dimensions.

---

### 2.3 CcC Krea2 - Subject Reference Node
- **Class Name**: `CcCKrea2SubjectImage`
- **Category**: `CcC/Krea2`
- **Description**: Defines a Subject reference specification for character identity, face, body, and person features.
- **Required Inputs**:
  - `prepared_image` (`PREPARED_VISION_IMAGE`): Prepared image from Layer 1.
  - `visual_fit_mode` (`["auto", "fit", "crop"]`, default: `"auto"`): VAE reference fitting mode.
  - `attention_boost` (`FLOAT`, default: `1.0`, min: `0.1`, max: `10.0`, step: `0.05`): Spatial cross-attention boost.
  - `masked_attention_boost` (`FLOAT`, default: `1.0`, min: `0.1`, max: `10.0`, step: `0.05`): Attention boost inside mask region.
  - `pose_anchor` (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.05`): Pose text anchor weight (Vision directive only).
  - `outfit_anchor` (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.05`): Outfit text anchor weight (Vision directive only).
  - `masked_identity_anchor` (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.05`): Masked identity text anchor weight (Vision directive only).
  - `extra_vision_directive` (`STRING`, default: `""`): Custom Qwen text directive.
  - `vision_slot` (`INT`, default: `0`, min: `0`, max: `16`, step: `1`): Logical slot assignment (`0` for auto).
  - `aliases` (`STRING`, default: `""`): Custom alias template string.
- **Optional Inputs**:
  - `attention_mask` (`MASK`): Optional spatial attention mask.
  - `reference_chain` (`REFERENCE_CHAIN`): Chain of previous references.
- **Outputs**:
  - `reference_chain` (`REFERENCE_CHAIN`): Updated immutable reference chain.

---

### 2.4 CcC Krea2 - Scene Reference Node
- **Class Name**: `CcCKrea2SceneImage`
- **Category**: `CcC/Krea2`
- **Description**: Defines a Scene reference specification for background, composition, environment, and lighting.
- **Required Inputs**:
  - `prepared_image` (`PREPARED_VISION_IMAGE`): Prepared image from Layer 1.
  - `visual_fit_mode` (`["auto", "fit", "crop"]`, default: `"auto"`): VAE reference fitting mode.
  - `attention_boost` (`FLOAT`, default: `1.0`, min: `0.1`, max: `10.0`, step: `0.05`): Spatial cross-attention boost.
  - `masked_attention_boost` (`FLOAT`, default: `1.0`, min: `0.1`, max: `10.0`, step: `0.05`): Attention boost inside mask region.
  - `scene_anchor` (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.05`): Scene composition text anchor weight (Vision directive only).
  - `masked_region_anchor` (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.05`): Masked region text anchor weight (Vision directive only).
  - `extra_vision_directive` (`STRING`, default: `""`): Custom Qwen text directive.
  - `vision_slot` (`INT`, default: `0`, min: `0`, max: `16`, step: `1`): Logical slot assignment (`0` for auto).
  - `aliases` (`STRING`, default: `""`): Custom alias template string.
- **Optional Inputs**:
  - `attention_mask` (`MASK`): Optional spatial attention mask.
  - `reference_chain` (`REFERENCE_CHAIN`): Chain of previous references.
- **Outputs**:
  - `reference_chain` (`REFERENCE_CHAIN`): Updated immutable reference chain.

---

### 2.5 CcC Krea2 - Outfit Reference Node
- **Class Name**: `CcCKrea2OutfitImage`
- **Category**: `CcC/Krea2`
- **Description**: Defines an Outfit reference specification for garments and clothing without wearer identity.
- **Required Inputs**:
  - `prepared_image` (`PREPARED_VISION_IMAGE`): Prepared image from Layer 1.
  - `visual_fit_mode` (`["auto", "fit", "crop"]`, default: `"auto"`): VAE reference fitting mode.
  - `attention_boost` (`FLOAT`, default: `1.0`, min: `0.1`, max: `10.0`, step: `0.05`): Spatial cross-attention boost.
  - `masked_attention_boost` (`FLOAT`, default: `1.0`, min: `0.1`, max: `10.0`, step: `0.05`): Attention boost inside mask region.
  - `outfit_anchor` (`FLOAT`, default: `0.0`, min: `0.0`, max: `1.0`, step: `0.05`): Outfit detail text anchor weight (Vision directive only).
  - `extra_vision_directive` (`STRING`, default: `""`): Custom Qwen text directive.
  - `vision_slot` (`INT`, default: `0`, min: `0`, max: `16`, step: `1`): Logical slot assignment (`0` for auto).
  - `aliases` (`STRING`, default: `""`): Custom alias template string.
- **Optional Inputs**:
  - `attention_mask` (`MASK`): Optional spatial attention mask.
  - `reference_chain` (`REFERENCE_CHAIN`): Chain of previous references.
- **Outputs**:
  - `reference_chain` (`REFERENCE_CHAIN`): Updated immutable reference chain.

---

### 2.6 CcC Krea2 - Style Reference Node
- **Class Name**: `CcCKrea2StyleImage`
- **Category**: `CcC/Krea2`
- **Description**: Defines a Style reference specification for color palette, lighting, texture, and Moodboard processing.
- **Required Inputs**:
  - `prepared_image` (`PREPARED_VISION_IMAGE`): Prepared image from Layer 1.
  - `style_processing` (`["2x2", "4x4", "full"]`, default: `"2x2"`): Crop tile expansion mode.
  - `style_fidelity` (`FLOAT`, default: `0.5`, min: `0.0`, max: `1.0`, step: `0.05`): Statistical style fidelity weight.
  - `indirect_style_transfer` (`BOOLEAN`, default: `True`): Whether to remove style vision rows after encoding.
  - `style_directive` (`BOOLEAN`, default: `True`): Whether to append automatic style text directives.
  - `extra_vision_directive` (`STRING`, default: `""`): Custom Qwen text directive.
  - `vision_slot` (`INT`, default: `0`, min: `0`, max: `16`, step: `1`): Logical slot assignment (`0` for auto).
  - `aliases` (`STRING`, default: `""`): Custom alias template string.
- **Optional Inputs**:
  - `reference_chain` (`REFERENCE_CHAIN`): Chain of previous references.
- **Outputs**:
  - `reference_chain` (`REFERENCE_CHAIN`): Updated immutable reference chain.

---

### 2.7 CcC Krea2 - Edit Orchestrator
- **Class Name**: `CcCKrea2Edit`
- **Category**: `CcC/Krea2`
- **Description**: Layer 5 orchestrator node executing Qwen tokenization, Moodboard transforms, model patching, and report formatting.
- **Required Inputs**:
  - `model` (`MODEL`): Diffusion model instance.
  - `clip` (`CLIP`): Qwen3-VL text encoder instance.
  - `vae` (`VAE`): VAE model instance.
  - `references` (`REFERENCE_CHAIN`): Resolved reference chain from Layer 4.
  - `target_latent` (`LATENT`): Resolved target latent container from Layer 2.
  - `positive_prompt` (`STRING`, default: `""`, multiline: `True`): Positive text prompt.
  - `negative_prompt` (`STRING`, default: `""`, multiline: `True`): Negative text prompt.
- **Optional Inputs**:
  - `prompt_augmentation` (`CCC_KREA2_PROMPT_AUGMENTATION`): Optional LoRA prompt settings stack.
  - `global_vision_directive` (`STRING`, default: `""`, multiline: `True`): System vision directive.
- **Outputs**:
  - `patched_model` (`MODEL`): Patched diffusion model.
  - `positive` (`CONDITIONING`): Positive conditioning.
  - `negative` (`CONDITIONING`): Negative conditioning.
  - `latent` (`LATENT`): Target latent container.
  - `edit_info` (`STRING`): Comprehensive execution diagnostic text report.

---

## 3. Visual Reference Fit Modes

Public modes: `auto`, `fit`, `crop`.

Internal Auto resolution outcomes:
- `exact`: Input dimensions match target dimensions exactly.
- `crop_only`: Aspect ratio matches within tolerance (0.01); resizes without letterboxing to exact target dimensions. `crop_only` is Auto-only.
- `crop_and_resize`: Fills exact target dimensions via aspect-preserving crop and scale.
- `fit`: Preserves original reference geometry using `/16`-aligned aspect ratio fitting.
- Manual `crop`: Fills exact target dimensions.

---

## 4. LoRA & T2I Nodes

- **`CcCKrea2LoRAStack`**: Combines up to 4 LoRAs with global and per-slot strength controls.
- **`CcCKrea2LoRAPromptSettings`**: Configures slot-level positive and negative prompt text fragments.
- **`CcCKrea2TextToImage`**: Executes native text-to-image generation.
