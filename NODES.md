# CcC Krea2 Node Reference

This document provides the complete specification of all public nodes in the `ComfyUI-CcC-Krea2` package.

---

## 1. Easy Nodes

### 1.1 CcC Krea2 - Easy Edit & Ostris Easy Edit
- **Class Name**: `CcCKrea2EasyEdit` / `CcCKrea2EasyEditOstris`
- **Category**: `CcC/Krea2`
- **Description**: Opinionated all-in-one nodes that bypass manual Reference Chain construction. Implements a unified policy-driven routing matrix to automatically assemble semantic instructions, style processing, and identity preservation based on a single workflow preset.
- **Required Inputs**:
  - `model` (`MODEL`), `clip` (`CLIP`), `vae` (`VAE`): Core models.
  - `positive_prompt` (`STRING`, default: `""`, multiline: `True`): Positive text prompt.
  - `use_default_prompt` (`BOOLEAN`, default: `True`): Controls system-managed default prompt resolution vs custom user prompt.
  - `preset` (`CHOICE`): `flexible`, `balanced`, `consistent`, `preserve_identity`, `max_identity`, `identity_transfer`, `subject_transfer_1`, `flexible_subject_transfer_1`, `flexible_subject_transfer_2`, `preserve_scene`, `outfit_transfer`, `style_transfer`, `scene_reinterpretation`.
  - `outfit_source` (`CHOICE`): `none`, `subject image`, `scene image`, `outfit image`, `style image` (default: `outfit image`).
  - `style_source` (`CHOICE`): `none`, `scene image`, `subject image`, `style image` (default: `none`). Selectable Style sources use indirect conditioning with a style-only semantic guardrail unless a preset locks a custom Style profile.
  - `flexible_subject_transfer_1/2` restrict `outfit_source` to `none`, `subject image` (default), or `scene image`. The selected Outfit uses a separate direct semantic Style path and can coexist with the independently selected indirect artistic Style.
  - Easy appearance-reference fitting is role-specific: Subject and Outfit use `contain_no_upscale`; Scene and combined Scene+Outfit use `contain`.
  - With Subject + Scene, target geometry is Subject-aware: Scene is first capped at 2 MP; the latent then expands with Scene aspect ratio only when needed to contain Subject, up to 2 MP. Subject is downscaled only if it still cannot fit at that limit.
  - `apply_krea2_edit_patch` (`BOOLEAN`, default: `True`): For `CcCKrea2EasyEdit`.
  - `apply_ostris_edit_patch` (`BOOLEAN`, default: `True`): For `CcCKrea2EasyEditOstris`.
  - `ostris_kv_cache` (`BOOLEAN`, default: `False`): For `CcCKrea2EasyEditOstris`.

#### System-Managed Default Prompts (`use_default_prompt`) & Preset Conversion

Easy Edit nodes provide centralized default positive prompt resolution:

- **Default Mode (`use_default_prompt = true`)**: The node automatically resolves an optimized positive prompt based on the active preset and connected reference images. The positive prompt widget displays the actual resolved system prompt and is set to read-only. Changing presets or inputs automatically refreshes the displayed default prompt.
- **Custom Mode (`use_default_prompt = false`)**: Gives complete prompt control to the user. The positive prompt widget is enabled and editable, preserving user-entered text without modification when presets or input connections change.
- **Use Preset as Custom Button**: Action button widget that converts the resolved preset prompt into custom text, sets `use_default_prompt = false`, and unlocks the positive prompt text field.
- **Subject-only Exception**: When only a Subject reference is connected, no default edit intent exists. `use_default_prompt` is disabled/greyed out and forced to custom mode. A positive prompt is required; an empty prompt raises a validation error.

##### Default Prompt Keys and Canonical Templates

- `outfit_transfer`: `Transfer only the outfit and accessories from the outfit reference to the subject. Preserve the subject identity, body, pose, framing, and composition. Do not preserve the subject clothing. Fit the transferred outfit and accessories naturally to the subject. Keep accessories physically attached to the subject in a natural way and never floating. Do not duplicate accessories.`
- `subject_scene`: `Place the subject from the subject reference naturally into the scene reference. Preserve the subject identity, body shape, and body proportions. Preserve the scene composition, environment, framing, perspective, and spatial layout. Adapt the subject naturally to the scene lighting and environment.`
- `subject_scene_outfit`: `Place the subject from the subject reference naturally into the scene reference wearing the outfit and accessories from the outfit reference. Preserve the subject identity, body shape, and body proportions. Preserve the scene composition, environment, framing, perspective, and spatial layout. Do not preserve the subject clothing. Fit the transferred outfit and accessories naturally to the subject and the scene. Keep accessories physically attached to the subject in a natural way and never floating. Do not duplicate accessories.`
- `style`: `Use the style image only as a visual style reference. Apply its color palette, lighting character, contrast, texture, rendering treatment, photographic treatment, and overall visual mood. Do not transfer subjects, identities, facial features, hair, anatomy, body shapes, clothing, accessories, poses, objects, environment, layout, framing, or scene composition from the style image.`
- `identity_transfer`: canonical identity-only replacement prompt preserving the Scene subject's position, action, pose, role, interaction, clothing, and accessories.
- `subject_transfer_1` (displayed as **Subject Transfer**): transfers the complete Subject while matching the Scene character's position, pose, action, role, and interactions. Scene is a guarded direct appearance reference at `1.0`; Subject is a guarded direct appearance reference at `2.5`; Subject clothing is reinforced through an automatic direct Outfit semantic reference; Scene also provides automatic indirect Style guidance.
- `flexible_subject_transfer_1` / `flexible_subject_transfer_2`: complete-subject replacement with Scene and Subject occupying the two appearance references. Outfit is selectable from `none`, Subject (default), or Scene and enters as a direct semantic Style reference with outfit-only guardrails. Artistic Style is independently selectable from `none`, Scene, Subject, or Style and always enters indirectly with the default style-only guardrail.
- `scene_reinterpretation`: Empty latent with Scene geometry, Subject appearance `7.0`, optional Outfit appearance `4.0`, and Scene locked as a direct custom Style reference. The Style instruction recreates composition, framing, environment, objects, people, lighting, pose, action, role, interactions, and spatial relationships while excluding the replaced subject's identity, anatomy, clothing, and accessories. Scene is no longer duplicated as a semantic-only reference.
- `scene_reinterpretation_outfit`: Outfit defaults to Scene and remains selectable from none, Subject, Scene, Outfit, or Style. Automatic prompts identify Scene clothing through `{reference_subject}` and Subject clothing through `{subject}`; Outfit and Style images are interpreted visually.

When `use_default_prompt = true`, Outfit selection is placeholder-aware for every preset and every technical Outfit path: Scene targets the clothing worn by `{reference_subject}`, Subject targets the clothing worn by `{subject}`, Outfit Image selects the principal identified outfit, and Style Image interprets relevant clothing. In custom prompt mode, Outfit selection is purely visual and does not use placeholders.

- **Optional Inputs**:
  - `subject`, `scene`, `outfit`, `style` (`IMAGE`): Visual references for routing.
  - `negative_prompt` (`STRING`): Negative text prompt.
- **Outputs**:
  - `patched_model` (`MODEL`), `positive` (`CONDITIONING`), `negative` (`CONDITIONING`), `latent` (`LATENT`), `edit_info` (`STRING`).

#### Preset Behavior & Identity Ladder

The identity presets form a progression representing increasing reference/identity anchoring and, in general, decreasing editing freedom:

```
Flexible
   ↓
Balanced
   ↓
Consistent
   ↓
Preserve Identity
   ↓
Max Identity
```

| Preset | Behavior |
| --- | --- |
| **Flexible** | Maximum editing freedom with neutral reference influence. |
| **Balanced** | Balance between reference consistency and editing freedom. |
| **Consistent** | Stronger consistency with the Subject while keeping composition and editing relatively flexible. |
| **Preserve Identity** | Strong identity preservation with Subject anchoring. |
| **Max Identity** | Maximum identity anchoring; reduced editing freedom is acceptable when needed to preserve identity. |

#### Technical Subject-Only Routing Contracts

The table below details the technical routing contract resolved when running Easy Edit with a single **Subject** image connected:

| Preset | Target Content | Subject Attention | Main Behavior |
| --- | --- | ---: | --- |
| **Flexible** | Empty | 1.0 | Maximum editing freedom |
| **Balanced** | Empty | 2.5 | Moderate balance |
| **Consistent** | Empty | 4.0 | Stronger Subject consistency |
| **Preserve Identity** | Subject image | 6.0 | Strong identity preservation |
| **Max Identity** | Subject image | 10.0 | Maximum identity anchoring |

> [!NOTE]
> **Subject Attention Boost Values**: An attention boost of `1.0` represents neutral reference attention (no extra amplification). The reference image still fully participates in Qwen Vision tokenization, VAE reference conditioning, and model patching. Boost values `>1.0` apply additional attention weight to the reference during Krea2 Edit RoPE/attention patching.

> [!IMPORTANT]
> **Multi-Reference Routing Warning**: These values describe the Subject-only Easy Edit contract. Multi-reference routing can use different target/reference combinations depending on Scene, Outfit, Style, and source selectors. The 3-phase routing engine in `ccc_krea2/easy_routing.py` remains authoritative.

#### Subject + Scene Routing Contracts

The table below details the technical routing contract resolved when running Easy Edit with both a **Subject** image and a **Scene** image connected (with Outfit and Style disconnected):

| Preset | Target Content | Geometry Source | Scene Attention | Subject Attention | Primary Intent |
| --- | --- | --- | ---: | ---: | --- |
| **Flexible** | Empty | Scene | 1.0 | 1.0 | Maximum editing freedom |
| **Balanced** | Empty | Scene | 1.0 | 2.5 | General balance |
| **Consistent** | Empty | Scene | 1.0 | 4.0 | Stronger Subject consistency |
| **Preserve Identity** | Subject image | Subject | 1.0 | 6.0 | Strong identity preservation |
| **Max Identity** | Subject image | Subject | 1.0 | 10.0 | Maximum identity anchoring |
| **Preserve Scene** | Scene image | Scene | 2.5 | 1.0 | Preserve Scene composition/context |

- **Structural Target & Geometry Transition**:
  - `Flexible`, `Balanced`, and `Consistent` use `Target Content = Empty` and `Geometry = Scene`. The Scene image provides output geometry and compositional guidance while target latents start from noise.
  - `Preserve Identity` and `Max Identity` transition to `Target Content = Subject` and `Geometry = Subject`. The Subject becomes structurally dominant, while the Scene remains available as Slot 1 appearance reference guidance.
  - `Preserve Scene` sets `Target Content = Scene` and `Geometry = Scene` with amplified Scene attention (`2.5`), prioritizing environmental context while maintaining the Subject as a normal reference.

- **Appearance Reference Ordering**:
  For all six presets listed above when Subject + Scene are connected, appearance references are assigned in the exact order: **Slot 1 = Scene**, **Slot 2 = Subject**.

---

## 2. Advanced Nodes

### 2.1 CcC Krea2 - Qwen Vision Image Prep
- **Class Name**: `CcCKrea2QwenVisionImagePrep`
- **Category**: `CcC/Krea2`
- **Description**: Prepares a derivative vision image (`vision_image`) optimized for Qwen Vision tokenization while preserving the untouched raw source image (`original_image`) for VAE processing.
- **Required Inputs**:
  - `clip` (`CLIP`), `image` (`IMAGE`), `mode` (`CHOICE`), `min_mp` (`FLOAT`), `max_mp` (`FLOAT`), `fixed_mp` (`FLOAT`), `downscale_method` (`CHOICE`), `upscale_method` (`CHOICE`).
- **Outputs**:
  - `prepared_image` (`PREPARED_VISION_IMAGE`), `vision_image` (`IMAGE`), `vision_info` (`STRING`).

### 2.2 CcC Krea2 - Target Latent
- **Class Name**: `CcCKrea2TargetLatent`
- **Category**: `CcC/Krea2`
- **Description**: Creates the target latent container.
- **Required Inputs**:
  - `target_content` (`CHOICE`), `geometry_mode` (`CHOICE`), `target_megapixels` (`FLOAT`), `fixed_megapixels` (`FLOAT`), `aspect_ratio` (`CHOICE`), `batch_size` (`INT`).
- **Optional Inputs**:
  - `include_in_vision` (`CHOICE`: `auto`, `yes`, `no`), `target_vision_slot` (`CHOICE`), `target_alias` (`STRING`), `target_vision_instruction` (`STRING`), `vae` (`VAE`), `target_image` (`PREPARED_VISION_IMAGE`), `geometry_image` (`PREPARED_VISION_IMAGE`).
- **Legacy Compatibility Optional Inputs**:
  - `subject_image` (`PREPARED_VISION_IMAGE`), `scene_image` (`PREPARED_VISION_IMAGE`).
- **Outputs**:
  - `target_latent` (`LATENT`), `latent_info` (`STRING`).

### 2.3 CcC Krea2 - Reference Image
- **Class Name**: `CcCKrea2ReferenceImage`
- **Category**: `CcC/Krea2`
- **Description**: Generic reference configuration for edit (appearance/semantic) and style paths.
- **Required Inputs**:
  - `reference_path` (`CHOICE`), `prepared_image` (`PREPARED_VISION_IMAGE`), `vision_slot` (`CHOICE`), `alias` (`STRING`), `vision_instruction` (`STRING`), `attention_boost` (`FLOAT`), `masked_attention_boost` (`FLOAT`), `visual_reference_fit` (`CHOICE`), `style_fidelity` (`FLOAT`), `style_processing` (`CHOICE`), `indirect_style_transfer` (`BOOLEAN`).
- **Optional Inputs**:
  - `previous_references` (`REFERENCE_CHAIN`), `attention_mask` (`MASK`).
- **Outputs**:
  - `reference_chain` (`REFERENCE_CHAIN`).

### 2.4 CcC Krea2 - Edit Orchestrator
- **Class Name**: `CcCKrea2Edit`
- **Category**: `CcC/Krea2`
- **Description**: Layer 5 orchestrator node executing Qwen tokenization, model patching, and report formatting.
- **Required Inputs**:
  - `model` (`MODEL`), `clip` (`CLIP`), `vae` (`VAE`), `references` (`REFERENCE_CHAIN`), `target_latent` (`LATENT`), `positive_prompt` (`STRING`), `negative_prompt` (`STRING`).
- **Optional Inputs**:
  - `global_vision_directive` (`STRING`), `reference_method` (`CHOICE`), `ostris_kv_cache` (`BOOLEAN`), `prompt_augmentation` (`CCC_KREA2_PROMPT_AUGMENTATION`).
- **Outputs**:
  - `patched_model` (`MODEL`), `positive` (`CONDITIONING`), `negative` (`CONDITIONING`), `latent` (`LATENT`), `edit_info` (`STRING`).
- **Notes**:
  - Backend `reference_method` values: `native` (ComfyUI native standard reference logic), `krea2_edit` (the original Krea2 backend), `ostris_edit` (the Ostris upstream-aligned backend).
  - `ostris_kv_cache=true` is currently unsupported and raises `NotImplementedError` when requested.

---

## 3. Utilities

- **CcCKrea2LoRAStack**: Combines up to 4 LoRAs with global and per-slot strength controls.
- **CcCKrea2LoRAPromptSettings**: Configures slot-level positive and negative prompt text fragments.
- **CcCKrea2TextToImage**: Executes native text-to-image generation.

---

## 4. Legacy Nodes

The following nodes are deprecated in favor of `CcCKrea2ReferenceImage` and the Easy presets, but remain for backward compatibility:
- **CcCKrea2SubjectImage**
- **CcCKrea2SceneImage**
- **CcCKrea2OutfitImage**
- **CcCKrea2StyleImage**

---

## 5. Visual Reference Fit Modes

Public modes: `auto`, `fit`, `crop`.

### Internal Auto Resolution Outcomes (`auto`)
- **`exact`**: Selected when source dimensions match target dimensions exactly (`src == tgt`). Performs no crop, no resize, and no interpolation.
- **`crop_only`**: Selected only inside `auto` when the source is slightly larger than the target (`src >= tgt`), requiring at most 5% discarded per dimension (`dw_pct <= 0.05` and `dh_pct <= 0.05`) and at most 10% total area discarded (`area_discarded <= 0.10`). Performs an exact target-size center crop without resize or interpolation.
- **`crop_and_resize`**: Selected for the upstream near-match aspect ratio path when dimensional coverage is at least 92% (`coverage_h >= 0.92` and `coverage_w >= 0.92`). Performs a center crop to target aspect ratio followed by bicubic interpolation and antialiasing to exact target pixel dimensions.
- **`fit`**: Selected for genuine aspect ratio mismatches (`coverage < 0.92`). Performs a Krea2Edit-compatible source crop and resizes to `/16`-aligned VAE reference dimensions with fractional centered RoPE offsets, preserving reference geometry without black or gray pixel padding canvas.
