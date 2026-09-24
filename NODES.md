# Krea2 CcC Edit Nodes

This document describes the **14 public nodes registered in ccc_krea2/nodes.py**. Internal helper classes and modules are intentionally excluded.

## Public Node Inventory

| Display name | Node id |
| --- | --- |
| Krea2 CcC Attention Region | CcCKrea2AttentionRegion |
| Krea2 CcC Visual Reference | CcCKrea2VisualReference |
| Krea2 CcC Semantic Reference | CcCKrea2SemanticReference |
| Krea2 CcC Size Resolver | CcCKrea2SizeResolver |
| Krea2 CcC Latent | CcCKrea2Latent |
| Krea2 CcC Edit Prompt Creator | CcCKrea2EditPromptCreator |
| Krea2 CcC Edit | CcCKrea2Edit |
| Krea2 CcC Character Sheet | CcCKrea2CharacterSheet |
| Krea2 CcC Paint Prepare | CcCKrea2PaintPrepare |
| Krea2 CcC Paint | CcCKrea2Paint |
| Krea2 CcC Paint Restore | CcCKrea2PaintRestore |
| CcC Krea2 - LoRA Prompt Settings | CcCKrea2LoRAPromptSettings |
| CcC Krea2 - LoRA Stack | CcCKrea2LoRAStack |
| CcC Krea2 - Text to Image | CcCKrea2TextToImage |

There are no public Reference Cache nodes and no public Krea2 CcC Paint Geometry node.

## Krea2 CcC Attention Region

Declares one tagged rectangular target region. Multiple nodes can be chained.

| Control | Values | Default | Meaning |
| --- | --- | --- | --- |
| tag | text | subject | Unique region name used by Visual Reference. |
| x | 0..100 | 0 | Left edge as target/content percentage. |
| y | 0..100 | 0 | Top edge as target/content percentage. |
| width | 0.1..100 | 100 | Width as percentage. |
| height | 0.1..100 | 100 | Height as percentage. |
| previous_regions | KREA2_ATTENTION_REGION_CHAIN | optional | Previous region chain. |

Output: attention_regions.

Latent transports these boxes through target geometry. If content_fit=crop touches or removes any part of a declared region, Latent raises an error instead of clipping or relocating the box.

## Krea2 CcC Visual Reference

Declares one ordered appearance reference.

Required controls:

| Control | Values | Default | Meaning |
| --- | --- | --- | --- |
| image | IMAGE | — | Source reference. |
| boost | 0..10 | 1.0 | Positive target-to-reference attention boost. |
| reference_fit | crop, resize, contain, native | native | Appearance/VAE pixel fit. |
| placement_grid | inside, outside | inside | RoPE placement grid; crop forces inside. |
| grid_horizontal_position | center, left, right | center | Horizontal RoPE/grid position. |
| grid_vertical_position | center, up, down | center | Vertical RoPE/grid position. |
| resize_method | lanczos, bicubic, bilinear, area | lanczos | Interpolation for resize/contain. |
| semantic | boolean | true | Also show this image to Qwen. |
| semantic_resize | boolean | true | Downscale Qwen image only when above the cap. |
| semantic_grounding_px | 32..4096, step 32 | 768 | Qwen longest-edge cap. |
| semantic_resize_method | lanczos, bicubic, bilinear, area | lanczos | Qwen-only resize interpolation. |
| prompt_annotation | text | empty | Role-only Image N annotation. |
| attention_scope | global, boost in region, only in region | global | Target-side reference scope. |
| region_tag | text | empty | Attention Region tag for regional modes. |

Optional input: previous_references.

Output: visual_references.

### Appearance fit

- crop: the target acts as an inside crop window over the source.
- resize: map the source long edge to the corresponding target axis, preserve aspect, then crop down only the minimum pixels needed for /16.
- contain: uniformly fit inside the target, preserve aspect, then crop down only the minimum pixels needed for /16.
- native: keep 1:1 pixel scale and center-crop width/height down to /16.

Qwen preparation is independent from appearance/VAE fit.

prompt_annotation is identification only, for example: This is the subject image. Edit instructions belong in Edit's positive_prompt.

The positive branch uses the configured boost. The grounded negative branch uses the same appearance-image vision blocks with no role annotation text, fixed empty negative text, and neutral boost 1.0.

## Krea2 CcC Semantic Reference

Adds one Qwen-only semantic/style reference. It never creates an appearance VAE latent.

| Control | Values | Default |
| --- | --- | --- |
| image | IMAGE | — |
| mode | semantic_only, style_direct, style_indirect | semantic_only |
| instruction | text | empty |
| grounding_px | 0..4096, step 32 | 768 |
| processing | full, 2x2, 4x4 | full |
| fidelity | 0..1 | 0.5 |
| previous_references | KREA2_SEMANTIC_REFERENCE_CHAIN | optional |

Output: semantic_references.

semantic_only always resolves processing to full. Tiled processing is used only by style modes.

## Krea2 CcC Size Resolver

Inputs:

- long_edge_image
- aspect_ratio_image

Outputs:

- width
- height

The node takes the longest edge from long_edge_image and the width/height ratio from aspect_ratio_image. It carries no image content forward. /16 alignment is handled by Latent.

## Krea2 CcC Latent

Builds the target LATENT and owns target geometry.

### Required controls

| Control | Values | Default |
| --- | --- | --- |
| vae | VAE | — |
| dimensions | from_image, fixed, preset | preset |
| width | 16..16384, step 16 | 1024 |
| height | 16..16384, step 16 | 1024 |
| preset_size | curated Krea geometries | 1024 x 1024 |
| geometry_policy | nearest_krea_aspect, preserve_aspect_krea_bounds | preserve_aspect_krea_bounds |
| content | empty, from_image | empty |
| content_fit | crop, contain, stretch | crop |
| resize_method | auto, nearest-exact, bilinear, bicubic, area, lanczos | auto |
| latent_semantic | boolean | false |
| latent_semantic_instruction | text | empty |
| latent_grounding_px | 0..4096, step 32 | 768 |
| batch_size | 1..64 | 1 |

Optional inputs:

- dimensions_image
- content_image
- attention_regions

Outputs:

- latent
- latent_info

from_image requires dimensions_image. latent_semantic requires content=from_image and content_image.

### Curated presets

| Geometry | Aspect | Approx. MP |
| --- | --- | --- |
| 1024 x 1024 | 1:1 | 1.05 |
| 1216 x 832 | ~3:2 | 1.01 |
| 832 x 1216 | ~2:3 | 1.01 |
| 1536 x 1024 | 3:2 | 1.57 |
| 1024 x 1536 | 2:3 | 1.57 |
| 1536 x 1152 | 4:3 | 1.77 |
| 1152 x 1536 | 3:4 | 1.77 |
| 2048 x 1536 | 4:3 | 3.15 |
| 1536 x 2048 | 3:4 | 3.15 |
| 2048 x 1152 | 16:9 | 2.36 |
| 1152 x 2048 | 9:16 | 2.36 |
| 2048 x 2048 | 1:1 | 4.19 |

The list is curated by CcC, not an official exhaustive Krea whitelist.

preset uses the selected geometry exactly. fixed and from_image resolve through geometry_policy. Final target dimensions are /16-aligned.

## Krea2 CcC Edit Prompt Creator

Optional multimodal prompt-generation node. It reuses the same Krea2/Qwen3-VL CLIP used by Edit and ComfyUI Generate Text.

Required inputs/controls:

| Control | Values | Default |
| --- | --- | --- |
| clip | CLIP | — |
| visual_references | KREA2_VISUAL_REFERENCE_CHAIN | — |
| mode | enhance, create_from_image, create_from_theme, custom | enhance |
| user_prompt | text | empty |
| system_prompt | text | enhance preset |
| thinking | boolean | false |
| max_tokens | 32..4096, step 32 | 512 |
| temperature | 0.01..2.0 | 0.25 |
| top_p | 0..1 | 0.90 |
| seed | 0..18446744073709551615 | 0 |

Optional input:

- reference_edit_image

Outputs:

- created_prompt
- visual_references
- creator_info
- thinking

The first three output slots remain compatible with the earlier node contract; thinking is the fourth output.

### Mode and system_prompt behavior

- enhance: treats the user's instructions as the authoritative guide and uses available visual context to make the requested final image clearer without changing intent.
- create_from_image: requires reference_edit_image and treats it as the authoritative visual blueprint for the desired result; the user's instructions decide what should be preserved, changed, emphasized, or adapted.
- create_from_theme: treats the user's theme or idea as the authoritative creative direction and uses the available visual context to build a coherent final-image prompt.
- custom: sends the editable system_prompt exactly as the Qwen system prompt.

For enhance, create_from_image, and create_from_theme, the frontend shows the effective preset system_prompt in the text field but keeps it greyed/read-only. Selecting custom enables the same field for editing. When entering custom for the first time, the current preset becomes the starting text; an existing custom value is preserved when switching away and back.

The backend does not trust the visible field for preset modes: it resolves the canonical preset by mode. In custom mode, it requires a non-empty system_prompt.

### Generalist preset behavior

The built-in presets intentionally avoid hard-coding people, clothing, poses, objects, or other content-specific checklists. Prompt Creator analyzes the available visual context and follows the user's request, producing a direct description of the desired final image rather than explaining the edit operation.

For create_from_image, the reference edit image is the primary visual blueprint. The preset does not predefine which visual details are important; relevance is determined by the image and the user's instructions.

### Thinking

thinking is passed directly to Qwen3-VL tokenization. When enabled, the Prompt Creator also uses a Qwen3-VL chat template whose assistant turn is prefilled with <think>, so generation starts inside the reasoning block instead of relying on the model to choose whether to enter thinking mode.

When Qwen closes the prefilled block with </think>:

- created_prompt receives only the final text after </think>;
- thinking receives the reasoning text;
- creator_info reports Thinking Output: present and also includes a Thinking section for debugging.

If generation ends inside the prefilled thinking block without producing </think> and a final prompt, the node raises a clear error suggesting a larger max_tokens value or disabling thinking. This prevents an unfinished reasoning trace from being mistaken for created_prompt.

### Final prompt cleanup

created_prompt is always post-processed before it reaches Edit. Visible references are named explicitly as `Image 1`, `Image 2`, and so on:

- accidental meta preambles such as "You are a professional image editor..." / "Your task is..." are removed when possible;
- an output consisting only of meta-instructions is rejected;
- "Krea Image N" is normalized to "Image N";
- the internal reference edit image must be converted into explicit scene instructions and must not be named in the final prompt.

The returned visual_references object is passthrough; the chain is not modified.

## Krea2 CcC Edit

Main edit orchestrator.

Required inputs/controls:

- model
- clip
- vae
- latent
- positive_prompt
- apply_krea2_edit_patch (default true)

Optional inputs:

- visual_references
- semantic_references

Outputs:

- patched_model
- positive
- negative
- latent
- edit_info

Execution:

~~~text
ordered references
  -> Qwen image preparation
  -> appearance pixel geometry
  -> VAE appearance latents
  -> CONDITIONING reference metadata
  -> Krea2 CcC Edit patch/runtime
  -> [text | references | target]
~~~

Negative text is fixed internally to an empty string. Appearance images remain available to the grounded negative pass with neutral boosts.

## Krea2 CcC Character Sheet

Deterministically composes generic portrait/body IMAGE inputs into one IMAGE.

Required controls:

| Control | Values | Default |
| --- | --- | --- |
| sheet_type | 9 layouts | 1 portrait |
| sheet_geometry | same curated Krea presets as Latent | 1024 x 1024 |
| resize_method | lanczos, bicubic, bilinear, area | lanczos |
| padding | 0..128 | 8 |
| outer_margin | 0..128 | 8 |
| background | white, gray, black | white |

Optional IMAGE inputs:

- portrait_1
- portrait_2
- portrait_3
- portrait_4
- body_1
- body_2

Layouts:

- 1 portrait
- 2 portraits
- 3 portraits
- 4 portraits
- 1 body
- 2 bodies
- 1 portrait + 2 bodies
- 3 portraits + 1 body
- 4 portraits + 1 body

Output: character_sheet.

Input numbers mean ordering only. Specific camera angles are not required.

Every source is uniformly resized to fit completely inside its slot with preserved aspect ratio. Character Sheet does not crop, cover, stretch, run Qwen, encode VAE, or create a latent.

## Krea2 CcC Paint Prepare

This node owns the complete pre-generation Paint preparation. The old standalone Paint Geometry public node has been removed.

Required controls:

| Control | Values | Default |
| --- | --- | --- |
| image | IMAGE | — |
| vae | VAE | — |
| geometry_mode | pad, crop | pad |
| padding_fill | edge, reflect, neutral, white | edge |
| horizontal_position | center, left, right | center |
| vertical_position | center, top, bottom | center |
| expand_left | 0..8192, step 16 | 0 |
| expand_top | 0..8192, step 16 | 0 |
| expand_right | 0..8192, step 16 | 0 |
| expand_bottom | 0..8192, step 16 | 0 |
| fill_holes | boolean | false |
| mask_grow | -256..256 | 0 |
| mask_blur_mode | standard, gaussian_sigma | gaussian_sigma |
| mask_blur_amount | 0..128 | 0 |
| mask_blur_direction | outside, inside, both | outside |

Optional input:

- mask

Outputs:

- paint_context
- latent
- prepared_image
- semantic_reference
- generated_mask
- keep_mask
- paint_geometry
- paint_prepare_info

Processing order:

~~~text
image + mask
 -> explicit outpaint expansion
 -> internal Krea pad/crop geometry
 -> optional fill enclosed holes
 -> signed mask grow/shrink
 -> directional feather
 -> generated_mask / keep_mask
 -> semantic-reference preparation
 -> VAE known-image latent
 -> token-aligned noise_mask
 -> sampling LATENT
~~~

The known source image is not resized during Paint geometry normalization.

Explicit outpaint expansion belongs to the requested/final canvas. Temporary Krea padding belongs only to the working canvas. paint_geometry carries enough information for Paint Restore to undo only the temporary geometry.

## Krea2 CcC Paint

Consumes KREA2_PAINT_CONTEXT from Paint Prepare.

Required inputs/controls:

| Control | Default |
| --- | --- |
| model | — |
| clip | — |
| paint_context | — |
| positive_prompt | empty background continuing naturally, no people, coherent perspective and lighting |
| apply_krea2_paint_patch | true |
| kv_cache | true |

Outputs:

- patched_model
- positive
- negative
- paint_info

Paint grounds Qwen with the semantic reference prepared by Paint Prepare, attaches the pre-encoded appearance reference latent, and installs the Paint runtime when enabled.

kv_cache is a runtime K/V optimization inside Paint. It is unrelated to the removed public Reference Cache nodes.

The LATENT sent to KSampler comes from Paint Prepare, not Paint.

## Krea2 CcC Paint Restore

Required inputs:

- image
- paint_geometry

Optional input:

- generated_mask

Output:

- image

pad mode removes only temporary Krea padding and preserves explicit user expansion.

crop mode composites the decoded generated crop back at its original coordinates. If generated_mask is provided, only the generated/feathered area replaces the preserved base canvas.

No inverse resize is performed.

## CcC Krea2 - LoRA Prompt Settings

Configures optional prompt augmentation for four LoRA slots.

Required controls:

- enabled
- for each slot 1..4:
  - lora_N_prompt_enabled
  - lora_N_prompt_position: append or prepend
  - lora_N_positive_prompt
  - lora_N_negative_prompt

Output:

- lora_prompt_settings

This node does not load a LoRA and does not alter prompts by itself.

## CcC Krea2 - LoRA Stack

Applies up to four model-only LoRAs.

Required controls:

- model
- enabled
- global_strength
- for each slot 1..4:
  - lora_N_enabled
  - lora_N_name
  - lora_N_strength

Optional inputs:

- prompt_augmentation
- lora_prompt_settings

Outputs:

- model
- prompt_augmentation

Effective per-slot LoRA strength is slot strength multiplied by global_strength.

If LoRA Prompt Settings is connected, prompt augmentation is accumulated only for active LoRA slots whose prompt setting is enabled.

Krea2 CcC Edit does not consume prompt_augmentation automatically.

## CcC Krea2 - Text to Image

Required inputs/controls:

| Control | Values | Default |
| --- | --- | --- |
| model | MODEL | — |
| clip | CLIP | — |
| prompt | text | — |
| aspect_ratio | 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3, custom | 1:1 |
| megapixels | 0.25..2.0 | 1.0 |
| batch_size | 1..64 | 1 |
| custom_aspect_width | 1..32 | 1 |
| custom_aspect_height | 1..32 | 1 |

Optional inputs:

- negative_prompt
- prompt_augmentation

Outputs:

- model
- positive
- negative
- latent

The node applies prompt augmentation, encodes positive and negative text independently, resolves width/height from aspect ratio + megapixels, rounds both axes to /16, and creates an empty SD3-compatible latent.

## Example Workflows

- workflows/01_scene_subject.json — Scene + Subject edit.
- workflows/02_anypaint_remove_people.json — Paint Prepare -> Paint -> Paint Restore.
- workflows/03_regional_attention.json — tagged regional references.
- workflows/04_character_sheet_identity.json — generic Character Sheet identity reference.
- workflows/05_edit_prompt_creator.json — create_from_image Prompt Creator with the optional Semantic Reference branch wired but disabled.
- workflows/06_text_to_image.json — direct T2I example using CcC Krea2 - Text to Image outputs as KSampler model/positive/negative/latent inputs.
