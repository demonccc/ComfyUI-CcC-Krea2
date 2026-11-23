# ComfyUI-Krea-CcC Nodes

This document describes the **15 registered nodes registered in ccc_krea2/nodes.py**. Internal helper classes and modules are intentionally excluded.

## Public Node Inventory

| Display name | Node id |
| --- | --- |
| Krea2 CcC Reference | CcCKrea2Reference |
| Visual Reference helper (internal) | CcCKrea2VisualReference |
| Vision Reference helper (internal) | CcCKrea2VisionReference |
| Krea2 CcC Size Resolver | CcCKrea2SizeResolver |
| Krea2 CcC Latent | CcCKrea2Latent |
| Krea2 CcC Edit Prompt Creator | CcCKrea2EditPromptCreator |
| Krea2 CcC Edit | CcCKrea2Edit |
| Krea2 CcC Layout Cell | CcCKrea2LayoutCell |
| Krea2 CcC Layout Composer | CcCKrea2LayoutComposer |
| Krea2 CcC Paint Prepare | CcCKrea2PaintPrepare |
| Krea2 CcC Paint | CcCKrea2Paint |
| Krea2 CcC Paint Restore | CcCKrea2PaintRestore |
| Krea2 CcC Lora Stack with Prompt | CcCKrea2LoRAStackWithPrompt |
| Krea2 CcC Lora Stack | CcCKrea2LoRAStack |
| Krea2 CcC Lora Stack Sliders | CcCKrea2LoRALoader |
| Krea2 CcC Text To Image | CcCKrea2TextToImage |

The Reference Cache implementation has been removed. There are no cache nodes and no public Krea2 CcC Paint Geometry node.

## Unified Krea2 CcC Reference

Node ID: `CcCKrea2Reference`. This is the recommended reference node for new workflows.

The required inputs start with `image`, then `vae_rope_enable` and the `vae_rope_*` controls, followed by `vision_enable`, `vision_weight`, resize, mode, and optional instruction. Optional inputs are `previous_references` and `mask_attention`. Disabled groups are ignored. At least one path must be enabled.

The Visual path produces an appearance/VAE entry with no automatic Qwen participation. The Vision path produces Visual or Semantic Qwen conditioning without any VAE appearance entry. Both paths can operate on the same source image, appended in Visual-then-Vision order to a shared `KREA2_REFERENCE_CHAIN`. This is compatible with Edit's existing separate spatial LoRA reference mechanism and Qwen conditioning interfaces, but Vision fidelity requires live sampling tests.

Standalone Visual and Vision classes remain internal helpers and are not independently registered. Saved workflows must use the unified Reference and valid Vision modes.

## Krea2 CcC Visual Reference

Declares one ordered appearance reference.

Required controls:

| Control | Values | Default | Meaning |
| --- | --- | --- | --- |
| image | IMAGE | — | Source reference. |
| boost | -5.0..+5.0, step 0.1 | 0.0 | Signed boost: multiplier `10^(boost/5)`; -5 blocks, 0 neutral (1×), +5 amplifies to 10×. |
| reference_fit | native, crop, contain, fit | native | Appearance/VAE pixel fit. |
| placement_grid | inside, outside | inside | RoPE placement grid; crop forces inside. Outside + center + center is invalid. |
| grid_horizontal_position | center, left, right | center | Horizontal RoPE/grid position. |
| grid_vertical_position | center, up, down | center | Vertical RoPE/grid position. |
| rope_mode | spatial, collapsed | spatial | Keep token positions or collapse every reference token onto one RoPE point. |
| reference_stack | overlay, stacked | overlay | Position refs over the same origin or accumulate in chain order (may overflow target). |
| rope_geometry | latent, source | latent | Dimensions used for placement; native forces source. |
| semantic | boolean | true | Also show this image to Qwen. |
| semantic_resize | boolean | true | Downscale Qwen image only when above the cap. |
| semantic_grounding_px | 32..4096, step 32 | 768 | Qwen longest-edge cap. |
| prompt_annotation | text | empty | Role-only Image N annotation. |

Reference exposes `mask_attention` for ordered regional Boost masks. Every masked region replaces the global Boost in its selected pixels; later overlapping regions take priority. The mask follows the reference image through VAE geometry. `reference_boost_mask` and the earlier target-attention-scope controls have been removed.

Output: `references` (`KREA2_REFERENCE_CHAIN`).

### Appearance fit

- crop: the target acts as an inside crop window over the source.
- fit: scale proportionally to cover the target and center-crop the excess; output is /16-aligned.
- contain: uniformly fit inside the target, preserve aspect, then crop down only the minimum pixels needed for /16.
- native: keep 1:1 pixel scale and center-crop width/height down to /16.

Qwen preparation is independent from appearance/VAE fit. Image resizing uses bicubic with antialiasing; native and crop do not scale.

prompt_annotation is identification only, for example: This is the subject image. Edit instructions belong in Edit's positive_prompt.

The positive branch uses the configured boost. The grounded negative branch uses the same appearance-image vision blocks with no role annotation text, fixed empty negative text, and neutral boost 0.0.

## Krea2 CcC Vision Reference

Adds a Qwen-only reference without an appearance VAE latent. There are **only
two modes**:

| Control | Values | Default | Meaning |
| --- | --- | --- | --- |
| image | IMAGE | — | Image provided to Qwen for multimodal encoding. |
| mode | visual, semantic | visual | Visual keeps image rows; Semantic drops image rows after multimodal encoding. |
| instruction | text | empty | Free user instruction for Qwen. Omitted entirely when empty. |
| weight | 0.0..2.0 | 1.0 | Multiplier for retained Visual image rows only. |
| previous_references | KREA2_REFERENCE_CHAIN | optional | Previous mixed reference chain. |

Outputs: `references` and `vision_info`.

Both modes use the **same image preparation**, Qwen encoder, and user-provided
instruction, without hidden prompts or preset-dependent guidance generation.
Neither mode invokes `clip.generate()` or appends an automatically generated
description to Edit/T2I's positive prompt. Semantic removes its image rows from
the final conditioning tensor; the Qwen-encoded text rows remain. The general
reference VAE/RoPE path is independent of this selection.

Old mode values including `reference`, `style`, and `context` must be
migrated explicitly. Outdated saved widgets are not reinterpreted.

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
| latent_semantic | boolean | false |
| latent_semantic_instruction | text | empty |
| latent_grounding_px | 0..4096, step 32 | 768 |
| batch_size | 1..64 | 1 |

Optional inputs:

- dimensions_image
- content_image

Outputs:

- latent
- latent_info
- target_image (the exact post-fit target canvas, or neutral gray when empty, for aligned mask painting)

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
| references | KREA2_REFERENCE_CHAIN | optional |
| mode | enhance_t2i/i2i, create_from_image_t2i/i2i, create_from_theme_t2i/i2i, none | enhance_t2i |
| instructions | text | empty |
| thinking | boolean | false |
| max_tokens | 32..32768, step 32 | 512 |
| temperature | 0.01..2.0 | 0.25 |
| top_p | 0..1 | 0.90 |
| seed | 0..18446744073709551615 | 0 |

Optional input:

- reference_edit_image

Outputs:

- created_prompt
- references
- creator_info
- thinking

The first three output slots remain compatible with the earlier node contract; thinking is the fourth output.

### Presets and instructions

The node exposes six presets from three shared templates. Each family generates either a text-to-image (T2I) or image-to-image (I2I) prompt, replacing only the `{kind}` placeholder:

- `enhance_t2i` / `enhance_i2i`: enrich the instructions while preserving their intent.
- `create_from_image_t2i` / `create_from_image_i2i`: analyze optional reference images and use the instructions to decide which visual information is relevant and how to use it.
- `create_from_theme_t2i` / `create_from_theme_i2i`: invent a scene inspired by the instructions.
- `none`: no preset instruction; the user writes the complete task in `instructions`.

There is no editable `system_prompt` textbox and no frontend preset preview or graying. The prompt sent to Qwen is assembled as preset instruction (if selected), `INSTRUCTIONS`, optional neutral `IMAGE MAPPING` (Image 1, Image 2, etc.), and shared `KREA2 PROMPT GUIDELINES`. The guidelines also apply to `none`. The full effective assembled prompt appears on the `creator_info` output.

Both Image and Theme families accept references optionally, and do not assign image roles. Enhance remains text-only for prompt analysis; reference chains remain available for downstream Edit.

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

Prompt Creator inspects only Visual Reference entries, and its `references` output passes the complete mixed chain through unchanged.

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

- references (KREA2_REFERENCE_CHAIN)

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

## Krea2 CcC Image Layout

`CcCKrea2LayoutCell` accepts an IMAGE and an optional `previous_layout` chain. Set `placement` to `cell`, `full_row`, or `full_column`. Rows/columns start at 1. Crop-only `resize_percent` (integer, 1–1000%, default 100%) proportionally resizes the source before clipping; bicubic antialiasing is used. Fit modes: `contain` scales the complete image proportionally into its slot; `stretch` fills the slot non-proportionally; `crop` clips at native pixel scale; `fit` proportionally covers then clips. All resize operations use bicubic antialiasing. `horizontal_position`, `vertical_position`, `offset_x` and `offset_y` apply only to `crop` and `fit`. Offsets are signed pixel additions to the selected alignment.

`CcCKrea2LayoutComposer` consumes `KREA2_LAYOUT_CHAIN`, Krea `preset_size`, `padding`, `gap`, and `background_color`. It produces `image` and `layout_info`. Output dimensions are resolved before row/column sizes; whole rows/columns span the occupied grid. Overlaps, gaps, invalid positions or impossible padding raise errors. The sample `workflows/07_scene_composition.json` uses Layout Cells and Composer to prepare a multi-image scene blueprint.


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

## Krea2 CcC Lora Stack

Applies up to **10 MODEL-only LoRAs**, in slot order. Each slot exposes an enabled switch, LoRA name, and strength (-10 to 10). It returns MODEL only; CLIP and text conditioning are unaffected.

## Krea2 CcC Lora Stack with Prompt

Applies up to **5 MODEL-only LoRAs** and builds optional **positive prompt augmentation**. Each slot has its own enabled switch, LoRA name, strength (-10 to 10), prompt position (append/prepend), and prompt text.

Optional input: `prompt_augmentation`, allowing prompt triggers to accumulate across stacks.

Outputs: `model`, `prompt_augmentation`.

Edit, Paint, and T2I accept `prompt_augmentation` on their own optional inputs. This does not alter Edit's fixed-empty grounded negative prompt.

## Krea2 CcC Lora Stack Sliders

Applies up to **10 MODEL-only LoRAs**, each with its own enabled toggle, name, and strength (-10 to 10, slider display). Returns MODEL only.

## Krea2 CcC Text To Image

Required inputs/controls:

| Control | Values | Default |
| --- | --- | --- |
| model | MODEL | — |
| clip | CLIP | — |
| prompt | text | — |
| dimensions | preset, fixed | preset |
| preset_size | shared curated Krea geometries | 1024 x 1024 |
| width | 16..16384, step 16 | 1024 |
| height | 16..16384, step 16 | 1024 |
| geometry_policy | nearest_krea_aspect, preserve_aspect_krea_bounds | preserve_aspect_krea_bounds |
| batch_size | 1..64 | 1 |

Optional inputs:

- negative_prompt
- prompt_augmentation
- references

Outputs:

- model
- positive
- negative
- latent
- t2i_info

The node applies prompt augmentation and creates an empty SD3-compatible latent from the same shared Krea geometry library used by Latent. Preset mode uses the shared curated Krea sizes directly; fixed mode resolves through the shared Krea geometry policy. Without Vision Reference it uses normal text-only CLIP encoding. With Vision Reference, each mode is analyzed by Qwen multimodal text generation and the resulting guidance is appended to the user's requested result before final text-only Krea conditioning. Direct Qwen vision rows are not used in T2I. The explicit negative prompt remains text-only.

## Example Workflows

The seven workflows are in `workflows/` and are ordered from simpler generation to more complex reference-guided editing:

1. `01_creative_generation.json` — Vision-only inspiration, Prompt Creator and T2I.
2. `02_outpainting_reframing.json` — Paint extension and restoration.
3. `03_local_image_editing.json` — Layout + SAM + inverted mask + Paint for studio background replacement.
4. `04_identity_scene_transfer.json` — Two references, mask Boost and Prompt Creator for scene/identity transfer.
5. `05_character_consistency.json` — Ordered body and facial masks for consistent character identity.
6. `06_outfit_transfer.json` — Three references: scene, subject and outfit.
7. `07_scene_composition.json` — Layout-based scene blueprint and reference-guided composition.

These files replace the previous isolated examples. Source photographs must be supplied in ComfyUI, and GPU image quality has not been verified.


### Edit RoPE annotated preview

The `rope_preview` IMAGE batch contains four spatial views: layout, overlay_boxes,
overlay_visuals and regional_boosts, followed by one RoPE-positioned Boost view
per appearance reference. All frames share one coordinate system. Collapsed
references appear as points and Qwen-only Vision References are excluded.
Latent exports the post-fit target image as `target_image`.


### Reference image transport (updated)

Every Reference always exposes its original source image to downstream nodes, even with `vae_rope_enable=false` and `vision_enable=false`. The image-only representation contributes no Edit conditioning. `vae_rope_enable` replaces `visual_enable`; `vae_rope_*` replaces the corresponding `visual_*` widget names. Vision independently controls Qwen conditioning and may resize its own analysis image without discarding the source.

Prompt Creator accepts an optional `references` input of type `KREA2_REFERENCE_CHAIN`. It analyzes source images in chain order (paired VAE/RoPE and Vision inputs are analyzed once), returns the generated prompt, and forwards the same reference chain to Edit. See `workflows/04_identity_scene_transfer.json` and `workflows/06_outfit_transfer.json` for complete examples.
