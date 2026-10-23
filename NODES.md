# ComfyUI-Krea-CcC Nodes

This document describes the **15 registered nodes registered in ccc_krea2/nodes.py**. Internal helper classes and modules are intentionally excluded.

## Public Node Inventory

| Display name | Node id |
| --- | --- |
| Krea2 CcC Reference | CcCKrea2Reference |
| Krea2 CcC Visual Reference | CcCKrea2VisualReference |
| Krea2 CcC Vision Reference | CcCKrea2VisionReference |
| Krea2 CcC Size Resolver | CcCKrea2SizeResolver |
| Krea2 CcC Latent | CcCKrea2Latent |
| Krea2 CcC Edit Prompt Creator | CcCKrea2EditPromptCreator |
| Krea2 CcC Edit | CcCKrea2Edit |
| Krea2 CcC Character Sheet | CcCKrea2CharacterSheet |
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

The required inputs start with `image`, then `visual_enable` and the complete `visual_*` group, then `vision_enable` and the complete `vision_*` group. Optional inputs are `previous_references`, `reference_boost_mask`, and `target_attention_mask`. Disabled groups are greyed out and ignored. At least one path must be enabled.

The Visual path produces an appearance/VAE entry with no automatic Qwen participation. The Vision path produces Qwen-specific Semantic, Visual, Style, or Context guidance without any VAE appearance entry. Both paths can operate on the same source image, appended in Visual-then-Vision order to a shared `KREA2_REFERENCE_CHAIN`. This is compatible with Edit's existing separate spatial LoRA reference mechanism and Qwen conditioning interfaces, but Vision fidelity requires live sampling tests.

The previous Visual Reference and Vision Reference node IDs are retained for existing workflows; do not use them for new workflows.

## Krea2 CcC Visual Reference

Declares one ordered appearance reference.

Required controls:

| Control | Values | Default | Meaning |
| --- | --- | --- | --- |
| image | IMAGE | — | Source reference. |
| boost | 0..10 | 1.0 | Positive target-to-reference attention boost. |
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
| attention_scope | global, boost in mask, only in mask | global | Scope on the prepared target; no tags. |

Optional inputs: `previous_references`, `reference_boost_mask` (MASK on the original reference, transformed alongside the image), and `target_attention_mask` (MASK drawn on the final target canvas, required for regional attention). Each Visual Reference owns both masks; no separate tags or mask ordering is needed.

`global` applies boost to selected reference tokens from the entire target; `boost in mask` limits the boost to white pixels in the target mask; `only in mask` additionally blocks direct target-to-reference attention outside the target mask. `reference_boost_mask` limits boost to selected reference tokens and combines with the target mask when both are connected.

Masks must be aligned to their own images: `reference_boost_mask` must match the original Visual Reference image pixel dimensions; `target_attention_mask` must match the final `Latent.target_image` dimensions. ComfyUI MASK inputs accept one mask shared by the batch or one per item. Mismatched sizes raise an error, not an implicit resize. The source mask follows the reference's crop/resize **before** VAE; at runtime both masks are projected to the DiT token grids using area reduction and a >0.5 threshold. `only in mask` restricts *direct attention*, not image editing/pixel preservation (use Paint for inpainting).

Output: `references` (`KREA2_REFERENCE_CHAIN`).

### Appearance fit

- crop: the target acts as an inside crop window over the source.
- fit: scale proportionally to cover the target and center-crop the excess; output is /16-aligned.
- contain: uniformly fit inside the target, preserve aspect, then crop down only the minimum pixels needed for /16.
- native: keep 1:1 pixel scale and center-crop width/height down to /16.

Qwen preparation is independent from appearance/VAE fit. Image resizing uses bicubic with antialiasing; native and crop do not scale.

prompt_annotation is identification only, for example: This is the subject image. Edit instructions belong in Edit's positive_prompt.

The positive branch uses the configured boost. The grounded negative branch uses the same appearance-image vision blocks with no role annotation text, fixed empty negative text, and neutral boost 1.0.

## Krea2 CcC Vision Reference

Adds one Qwen/Vision-only reference. It never creates an appearance VAE latent.

| Control | Values | Default | Meaning |
| --- | --- | --- | --- |
| image | IMAGE | — | Source image shown to Qwen Vision. |
| mode | semantic, visual, style, context | semantic | Chooses what the reference contributes. |
| instruction | text | empty | Optional extra instruction for this reference. |
| style_processing | full, 2x2, 4x4 | full | Used only by Style; all other modes force full. |
| guidance_seed_mode | fixed, incremental, random | fixed | Seed behavior for Qwen text guidance generation. |
| guidance_seed | integer | 0 | Base/fixed Qwen guidance seed; editable only in fixed mode. |
| guidance_detail | compact, balanced, detailed | balanced | Extraction-depth policy plus max-length budget: compact keeps only highest-impact facts (256), balanced keeps major + secondary facts (512), detailed is exhaustive within the selected mode (1024). |
| vision_guidance | conservative, balanced, flexible | balanced | Qwen sampling preset. Conservative = temp 0.05 / top_p 0.70 / top_k 32 / min_p 0.10; balanced = 0.25 / 0.90 / 64 / 0.05; flexible = 0.70 / 0.99 / 128 / 0.01. |
| previous_references | KREA2_REFERENCE_CHAIN | optional | Previous mixed reference chain. |

Outputs:

- `references` (`KREA2_REFERENCE_CHAIN`)
- `vision_info` (`STRING`) with consumer-neutral declaration metadata including mode, processing, Qwen guidance preset/seed, source resolution, and chain size.

Vision Reference defines information semantics rather than a transport algorithm:

- **Semantic**: content/meaning, pose, action, objects, composition, and spatial relationships.
- **Visual**: near-copy/full visual blueprint. Balanced/Detailed strongly anchor pose, gaze, framing, composition, and appearance; Flexible may adapt pose/composition while retaining visual identity. Subject count is always locked and must not be expanded into clones, variants, panels, or repeated subjects.
- **Style**: intentionally reuses the exact legacy Visual guidance preset that previously produced the desired style-transfer behavior, and its final conditioning explicitly treats that guidance as the dominant visual treatment. Supports full, 2x2, or 4x4 analysis.
- **Context**: mandatory scene staging including setting, viewpoint, exact pose and limb placement, head orientation, gaze target, gesture, action, interaction, crop/framing, composition, object placement, depth, and spatial relationships while excluding identity, clothing, and visual style.

There is no public `grounding_px` or `fidelity` control. Qwen text-generation settings travel with the Vision Reference and are consumed by T2I's text-guidance adapter for Semantic, Visual, Style, Context, and Native. `guidance_seed` is editable only when `guidance_seed_mode=fixed`; `style_processing` is the only mode-specific control and is enabled only for Style.

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
| mode | enhance, create_from_image, create_from_theme, custom | enhance |
| user_prompt | text | empty |
| system_prompt | text | enhance preset |
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

### Mode and system_prompt behavior

- enhance: treats the user's instructions as the authoritative guide and uses available visual context to make the requested final image clearer without changing intent.
- create_from_image: requires reference_edit_image and treats it as the authoritative visual blueprint for the desired result; the user's instructions decide what should be preserved, changed, emphasized, or adapted. Visual References remain part of the multimodal analysis and keep their Image N mapping; reference_edit_image is an additional internal blueprint image, not a replacement for those references.
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

## Krea2 CcC Character Sheet

Deterministically composes generic portrait/body IMAGE inputs into one IMAGE.

Required controls:

| Control | Values | Default |
| --- | --- | --- |
| sheet_type | 9 layouts | 1 portrait |
| sheet_geometry | same curated Krea presets as Latent | 1024 x 1024 |
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

Examples are historical fixtures and have not been migrated in this cleanup; a workflow referring to the removed Attention Region node will not load as a current graph.

- workflows/01_scene_subject.json — Scene + Subject edit.
- workflows/02_anypaint_remove_people.json — Paint Prepare -> Paint -> Paint Restore.
- workflows/03_regional_attention.json — **legacy/incompatible**: still contains the removed `CcCKrea2AttentionRegion` nodes. Intentionally unchanged; migrate its regions to per-Visual-Reference target masks before using.
- workflows/04_character_sheet_identity.json — generic Character Sheet identity reference.
- workflows/05_edit_prompt_creator.json — create_from_image Prompt Creator with the optional Vision Reference branch wired but disabled.
- workflows/06_text_to_image.json — T2I example using Vision Reference analysis and text-only final conditioning.

- workflows/12_edit_vision_semantic.json — Vision Reference / Semantic.
- workflows/13_edit_vision_visual.json — Vision Reference / Visual.
- workflows/14_edit_vision_style.json — Vision Reference / Style.
- workflows/18_edit_vision_context.json — Vision Reference / Context.

### Edit RoPE annotated preview

`rope_preview` returns a three-image `IMAGE` batch in fixed order: **layout**, **overlay_boxes**, **overlay_visuals**. All three share the same spatial extent and layout planner. `layout` draws the target and colored reference boxes. `overlay_boxes` draws the target's VAE-prepared image when the input Latent was created `from_image` (otherwise a neutral canvas), with translucent reference boxes. `overlay_visuals` draws the same opaque target with VAE-prepared Visual Reference images overlaid transparently. `collapsed` references are points in all three views, not image thumbnails. Qwen-only Vision References are excluded. The target image is preserved as Latent metadata and exported as `target_image` for mask painting and RoPE preview; it is the original post-fit pixel canvas, not a VAE decode of latent samples.


### Reference image transport (updated)

Every Reference always exposes its original source image to downstream nodes, even with `vae_rope_enable=false` and `vision_enable=false`. The image-only representation contributes no Edit conditioning. `vae_rope_enable` replaces `visual_enable`; `vae_rope_*` replaces the corresponding `visual_*` widget names. Vision independently controls Qwen conditioning and may resize its own analysis image without discarding the source.

Prompt Creator's optional `reference_edit_image` input now accepts `KREA2_REFERENCE_CHAIN` (not `IMAGE`). For analysis, edit images are first and the optional `references` chain follows. Paired VAE/RoPE+Vision entries are analyzed once. The output `references` combines the edit chain first and the other chain next, keeping the processing switches unchanged. The source image is available to Prompt Creator even when both conditioning paths are disabled. Existing saved workflows need socket and widget migration; workflow JSONs were not changed.
