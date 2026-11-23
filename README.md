# ComfyUI-Krea-CcC

ComfyUI-Krea-CcC provides ComfyUI nodes for Krea 2 generation, text-to-image, reference-guided editing, prompt creation, Vision guidance, grid-based image layouts, LoRA workflows, and AnyPaint-style inpaint/outpaint.

The public surface currently contains **15 registered nodes** (including the unified Reference and the two legacy reference nodes). The former Reference Cache feature has been removed. Paint geometry remains internal to Paint Prepare/Restore.

## Unified Reference (new workflows)

**Krea2 CcC Reference** combines the former Visual and Vision reference controls in one node and one source `image`. The reference role label (`vae_rope_prompt_annotation`) belongs to the VAE/RoPE section, independently of the Vision mode; the shipped example workflows store this field explicitly; ComfyUI does not automatically migrate or repair saved widget values. The controls are ordered with `vae_rope_enable` and the VAE/RoPE geometry and annotation first, followed by `vision_enable`, `vision_weight`, then Vision resize, mode and optional instruction. Both groups are enabled by default. Disabled groups are greyed out in ComfyUI and are not processed. Vision optionally downsizes the source with bicubic antialiasing (`vision_resize=true`, `vision_max_edge=768`) independently of Visual, without cropping or upscaling. The combined node omits the legacy guidance detail/sampling presets, which have no equivalent Edit effect. `vision_weight` (0.0–2.0, default 1.0) multiplies only this reference’s Qwen image-token embeddings after Vision-mode processing; 0.0 zeros those rows, 1.0 is neutral, and 2.0 doubles them. It does not change the prompt, VAE/RoPE Boost, or target geometry. The image remains present during Qwen encoding, so 0.0 does not guarantee removal of information already mixed into other contextual tokens.

- **Visual only**: VAE appearance reference, Fit, RoPE, Boost, reference mask and target attention mask. It does **not** send the image to Qwen.
- **Vision only**: Visual / Semantic guidance without any appearance/VAE reference.
- **Both**: one source image contributes one Visual entry and one Vision entry to the same ordered reference chain; Qwen is not fed a duplicate semantic copy from the Visual path.
- **Neither**: invalid configuration. At least one path must be enabled.

Edit exposes an optional `isolate_references` switch (off by default): when enabled, each VAE/RoPE reference can attend to text and itself but not to another reference or the target. Target-to-reference attention remains controlled by existing masks and Boost; this does not constitute full subject/region ownership and requires compatible attention backends.\n\nThe Krea2 Edit LoRA remains attached to the model as before. Its image-reference attention is provided by the enabled Visual path; the Vision path supplies separate Qwen conditioning and is **not** a substitute for the LoRA's spatial reference mechanism. Fidelity of the Vision modes with the LoRA still needs image-generation testing.

The standalone Visual and Vision classes are internal implementation helpers, not independently registered public nodes. Use **Krea2 CcC Reference** for new workflows. Saved workflows with removed Vision mode values must be explicitly migrated; workflow JSON files have not been modified.

## Public Nodes

| Area | Node | Purpose |
| --- | --- | --- |
| Edit | **Krea2 CcC Visual Reference** | Ordered appearance reference with VAE fit, RoPE placement, Qwen grounding, boost and optional source/target attention masks. |
| Edit | **Krea2 CcC Vision Reference** | Qwen/Vision-only Visual or Semantic reference; no appearance latent. |
| Edit | **Krea2 CcC Size Resolver** | Resolves width/height from one image's long edge and another image's aspect ratio. |
| Edit | **Krea2 CcC Latent** | Owns target dimensions, Krea geometry policy, optional target content, semantic target context, and an aligned `target_image` output for masks. |
| Edit | **Krea2 CcC Edit Prompt Creator** | Optional multimodal prompt creator using the same Krea2/Qwen3-VL CLIP as Edit. |
| Edit | **Krea2 CcC Edit** | Final edit orchestrator and Krea2 CcC edit runtime. |
| Utility | **Krea2 CcC Layout Cell / Layout Composer** | Compose a chain of images into a Krea-sized grid, including full rows or columns. |
| Paint | **Krea2 CcC Paint Prepare** | Owns Paint geometry, outpaint expansion, mask processing, semantic reference, VAE preparation, and sampling latent. |
| Paint | **Krea2 CcC Paint** | Builds AnyPaint/Krea Paint conditioning and installs the Paint runtime. |
| Paint | **Krea2 CcC Paint Restore** | Restores decoded Paint output to the requested canvas. |
| LoRA | **Krea2 CcC Lora Stack with Prompt** | Applies MODEL-only LoRAs with optional prompt augmentation. |
| LoRA | **Krea2 CcC Lora Stack** | Applies MODEL-only LoRAs. |
| LoRA | **Krea2 CcC Lora Stack Sliders** | Slider-based MODEL-only LoRA loader. |
| T2I | **Krea2 CcC Text To Image** | Native Krea2 text-to-image conditioning and empty latent helper. |

## Edit Architecture

The direct edit path remains simple:

~~~text
Visual Reference --> Vision Reference --> Visual Reference
       (each adds to the same KREA2_REFERENCE_CHAIN)
                               |
                               v
                         references --> Krea2 CcC Edit
Size Resolver -------> Latent --------------^
~~~

The unified Reference accepts chained `mask_attention` regional Boosts, with **Last Mask Wins** priority. Masks must match the reference image geometry. These are reference-side attention controls, not Paint/Inpaint pixel replacement.

The reference nodes can be chained in any order using `previous_references`; they expose one shared `references` output. Edit accepts a single `references` input. Appearance references go through VAE and (optionally) Qwen; Vision references are Qwen-only.

**Edit Prompt Creator** is optional:

~~~text
references --> Edit Prompt Creator --> references --> Edit
                     |                    ^
CLIP ----------------+                    |
instructions -------+--> created_prompt -+--> Edit.positive_prompt
reference_edit_image-+
~~~

Prompt Creator examines **only Visual Reference entries** from the mixed reference chain and passes the entire chain through unchanged. Its optional reference_edit_image is used internally for prompt analysis, not as an appearance reference.

**Layout Cell / Layout Composer** build image sheets from a chain of positioned source images; the composited IMAGE can feed a Reference or Edit workflow.

## Visual Reference

Each Visual Reference is an ordered appearance reference with two independent representations.

The VAE/appearance path supports **native, crop, contain, and fit**:

- crop: use the target as an inside crop window over the source.
- fit: scale proportionally to cover the target, center-crop the excess, and align to /16.
- contain: uniformly fit the source inside the target, preserve aspect ratio, then crop down only the minimum pixels required for /16 alignment.
- native: keep 1:1 source pixel scale and center-crop width/height down to /16.

RoPE placement is independent from pixel fit. Visual Reference supports `rope_mode=spatial|collapsed`, `reference_stack=overlay|stacked`, and `rope_geometry=latent|source`. `native` forces `source`. `outside + center + center` is rejected. Stacks may extend beyond the target. Edit exposes `rope_preview` (IMAGE) showing the target and Visual References only; collapsed references render as points.

The Qwen path is controlled by semantic. When enabled, the same image can be downscaled only when it exceeds semantic_grounding_px. prompt_annotation is identification-only and becomes Image N: <annotation>; edit instructions belong in Edit's positive prompt.

Image scaling uses bicubic interpolation with antialiasing. Native and crop avoid image scaling.

Positive conditioning uses each reference's configured boost. The grounded negative branch uses the same appearance images, no negative text, no role annotations, and neutral reference boost 0.0.

**VAE/RoPE Boost semantics:** `vae_rope_boost` in Reference and `boost` in Mask use **-5.0 to +5.0, step 0.1, default 0.0**. Zero is neutral; negative values attenuate and positive values amplify direct target-to-reference attention. Values use a symmetric exponential mapping `10^(boost/5)` (attention bias `boost × ln(10)/5`), except `-5`, which is a special hard block (`-inf`). Thus `0` is 1×, `+5` is 10×, and opposite finite boosts have reciprocal multipliers. This is not a guaranteed identity preservation or exact-copy strength. Regional Mask Boost values override the global Boost inside selected pixels, with **Last Mask Wins** priority. The mechanism affects VAE/RoPE attention, not Qwen Vision. Only positive conditioning receives regional Boost maps; the grounded negative branch is neutral (0.0). Workflows saved under the previous 0–10 scale need their Boost values reviewed; the new values are not automatically migrated.

The unified Reference uses `mask_attention` regional Boosts to override its global Boost. There is no separate target-attention mask or attention-scope selector. Latent still exports `target_image` for workflows that need a target-image painting canvas.

Target Latent retains its independent image-content Vision path, but its public toggle is now `latent_vision_enable` (formerly `latent_semantic`) and its mode selector uses the same two presets as Reference: `Visual / Semantic`. This path applies to the target content image when `content=from_image`; it does not alter VAE geometry or replace a dedicated Scene/Subject Reference. Existing positional workflows retain the toggle position, and legacy Python calls using `latent_semantic` are accepted. Invalid saved mode values are reported instead of silently interpreted as a different preset.\n\n## Vision Reference

Vision has **only two modes**. They differ solely in whether image-token rows
remain in the final Qwen conditioning:

- **Visual** (default): encode the reference image with Qwen and retain its visual rows.
- **Semantic**: encode the **same** image with Qwen, then remove only that reference's
  visual rows from the final conditioning. Text rows that Qwen encoded with the
  image in context are retained.

Neither mode runs Qwen text generation. Neither mode has a built-in descriptive
prompt, an interpretation policy, a style/context extractor, or special sampling
settings. They never append a generated image description to the user prompt.

`instruction` is an optional, completely free user instruction. It is included
as `Image N guidance: ...` only when nonempty, in **both** modes. An empty
instruction emits no such text. The Edit positive prompt is otherwise unchanged.

`vision_weight` scales retained visual rows (Visual only). It has no direct
rows to scale in Semantic. Source images can be resized before encoding using
`vision_resize` / `vision_max_edge`. VAE/RoPE positioning and Boost are separate.

Old saved mode values (`reference`, `style`, `context`, internal names) are
**not silently mapped** to these modes. Migrate outdated workflows explicitly.

## Latent and Target Geometry

Krea pixel geometry is shared by the suite through one library of curated presets and resolution policies. Krea2 CcC Latent and Text to Image both consume that same source of truth.

Dimension modes:

- from_image
- fixed
- preset

For from_image and fixed, geometry policies are:

- nearest_krea_aspect
- preserve_aspect_krea_bounds

preset uses the selected curated geometry exactly.

Content modes:

- empty
- from_image

When content is from_image, fit modes are crop, contain, and stretch.

Current curated Krea target geometries:

| Size | Aspect ratio | Approx. pixels |
| --- | --- | --- |
| 1024 x 1024 | 1:1 | ~1.05 MP |
| 1216 x 832 | ~3:2 | ~1.01 MP |
| 832 x 1216 | ~2:3 | ~1.01 MP |
| 1536 x 1024 | 3:2 | ~1.57 MP |
| 1024 x 1536 | 2:3 | ~1.57 MP |
| 1536 x 1152 | 4:3 | ~1.77 MP |
| 1152 x 1536 | 3:4 | ~1.77 MP |
| 2048 x 1536 | 4:3 | ~3.15 MP |
| 1536 x 2048 | 3:4 | ~3.15 MP |
| 2048 x 1152 | 16:9 | ~2.36 MP |
| 1152 x 2048 | 9:16 | ~2.36 MP |
| 2048 x 2048 | 1:1 | ~4.19 MP |

These are curated CcC presets, not an official exhaustive Krea whitelist. Final target dimensions are /16-aligned.

Official references used for the target policy:

- Krea 2 official repository: https://github.com/krea-ai/krea-2
- Krea 2 sampling implementation: https://github.com/krea-ai/krea-2/blob/main/sampling.py
- Krea 2 Hugging Face Space: https://huggingface.co/spaces/krea/Krea-2/blob/main/app.py

## Edit Prompt Creator

The Creator reuses the same multimodal Krea2/Qwen3-VL CLIP as ComfyUI Generate Text and Edit.

The node uses one editable `instructions` field and seven selector choices: six presets in three families plus `none`:

- `enhance_t2i` / `enhance_i2i`: enrich the instruction without changing intent.
- `create_from_image_t2i` / `create_from_image_i2i`: analyze available images, using the instructions to decide which details matter and how to apply them.
- `create_from_theme_t2i` / `create_from_theme_i2i`: invent a scene inspired by the instructions.
- `none`: user supplies the entire task through `instructions`, without a preset.

Each pair uses one shared template with a `{kind}` placeholder for `text-to-image` or `image-to-image`. Every mode appends the common Krea2 guidelines; the effective full prompt sent to Qwen is available in `creator_info`. The old visible `system_prompt` widget and preset graying behavior have been removed.

`thinking=true` uses **one Qwen3-VL generation with two explicit sections** rather than forcing a native `<think>` prefix. The system and user instructions require `ANALYSIS:` followed by `FINAL PROMPT:`. In `ANALYSIS`, Qwen must separately inspect each image, decide the roles specified by the user's instructions (identity, clothing, pose, environment, interactions), and describe conflicts. That section is returned on the `thinking` output; only the text after `FINAL PROMPT:` is returned on `created_prompt`. This is *requested explicit visual analysis*, not proof that the model has a native internal thinking mode. A missing analysis section, missing final section, or empty final prompt leaves `created_prompt` empty, preserves available text on `thinking`, and puts `ERROR:` at the beginning of `creator_info`. No fallback or second generation is made.

All preset modes enforce final edit text with no system/meta preamble. Downstream visible references are named Image 1, Image 2, etc.; accidental "Krea Image N" wording is normalized to "Image N".

The Visual Reference chain is returned unchanged.

## Image Layout

`Krea2 CcC Layout Cell` appends one IMAGE to a `KREA2_LAYOUT_CHAIN` with row, column, `cell/full_row/full_column`, fit (`contain/stretch/crop/fit`), crop-only positive `resize_percent` (default 100%), alignment and signed pixel offsets. `Krea2 CcC Layout Composer` resolves the grid using a Krea output preset, padding, gap, and background color. Overlapping or missing cells raise an error. `workflows/07_scene_composition.json` demonstrates using Layout Cells and Layout Composer to prepare a multi-image scene blueprint.


## Paint

Paint uses exactly three public nodes:

~~~text
IMAGE + MASK + VAE
        |
        v
Krea2 CcC Paint Prepare
        +--> paint_context --> Krea2 CcC Paint --> conditioning/model
        +--> latent -------------------------------> KSampler
        +--> paint_geometry ------------------------+
                                                     |
KSampler --> VAE Decode --> Krea2 CcC Paint Restore-+
~~~

**Paint Prepare** now owns what used to be the standalone geometry stage:

- pad to a curated Krea working geometry (no crop selector)
- no source-image resize during geometry normalization
- edge, reflect, neutral, or white padding fill
- explicit expand_left/top/right/bottom outpaint canvas expansion
- optional enclosed-hole fill
- signed mask grow/shrink
- directional feather
- semantic-reference preparation
- VAE known-image latent
- token-aligned noise_mask
- paint_geometry output for Restore

There is **no public Krea2 CcC Paint Geometry node**.

**Paint** consumes paint_context, grounds Qwen with the prepared semantic reference, attaches the pre-encoded appearance reference, and optionally enables the Paint runtime's internal reference K/V cache.

**Paint Restore** consumes the decoded image plus paint_geometry and removes only temporary Krea padding; its former crop-only generated_mask input was removed.

## LoRA and Text to Image

**LoRA Stack**, **LoRA Stack Sliders**, and **LoRA Stack with Prompt** each support 5 model-only slots. **LoRA Stack with Prompt** supports per-slot positive prepend/append triggers with per-slot positive prepend/append triggers and outputs `prompt_augmentation`. Connect that output explicitly to Edit, Paint, or T2I to apply it; Edit always keeps its grounded negative prompt empty.

**Text to Image** encodes positive/negative text with the supplied CLIP, creates its empty latent from the shared Krea geometry presets/policies, and returns MODEL / positive / negative / LATENT. Its Vision references follow the same two-mode contract as Edit: Visual retains image-token rows, and Semantic removes them after multimodal encoding. No additional descriptions or preset prompts are generated. A nonempty free-form instruction is added to the conditioning text; an empty instruction is omitted. The explicit negative prompt remains text-only.

## Example Workflows

The `workflows/` directory contains **seven integrated workflows ordered from the simplest use case to the most complex**. Start at 01 and progress as you become familiar with Krea2 CcC:

| Step | Workflow | What it demonstrates |
| --- | --- | --- |
| 01 | [Creative Generation](workflows/01_creative_generation.json) | Vision Reference + Prompt Creator + T2I; optional LoRA Stack |
| 02 | [Outpainting & Reframing](workflows/02_outpainting_reframing.json) | Extend a source image with Paint Prepare, Paint and Restore |
| 03 | [Local Image Editing](workflows/03_local_image_editing.json) | Layout Composer, SAM masking, inverse selection and background replacement with Paint |
| 04 | [Identity & Scene Transfer](workflows/04_identity_scene_transfer.json) | Two visual references, regional masks, Prompt Creator and Edit |
| 05 | [Character Consistency](workflows/05_character_consistency.json) | Ordered body/face mask boosts and identity preservation across scenes |
| 06 | [Outfit Transfer](workflows/06_outfit_transfer.json) | Three coordinated references (scene, identity, outfit) plus regional masks |
| 07 | [Scene Composition](workflows/07_scene_composition.json) | Multiple layout cells, composited scene blueprint, subject reference and Edit |

**How to try them:** replace `LoadImage` placeholders with your own images and verify all model, LoRA, and SAM filenames against your ComfyUI installation. Use a fixed seed when comparing results. These are starting workflows and still need GPU generation testing.

**Boost:** `-5` blocks direct VAE/RoPE reference attention, `0` is neutral, and `+5` corresponds to 10×. For Paint, the mask selects pixels to regenerate; it is not a reference-attention Boost mask.


See NODES.md for exact public-node controls and ARCHITECTURE.md for runtime/data-flow details.

## Acknowledgements

ComfyUI-Krea-CcC was informed by work from the ComfyUI and Krea 2 community. Projects and ideas that influenced the implementation are documented in NOTICE.

**RoPE preview:** Edit emits an `IMAGE` batch with four canonical views (`layout`, `overlay_boxes`, `overlay_visuals`, `regional_boosts`) followed by **one separate spatial Boost view per VAE/RoPE reference**. Every view uses the same coordinates computed by the runtime RoPE planner; the fourth no longer uses a centered gallery. When references overlap, the fourth view shows the actual overlay order; the additional per-reference frames preserve each image and mask at its own RoPE position so no reference disappears behind another. Target content is shown if available. All reference previews use the same source geometry as VAE preparation; collapsed references are displayed as points. Vision-only references do not appear. Boost -5 darkens the image by 90%, Boost +5 by 5%, with intermediate values interpolated linearly. The source image is not alpha-blended a second time into the preview, preserving visible detail. This is preview rendering only and does not change generation attention.


**Reference routing update:** Every Reference retains its source image whether VAE/RoPE and Vision are on or off. The previous Visual widgets are named VAE/RoPE (`vae_rope_enable` and `vae_rope_*`). Prompt Creator accepts one ordered `references` chain, analyzes its images once in chain order and forwards that same chain unchanged. To use a scene and a subject, connect Scene Reference to Subject Reference via `previous_references`, then connect Subject Reference to Prompt Creator. Both VAE/RoPE and Vision enable states are preserved. Existing workflows using the former IMAGE socket/widget names require migration.

## SAM 3.1 masks for Edit and Paint

Install the official Comfy-Org model `sam3.1_multiplex_fp16.safetensors` in
`ComfyUI/models/checkpoints/` and use a recent ComfyUI containing the native
`SAM3_Detect` node. Source: https://huggingface.co/Comfy-Org/sam3.1

`CheckpointLoaderSimple` provides SAM's own `MODEL` and `CLIP` to
`Krea2 CcC Mask` through `sam_model` and `sam_clip`. These are **not**
the Krea2 generation model/CLIP.

Each Mask accepts an IMAGE and a `source` selector (`sam3` or `mask`).
`sam3` runs native SAM 3.1 with `prompt` and `confidence` (plus optional
`unload_sam_model` at the end of the chain); `mask` uses the connected
standard `mask` socket and disables irrelevant SAM controls in the UI. Feed the prior `mask_attention`
into the next `previous_mask_attention`, and feed the prior `image` into its
`image` input. The required `invert` boolean (default `false`) flips the normalized selected mask (`1 - mask`) **before** applying its regional Boost, merging with `previous_mask_attention`, calculating the union MASK output, and rendering `composite_image`. It works with both `sam3` and external `mask` sources. The original `image` output is unchanged. `invert` is the last widget to preserve positional compatibility with older saved workflows.
Outputs are the unchanged image, the union of all effective masks,
the regional-attention chain, `composite_image` and diagnostic info.
`composite_color` uses ComfyUI’s native color picker (default black, `#000000`); the saved value is a six-digit RGB hex color.
The `composite_image` output replaces the former `preview` at the **same output
index** and may be connected directly to a Reference's `image` input
(including Vision). It is a real composited IMAGE, not an attention-weight
input. Compositing uses the full ordered Mask chain: the **last mask wins**
where regions overlap. Pixels outside all masks remain unchanged in
`composite_image`. Only masked pixels receive the selected overlay:
Boost -5 is 90% composite color and Boost +5 is 5% composite color,
with linear interpolation. The untouched `image`, union `mask`, and
`mask_attention` outputs retain their prior behavior. The Reference's
own global VAE/RoPE Boost remains independent of the image composite.

For Paint, connect the **last** Mask's `image` and `mask` to
`Krea2 CcC Paint Prepare`; Paint consumes Paint Prepare's context.
For Edit, connect the **last** Mask's `mask_attention` into
`Krea2 CcC Reference.mask_attention` and ensure its `image` is
the corresponding Reference image. Pixels outside regional masks retain
Reference's global `boost`, whereas masked pixels take their own
**absolute** `boost` (last-overlapping region wins). A regional boost
of -5 hard-blocks direct target-to-reference attention for that region; zero is neutral.

To release SAM's GPU weights, set `unload_sam_model=true` **only in the
last SAM3 Mask** of a sequential chain. This requests a targeted offload
through ComfyUI's model manager. Other independent branches sharing the
same SAM model should finish before this point; unconnected branches may
execute in a different order. No global model unload is used.

SAM 3.1 text inference depends on the native ComfyUI API; upgrading the
ComfyUI core may be necessary. GPU inference and unload have not yet
been validated on the user's machine.

### Regional attention in Edit RoPE preview

`Krea2 CcC Edit.rope_preview` produces four initial IMAGE frames:
(1) RoPE layout, (2) placement boxes, (3) positioned reference images,
(4) positioned effective regional Boost masks. It then appends one frame
per VAE/RoPE reference showing only that reference's masked source image
**at its actual RoPE position**. These extra frames prevent one reference
from concealing another when RoPE positions overlap. Every frame shares
one canvas and RoPE coordinate system. Boost darkness is applied once
to the reference image (0 nearly black, 10 almost unchanged), and the
result is pasted without a second transparency layer. All images and
regional maps follow the same VAE-input crop/resize geometry. This is
a preview-only change; reference tokens and attention remain unchanged.

`Krea2 CcC Mask.confidence` controls native SAM3.1 detection threshold.
Native detector refinement remains at 2 iterations. Paint workflows
`16_paint_crop_restore` and `17_paint_no_kv_cache` exercise the new
Mask node in `external_mask` mode. Outpaint workflow
`15_paint_outpaint` intentionally uses no external segmentation:
Paint Prepare generates the expansion mask based on canvas geometry.

Prompt Creator `max_tokens` (32–32768) is passed unchanged as `max_length` to one ComfyUI `CLIP.generate` call. The native Qwen3-VL backend limits **newly generated tokens** and may stop earlier at EOS. `ANALYSIS` and `FINAL PROMPT` share this budget; reaching EOS after the first section does not prove token exhaustion. `creator_info` includes a decoded output prefix, generated token count, and last token ID where available. Missing sections are explicit errors rather than repaired automatically.
