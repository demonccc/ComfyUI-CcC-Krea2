# ComfyUI-Krea-CcC

ComfyUI-Krea-CcC provides ComfyUI nodes for Krea 2 generation, text-to-image, reference-guided editing, prompt creation, Vision guidance, character sheets, LoRA workflows, and AnyPaint-style inpaint/outpaint.

The public surface currently contains **15 registered nodes** (including the unified Reference and the two legacy reference nodes). The former Reference Cache feature has been removed. Paint geometry remains internal to Paint Prepare/Restore.

## Unified Reference (new workflows)

**Krea2 CcC Reference** combines the former Visual and Vision reference controls in one node and one source `image`. The controls are ordered with `visual_enable` and all Visual settings first, followed by `vision_enable` and all Vision settings. Both groups are enabled by default. Disabled groups are greyed out in ComfyUI and are not processed. Vision optionally downsizes the source with bicubic antialiasing (`vision_resize=true`, `vision_max_edge=768`) independently of Visual, without cropping or upscaling. The combined node omits the legacy guidance detail/sampling presets, which have no equivalent Edit effect.

- **Visual only**: VAE appearance reference, Fit, RoPE, Boost, reference mask and target attention mask. It does **not** send the image to Qwen.
- **Vision only**: Qwen Semantic / Visual / Style / Context / Native guidance without any appearance/VAE reference.
- **Both**: one source image contributes one Visual entry and one Vision entry to the same ordered reference chain; Qwen is not fed a duplicate semantic copy from the Visual path.
- **Neither**: invalid configuration. At least one path must be enabled.

The Krea2 Edit LoRA remains attached to the model as before. Its image-reference attention is provided by the enabled Visual path; the Vision path supplies separate Qwen conditioning and is **not** a substitute for the LoRA's spatial reference mechanism. Fidelity of the Vision modes with the LoRA still needs image-generation testing.

Legacy `Krea2 CcC Visual Reference` and `Krea2 CcC Vision Reference` nodes remain registered solely so saved workflows are not broken; new workflows should use **Krea2 CcC Reference**. Existing `workflows/` files were intentionally not modified.

## Public Nodes

| Area | Node | Purpose |
| --- | --- | --- |
| Edit | **Krea2 CcC Visual Reference** | Ordered appearance reference with VAE fit, RoPE placement, Qwen grounding, boost and optional source/target attention masks. |
| Edit | **Krea2 CcC Vision Reference** | Qwen/Vision-only Semantic, Visual, Style, or Context reference; no appearance latent. |
| Edit | **Krea2 CcC Size Resolver** | Resolves width/height from one image's long edge and another image's aspect ratio. |
| Edit | **Krea2 CcC Latent** | Owns target dimensions, Krea geometry policy, optional target content, semantic target context, and an aligned `target_image` output for masks. |
| Edit | **Krea2 CcC Edit Prompt Creator** | Optional multimodal prompt creator using the same Krea2/Qwen3-VL CLIP as Edit. |
| Edit | **Krea2 CcC Edit** | Final edit orchestrator and Krea2 CcC edit runtime. |
| Utility | **Krea2 CcC Character Sheet** | Deterministically composes generic portrait/body references into one Krea-aligned image. |
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
Latent.target_image -> ComfyUI mask editor -> Visual Reference.target_attention_mask
~~~

Each Visual Reference may receive `reference_boost_mask` (painted on the reference image) and `target_attention_mask` (painted on `Latent.target_image`). The first selects which reference tokens receive boost; the second controls which target tokens receive the boost or direct-attention restriction. Both travel with their reference; no tags or separate region chain exist. Masks must have the **exact pixel dimensions** of their respective source or final target image; mismatches are errors, not automatically corrected. An Attention Mask adjusts reference influence, whereas Paint/Inpaint controls pixel replacement.

The reference nodes can be chained in any order using `previous_references`; they expose one shared `references` output. Edit accepts a single `references` input. Appearance references go through VAE and (optionally) Qwen; Vision references are Qwen-only.

**Edit Prompt Creator** is optional:

~~~text
references --> Edit Prompt Creator --> references --> Edit
                     |                    ^
CLIP ----------------+                    |
user_prompt ---------+--> created_prompt -+--> Edit.positive_prompt
reference_edit_image-+
~~~

Prompt Creator examines **only Visual Reference entries** from the mixed reference chain and passes the entire chain through unchanged. Its optional reference_edit_image is used internally for prompt analysis, not as an appearance reference.

**Character Sheet** is also a utility before Visual Reference. It creates one deterministic IMAGE; the result can then be connected to a normal Visual Reference.

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

Positive conditioning uses each reference's configured boost. The grounded negative branch uses the same appearance images, no negative text, no role annotations, and neutral reference boost 1.0.

Visual Reference attention supports `global`, `boost in mask`, and `only in mask`. Each Visual Reference accepts an optional `reference_boost_mask` (source-image pixels, transformed with the image) and `target_attention_mask` (final target-image pixels). Masks stay with their owning reference; there are no tags or separate Attention Region nodes. Latent exports `target_image` to provide a mask-painting canvas matching its final geometry.

## Vision Reference

Vision Reference declares what information a source image represents. It never creates an appearance/VAE reference latent, and it does not prescribe how a consumer must transport that information.

Modes:

- **Semantic** — content/meaning: subject category and attributes, pose, action, objects, composition, and spatial relationships.
- **Visual** — near-copy/full visual blueprint: subject appearance, clothing, pose, gaze, scene, composition, lighting, colors, and visual style are preserved as faithfully as possible unless the target prompt explicitly changes them. Subject count is locked: no clones, repeated variants, panels, or duplicated subjects.
- **Style** — reuses the exact legacy Visual guidance preset that previously produced the desired style-transfer behavior, then applies that guidance strongly as the dominant visual treatment. Style processing may still analyze the full image or 2x2 / 4x4 tiles.
- **Context** — mandatory scene staging: setting, framing/viewpoint, exact pose, head orientation, gaze, gesture, hands/limbs, interaction, composition, depth, object placement, and spatial relationships are treated as constraints rather than loose description, without copying identity, clothing, or visual style.

Only **Style** uses the `style_processing` control. Semantic, Visual, and Context are fixed to full-image processing. The optional `instruction` is stored with the reference and each consumer decides how to apply it within the selected mode. T2I currently converts all Vision modes into Qwen-generated text guidance before final Krea conditioning. Guidance Detail changes the extraction policy as well as its token budget; Vision Guidance has deliberately separated conservative/balanced/flexible sampling behavior. Guidance Seed, Guidance Detail, and Vision Guidance are available for Semantic, Visual, Style, Context, and Native.

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

Modes:

- enhance: improve an existing edit instruction without changing its intent.
- create_from_image: analyze reference_edit_image and reconstruct a detailed situation instead of collapsing it to a short summary.
- create_from_theme: expand a high-level theme into a concrete, detailed new situation while keeping referenced subjects anchored.
- custom: edit and send the system_prompt directly.

The system_prompt field always shows the effective prompt. For preset modes it is greyed/read-only; custom enables editing. Switching to custom starts from the current preset unless an earlier custom value exists.

create_from_image asks Qwen to capture relevant pose/body position, other people and interactions, props, environment, spatial relationships, framing, camera angle, composition, and lighting while preserving explicit user constraints such as face, identity, anatomy, body shape, and body proportions.

thinking uses the Qwen3-VL thinking convention and pre-fills the assistant turn with <think> when enabled. The reasoning before </think> is returned separately on the thinking output, while only the final edit text remains in created_prompt. An unfinished thinking block raises an error instead of producing an empty/invalid final prompt.

All preset modes enforce final edit text with no system/meta preamble. Downstream visible references are named Image 1, Image 2, etc.; accidental "Krea Image N" wording is normalized to "Image N".

The Visual Reference chain is returned unchanged.

## Character Sheet

Character Sheet accepts generic portrait_1..4 and body_1..2 inputs. Numbers indicate ordering only; specific camera angles are not required.

Available layouts:

- 1 portrait
- 2 portraits
- 3 portraits
- 4 portraits
- 1 body
- 2 bodies
- 1 portrait + 2 bodies
- 3 portraits + 1 body
- 4 portraits + 1 body

It always preserves source aspect ratio and fits the complete image inside each slot. There is no crop/cover/stretch mode.

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

- pad or crop to a curated Krea working geometry
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

**Paint Restore** consumes the decoded image plus paint_geometry. Pad mode removes only temporary Krea padding; crop mode composites the generated crop back into the preserved requested canvas.

## LoRA and Text to Image

**LoRA Stack**, **LoRA Stack Sliders**, and **LoRA Stack with Prompt** each support 5 model-only slots. **LoRA Stack with Prompt** supports per-slot positive prepend/append triggers with per-slot positive prepend/append triggers and outputs `prompt_augmentation`. Connect that output explicitly to Edit, Paint, or T2I to apply it; Edit always keeps its grounded negative prompt empty.

**Text to Image** encodes positive/negative text with the supplied CLIP, creates its empty latent from the shared Krea geometry presets/policies, and returns MODEL / positive / negative / LATENT. Qwen guidance controls do not live on T2I: they travel with **Vision Reference**. With a Vision Reference, T2I uses Qwen multimodal generation only as an analysis step, then encodes the resulting mode-scoped guidance as text-only Krea2 conditioning. Direct Vision rows never enter the final T2I conditioning, so the source image cannot overpower the prompt or appear as an extra subject. The explicit negative prompt remains text-only.

## Example Workflows

**Compatibility note:** Workflow JSON files have not been edited during the refactor. Saved workflows may contain removed node types or outdated inputs; the legacy regional-attention workflow is explicitly identified below.

- workflows/01_scene_subject.json — ordered Scene + Subject Visual References, Size Resolver, Latent, Edit; BodySwap disabled by default.
- workflows/02_anypaint_remove_people.json — current three-node inpaint path: Paint Prepare -> Paint -> Paint Restore.
- workflows/03_regional_attention.json — **legacy/incompatible** with the current mask API: still uses removed tagged Attention Region nodes. Kept unchanged pending workflow migration.
- workflows/04_character_sheet_identity.json — generic portrait/body Character Sheet used as one Visual Reference; BodySwap disabled by default.
- workflows/05_edit_prompt_creator.json — Prompt Creator create_from_image preset.
- workflows/06_text_to_image.json — Krea2 text-to-image path with optional Vision Reference used to produce text guidance before final conditioning.
- workflows/07_prompt_creator_enhance.json — Prompt Creator enhance preset.
- workflows/08_prompt_creator_create_from_theme.json — Prompt Creator create_from_theme preset.
- workflows/09_prompt_creator_custom.json — Prompt Creator custom-mode contract test.
- workflows/10_edit_fit_crop.json — Edit reference_fit=crop path.
- workflows/11_edit_fit_contain.json — Edit reference_fit=contain path.
- workflows/12_edit_vision_semantic.json — Edit with Vision Reference / Semantic.
- workflows/13_edit_vision_visual.json — Edit with Vision Reference / Visual.
- workflows/14_edit_vision_style.json — Edit with Vision Reference / Style using 4x4 style processing.
- workflows/15_paint_outpaint.json — expansion-only outpaint path.
- workflows/16_paint_crop_restore.json — crop geometry + reverse restore/composite path.
- workflows/17_paint_no_kv_cache.json — Paint runtime with KV cache disabled.
- workflows/18_edit_vision_context.json — Edit with Vision Reference / Context.

See NODES.md for exact public-node controls and ARCHITECTURE.md for runtime/data-flow details.

## Acknowledgements

ComfyUI-Krea-CcC was informed by work from the ComfyUI and Krea 2 community. Projects and ideas that influenced the implementation are documented in NOTICE.

**RoPE preview:** Edit emits one `IMAGE` batch containing three annotated views in order: `layout`, `overlay_boxes` and `overlay_visuals`. When target Latent content is `from_image`, the VAE-prepared target image is used; when `empty`, a neutral rectangle is used. Visual Reference images are prepared with the same pixel-space geometry as the VAE. `collapsed` references show a point, and Qwen-only Vision References are not shown.


**Reference routing update:** Every Reference retains its source image whether VAE/RoPE and Vision are on or off. The previous Visual widgets are named VAE/RoPE (`vae_rope_enable` and `vae_rope_*`). Prompt Creator accepts a Reference chain for `reference_edit_image`, analyzes it first and forwards both chains, while disabled conditioning stays disabled. Existing workflows using the former IMAGE socket/widget names require migration.
