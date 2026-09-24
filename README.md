# ComfyUI-CcC-Krea2

Krea2 CcC Edit provides ComfyUI nodes for Krea 2 generation, reference-guided editing, prompt creation, character sheets, and AnyPaint-style inpaint/outpaint.

The public surface currently contains **14 nodes**. Reference Cache nodes and the standalone Paint Geometry node are not part of the public node registry.

## Public Nodes

| Area | Node | Purpose |
| --- | --- | --- |
| Edit | **Krea2 CcC Visual Reference** | Ordered appearance reference with VAE fit, boost, RoPE placement, optional Qwen grounding, and optional regional attention. |
| Edit | **Krea2 CcC Semantic Reference** | Qwen-only semantic/style reference; no appearance latent. |
| Edit | **Krea2 CcC Attention Region** | Tagged target box used by regional Visual Reference attention. |
| Edit | **Krea2 CcC Size Resolver** | Resolves width/height from one image's long edge and another image's aspect ratio. |
| Edit | **Krea2 CcC Latent** | Owns target dimensions, Krea geometry policy, optional target content, semantic target context, and attention-region transport. |
| Edit | **Krea2 CcC Edit Prompt Creator** | Optional multimodal prompt creator using the same Krea2/Qwen3-VL CLIP as Edit. |
| Edit | **Krea2 CcC Edit** | Final edit orchestrator and Krea2 CcC edit runtime. |
| Utility | **Krea2 CcC Character Sheet** | Deterministically composes generic portrait/body references into one Krea-aligned image. |
| Paint | **Krea2 CcC Paint Prepare** | Owns Paint geometry, outpaint expansion, mask processing, semantic reference, VAE preparation, and sampling latent. |
| Paint | **Krea2 CcC Paint** | Builds AnyPaint/Krea Paint conditioning and installs the Paint runtime. |
| Paint | **Krea2 CcC Paint Restore** | Restores decoded Paint output to the requested canvas. |
| LoRA | **CcC Krea2 - LoRA Prompt Settings** | Defines optional prompt augmentation associated with up to four LoRA slots. |
| LoRA | **CcC Krea2 - LoRA Stack** | Applies up to four model-only LoRAs and carries optional prompt augmentation. |
| T2I | **CcC Krea2 - Text to Image** | Native Krea2 text-to-image conditioning and empty latent helper. |

## Edit Architecture

The direct edit path remains simple:

~~~text
Visual Reference(s) --------------------+
Semantic Reference(s) ------------------+--> Krea2 CcC Edit --> MODEL / CONDITIONING / LATENT
                                        ^
Attention Region(s) --> Latent ---------|
Size Resolver -------> Latent ----------+
~~~

**Edit Prompt Creator** is optional. Existing workflows can continue connecting a prompt and Visual References directly to Edit.

~~~text
Krea2/Qwen3-VL CLIP ----+-------------------------------> Edit
                        |
                        +--> Edit Prompt Creator
Visual References ------+--> Edit Prompt Creator --> Visual References passthrough --> Edit
user_prompt ------------+--> Edit Prompt Creator --> created_prompt ----------------> Edit
reference_edit_image ---> Edit Prompt Creator
~~~

The Creator does not modify Visual Reference, Semantic Reference, Latent, or Edit. In create_from_image, reference_edit_image is internal analysis input: the generated text must describe the observed situation explicitly instead of assuming Edit can see that image.

**Character Sheet** is also a utility before Visual Reference. It creates one deterministic IMAGE; the result can then be connected to a normal Visual Reference.

## Visual Reference

Each Visual Reference is an ordered appearance reference with two independent representations.

The VAE/appearance path supports:

- crop: use the target as an inside crop window over the source.
- resize: map the source long edge to the corresponding target axis, preserve aspect ratio, then crop down only the minimum pixels required for /16 alignment.
- contain: uniformly fit the source inside the target, preserve aspect ratio, then crop down only the minimum pixels required for /16 alignment.
- native: keep 1:1 source pixel scale and center-crop width/height down to /16.

RoPE placement is independent from pixel fit.

The Qwen path is controlled by semantic. When enabled, the same image can be downscaled only when it exceeds semantic_grounding_px. prompt_annotation is identification-only and becomes Image N: <annotation>; edit instructions belong in Edit's positive prompt.

Positive conditioning uses each reference's configured boost. The grounded negative branch uses the same appearance images, no negative text, no role annotations, and neutral reference boost 1.0.

Regional attention is optional:

- global
- boost in region
- only in region

A regional reference binds to a tagged **Attention Region** carried through Latent geometry.

## Semantic Reference

Semantic Reference is Qwen-only and does not create an appearance/VAE reference latent.

Modes:

- semantic_only
- style_direct
- style_indirect

semantic_only always uses the full image. Style modes can use full, 2x2, or 4x4 processing.

## Latent and Target Geometry

Krea2 CcC Latent owns final target geometry.

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
- create_from_image: analyze reference_edit_image and turn its useful scene/action/composition into explicit edit text.
- create_from_theme: expand a high-level theme into a concrete new situation while keeping referenced subjects anchored.

Controls are max_tokens, temperature, top_p, and seed. Sampling uses that explicit seed, as required by the Qwen3-VL Generate Text path. The Visual Reference chain is returned unchanged.

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

**LoRA Stack** applies up to four model-only LoRAs with per-slot strength and a global multiplier.

**LoRA Prompt Settings** can associate prepend/append positive and negative text with those four LoRA slots. The resulting prompt_augmentation is carried by LoRA Stack. The public **Text to Image** node consumes that augmentation; **Krea2 CcC Edit does not automatically consume it**.

**Text to Image** encodes positive/negative text with the supplied CLIP, resolves a /16-aligned latent from aspect ratio + megapixels, and returns MODEL / positive / negative / LATENT.

## Example Workflows

- workflows/01_scene_subject.json — ordered Scene + Subject Visual References, Size Resolver, Latent, Edit.
- workflows/02_anypaint_remove_people.json — current three-node Paint path: Paint Prepare -> Paint -> Paint Restore.
- workflows/03_regional_attention.json — tagged regional attention with two identity references.
- workflows/04_character_sheet_identity.json — generic portrait/body Character Sheet used as one Visual Reference.
- workflows/05_edit_prompt_creator.json — create_from_image Prompt Creator; the same reference image is wired to a Semantic Reference that is intentionally disabled/muted.

See NODES.md for exact public-node controls and ARCHITECTURE.md for runtime/data-flow details.

## Acknowledgements

Krea2 CcC Edit was informed by work from the ComfyUI and Krea 2 community. Projects and ideas that influenced the implementation are documented in NOTICE.
