# ComfyUI-Krea-CcC

ComfyUI-Krea-CcC provides ComfyUI nodes for Krea 2 generation, text-to-image, reference-guided editing, prompt creation, Vision guidance, character sheets, LoRA workflows, and AnyPaint-style inpaint/outpaint.

The public surface currently contains **13 nodes**. Reference Cache nodes and the standalone Paint Geometry node are not part of the public node registry.

## Public Nodes

| Node | ID | Purpose |
| --- | --- | --- |
| Krea2 CcC Reference | `CcCKrea2Reference` | Dual Qwen/VAE reference with tags and strengths |
| Krea2 CcC Size Resolver | `CcCKrea2SizeResolver` | Resolve target image size |
| Krea2 CcC Latent | `CcCKrea2Latent` | Prepare target latent |
| Krea2 CcC Edit | `CcCKrea2Edit` | Encode references and configure edit runtime |
| Krea2 CcC Edit Prompt Creator | `CcCKrea2EditPromptCreator` | Generate edit prompt from image and/or instructions |
| Krea2 CcC Character Sheet | `CcCKrea2CharacterSheet` | Compose reference image |
| Krea2 CcC Paint Prepare | `CcCKrea2PaintPrepare` | Prepare masked image and latent |
| Krea2 CcC Paint | `CcCKrea2Paint` | Paint conditioning and runtime |
| Krea2 CcC Paint Restore | `CcCKrea2PaintRestore` | Restore output |
| Krea2 CcC Lora Stack with Prompt | `CcCKrea2LoRAStackWithPrompt` | Patch LoRAs plus prompt metadata |
| Krea2 CcC Lora Stack | `CcCKrea2LoRAStack` | Patch LoRAs |
| Krea2 CcC Lora Stack Sliders | `CcCKrea2LoRALoader` | Slider LoRA loader |
| Krea2 CcC Text To Image | `CcCKrea2TextToImage` | Generate T2I conditioning and latent |

## Edit Architecture

Single Edit runtime, with separate Qwen conditioning and VAE appearance paths.

## Krea2 CcC Reference

The reference node unifies the previous Visual Reference and Vision Reference interfaces.

- `tag`: reference identifier such as `subject`, `outfit` or `scene`
- `semantic_enabled`: send the reference to Qwen Vision; use `vision_mode` (Semantic, Visual, Style, Context) and `vision_instruction`
- `visual_enabled`: encode the appearance reference via VAE/DiT, independently
- `vision_strength`: influence of that reference's Qwen Vision rows **only when** `vision_attention_patch` is enabled in Edit
- `visual_boost`: independent VAE/DiT appearance reference boost, unaffected by the Qwen patch
- `reference_fit`: crop, resize, contain, native; related RoPE and resampling controls remain available
- `semantic_grounding_px`: controls Qwen image preparation
- optional Style processing and Qwen generation controls are retained

The node emits separate typed visual and vision chains for compatibility with Edit, Prompt Creator and T2I. Edit merges matching tags into one logical reference to avoid duplicate Qwen vision images. Qwen-only and VAE-only operation is supported.

The old regional Attention Region / attention mask node and sockets are removed from the **public** workflow API. Regional attention is not replaced by the Qwen vision weighting feature.

## Vision attention patch

`Krea2 CcC Edit` has a `vision_attention_patch` toggle (default **OFF**). When ON, the runtime takes per-image Qwen vision token spans from the conditioning builder and applies `vision_strength` using a ComfyUI `attn1_patch` V-scaling hook. It skips unconditional CFG rows. When OFF, the existing Qwen path is unchanged. `visual_boost` remains separate and always governs appearance latents.

This is experimental. Images with heavy Style or Context row compaction retain only their remaining Qwen rows, which are remapped before weighting. No regional attention mask is used by the public nodes.

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

**LoRA Stack** applies up to four model-only LoRAs with per-slot strength and a global multiplier.

**LoRA Prompt Settings** can associate prepend/append positive and negative text with those four LoRA slots. The resulting prompt_augmentation is carried by LoRA Stack. The public **Text to Image** node consumes that augmentation; **Krea2 CcC Edit does not automatically consume it**.

**Text to Image** encodes positive/negative text with the supplied CLIP, creates its empty latent from the shared Krea geometry presets/policies, and returns MODEL / positive / negative / LATENT. Qwen guidance controls do not live on T2I: they travel with **Vision Reference**. With a Vision Reference, T2I uses Qwen multimodal generation only as an analysis step, then encodes the resulting mode-scoped guidance as text-only Krea2 conditioning. Direct Vision rows never enter the final T2I conditioning, so the source image cannot overpower the prompt or appear as an extra subject. The explicit negative prompt remains text-only.

## Example Workflows

| File | Scenario |
| --- | --- |
| [`workflows/01_t2i_one_reference.json`](workflows/01_t2i_one_reference.json) | T2I one image reference |
| [`workflows/02_t2i_prompt_edit.json`](workflows/02_t2i_prompt_edit.json) | T2I with prompt enhancement |
| [`workflows/03_i2i_one_reference.json`](workflows/03_i2i_one_reference.json) | I2I one reference |
| [`workflows/04_i2i_two_references.json`](workflows/04_i2i_two_references.json) | I2I two references |
| [`workflows/05_i2i_three_references.json`](workflows/05_i2i_three_references.json) | I2I three references |
| [`workflows/06_i2i_prompt_edit.json`](workflows/06_i2i_prompt_edit.json) | I2I with Edit Prompt Creator |
| [`workflows/07_i2i_paint.json`](workflows/07_i2i_paint.json) | I2I Paint |

## Acknowledgements

ComfyUI-Krea-CcC was informed by work from the ComfyUI and Krea 2 community. Projects and ideas that influenced the implementation are documented in NOTICE.
