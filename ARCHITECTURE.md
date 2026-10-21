# Architecture

## Public Surface

The current package registers **15 public nodes**:

~~~text
Edit:
  Krea2 CcC Attention Region
  Krea2 CcC Visual Reference
  Krea2 CcC Vision Reference
  Krea2 CcC Size Resolver
  Krea2 CcC Latent
  Krea2 CcC Edit Prompt Creator
  Krea2 CcC Edit

Utility:
  Krea2 CcC Character Sheet

Paint:
  Krea2 CcC Paint Prepare
  Krea2 CcC Paint
  Krea2 CcC Paint Restore

LoRA / T2I:
  Krea2 CcC Lora Stack with Prompt
  Krea2 CcC Lora Stack
  Krea2 CcC Lora Stack Sliders
  Krea2 CcC Text To Image
~~~

The Reference Cache implementation was removed. There is no public Paint Geometry node. Paint geometry helpers remain internal implementation details used by Paint Prepare and Paint Restore.

## Single Edit Runtime

Krea2 CcC Edit has one edit execution path.

~~~text
IMAGE --> Visual Reference --> Vision Reference --> Visual Reference
             |  shared KREA2_REFERENCE_CHAIN         |
             +----------- previous_references --------+
                                                      |
                                                      v
                                                Edit.references
Attention Region(s) --> Latent -----------------> Edit.latent
Size Resolver ---------> Latent
~~~

Visual and Vision references may be interleaved. Each node appends its entry; Edit receives one ordered `references` chain and processes each type according to its own contract. Prompt Creator optionally reads only Visual entries and returns the complete chain without changes.

No alternate edit runtime is selected based on other installed custom nodes.

### Optional Prompt Creator

Edit Prompt Creator is an optional pre-processing layer, not another edit runtime.

~~~text
Krea2/Qwen3-VL CLIP ----------------+-----------------------> Edit
                                    |
Visual Reference chain ------------>+--> Prompt Creator --> Visual Reference passthrough --> Edit
user_prompt ------------------------>+--> Prompt Creator --> created_prompt ----------------> Edit
reference_edit_image --------------->+                    \--> thinking
system_prompt ---------------------->+
~~~

The Creator uses the same multimodal CLIP generation path as ComfyUI Generate Text. It does not mutate the Visual Reference chain.

Preset modes are enhance, create_from_image, and create_from_theme. Their canonical system prompts live in Python and are displayed by the frontend as read-only text. custom enables the same system_prompt field and sends that custom text to Qwen.

thinking is passed into Qwen3-VL tokenization. When enabled, Prompt Creator supplies a Qwen3-VL chat template with the assistant turn prefilled by <think>, forcing generation to begin inside the reasoning block. Text before </think> goes to the thinking output; text after </think> goes to created_prompt. If the block never closes, execution fails instead of treating unfinished reasoning as a final prompt.

create_from_image treats the internal reference edit image as a detailed visual blueprint rather than an identity source. Its preset asks Qwen to reconstruct relevant pose, body position, people/interactions, objects, scene layout, framing, viewpoint, composition, and lighting while preserving explicit user identity/body constraints.

Visible downstream references are named Image N. The final-output sanitizer removes known meta-preamble leakage and normalizes accidental "Krea Image N" wording before created_prompt is emitted.

reference_edit_image is visible to the Creator only unless the user separately wires that image into another existing path. The final prompt therefore verbalizes useful scene details instead of referring to the hidden image itself.

### Character Sheet

Character Sheet is a deterministic IMAGE utility before Visual Reference:

~~~text
portrait/body IMAGE inputs
        |
        v
Character Sheet
  preserve aspect
  fit complete source
  no crop/stretch
        |
        v
normal Visual Reference
~~~

It does not run Qwen, VAE, conditioning, or latent processing.

## Visual Reference: Two Independent Representations

A Visual Reference can travel through two independent paths.

~~~text
source image
   |
   +--> Qwen path
   |      optional semantic participation
   |      optional downscale-only cap
   |      optional role-only Image N annotation
   |
   +--> appearance path
          native | crop | contain | fit
          VAE encode
          reference latent
          boost
          RoPE placement
          optional target attention scope
~~~

The Qwen copy does not need to match target pixel geometry. Appearance preparation resolves against the target geometry before VAE encoding.

The positive branch uses configured reference boosts. The grounded negative branch uses the same appearance-image vision blocks, fixed empty negative text, no role annotation text, and neutral boost 1.0.

### Attention masks owned by Visual Reference

- `reference_boost_mask`: optional MASK on the source reference image. It follows the exact same pixel crop/resize as the reference before VAE encoding. Its selected tokens receive the configured reference `boost`.
- `target_attention_mask`: optional MASK on `Latent.target_image`, i.e. the final target canvas, **after** all content-fit geometry. Edit rejects wrong target dimensions instead of silently warping the mask.
- `attention_scope=global`: boost from all target queries to selected source tokens.
- `attention_scope=boost in mask`: boost only from target queries selected by the target mask.
- `attention_scope=only in mask`: also deny direct target queries outside the target mask from attending to this reference.

Each Visual Reference carries both optional masks in its own chain entry, so no independent mask ordering, separate Attention Region node or tag lookup exists. The source and target masks are downsampled to the actual reference/target DiT token grids in the attention computation; **latents are never refit after VAE**. Masks are batched independently.

## Vision Reference

Vision References declare Qwen/Vision information and do not create appearance latents. The declaration is consumer-neutral:

- **Semantic** -> content/meaning, pose/action, objects, composition, and spatial relationships.
- **Visual** -> near-copy/full visual blueprint. Balanced/Detailed treat pose, gaze, framing, composition, appearance, and subject count as anchors; Flexible may adapt pose/composition but never duplicates or multiplies subjects.
- **Style** -> exact legacy Visual guidance behavior, retained because that preset produced the desired style-transfer result; the resulting guidance is applied strongly as the dominant visual treatment; only this mode exposes full / 2x2 / 4x4 processing.
- **Context** -> mandatory scene staging: environment, viewpoint/framing, exact pose and limb placement, head orientation, gaze target, gesture, action/interaction, object placement, composition, crop, depth, and spatial relationships are treated as constraints without copying identity, clothing, or style.

Vision Reference has no public grounding or fidelity control. It carries Qwen text-guidance generation configuration used by the T2I Vision adapter for Semantic, Visual, Style, and Context. Guidance Detail changes extraction depth, not only token ceiling, and Vision Guidance intentionally uses widely separated conservative/balanced/flexible sampling presets. The consumer still owns how the declared information is transformed into conditioning.

## Target Geometry

Krea pixel geometry is defined in the shared `krea_geometry.py` library: curated presets, /16 alignment, bounds, and geometry policies. Latent consumes that library while owning target content, content placement, optional target semantic metadata, and a final target image for painting aligned masks. T2I consumes the same geometry source of truth.

Dimension modes are from_image, fixed, and preset.

For image-derived and fixed dimensions:

- nearest_krea_aspect
- preserve_aspect_krea_bounds

Preset dimensions are explicit curated width/height pairs and bypass geometry policy.

When target content is enabled, content fit is independent: crop, contain, or stretch.

Visual Reference fit is a separate operation and supports native, crop, contain, and fit. RoPE placement is independent from Visual Reference pixel preparation. Visual Reference supports overlay/stacked placement, spatial/collapsed token coordinates, and latent/source geometry selection; native uses source geometry. Edit emits a `rope_preview` IMAGE built with the runtime's shared RoPE planner. Only Visual References and the target are depicted.

### Semantic Grounding Cadence

Visual Reference `semantic_grounding_px` and Latent `latent_grounding_px` use a step of 32. Vision Reference does not expose a grounding cap; its source image is passed to Qwen at tokenizer-native resolution.

## Conditioning and Runtime Transport

Appearance reference latents are attached to CONDITIONING together with reference fit, positive boost, neutral negative boost, RoPE placement, optional target region, and target attention scope.

The Edit runtime executes one in-context sequence:

~~~text
[text | reference 1 | reference 2 | ... | target]
~~~

## Paint

Paint has three public phases.

~~~text
source IMAGE + MASK + VAE
        |
        v
Paint Prepare
  explicit outpaint expansion
  pad/crop to curated Krea working geometry
  no source resize
  fill holes / grow / feather
  semantic reference
  VAE known canvas
  token-aligned noise_mask
        |
        +--> paint_context --> Paint
        +--> LATENT ---------> KSampler
        +--> paint_geometry ----------------------+
                                                   |
KSampler -> VAE Decode                            |
        |                                          |
        +------------------> Paint Restore <--------+
~~~

Paint Prepare uses internal geometry helpers but exposes the geometry controls directly. pad chooses a curated Krea geometry that contains the requested canvas; crop chooses one that fits inside it. Impossible directions fail rather than resizing the known source.

Explicit outpaint expansion belongs to the requested/final canvas. Temporary Krea padding belongs only to the working canvas and is removed by Restore.

Paint consumes KREA2_PAINT_CONTEXT. It grounds Qwen with the prepared semantic reference, attaches the pre-encoded appearance reference, and installs the Paint runtime. Its kv_cache switch is an internal runtime K/V optimization; it is not the removed public Reference Cache feature.

Paint Restore uses paint_geometry from Paint Prepare:

- pad: remove temporary Krea padding only.
- crop: composite the decoded generated crop back at the exact original coordinates; optional generated_mask limits replacement to the generated/feathered region.

## LoRA and T2I

LoRA Stack and LoRA Stack Sliders each apply 10 MODEL-only LoRAs. LoRA Stack with Prompt applies 5 MODEL-only LoRAs and emits optional positive prompt augmentation.

Text to Image consumes optional prompt_augmentation and optional references, resolves its empty latent through the shared Krea geometry library, and returns the unchanged MODEL plus conditioning and LATENT. Its Vision adapter uses Qwen multimodal generation as an analysis step for Semantic, Visual, Style, and Context, then converts that analysis into text-only Krea conditioning. Direct Qwen vision rows are deliberately not used in T2I because the attempted direct transport was unsafe with the T2I conditioning path. The explicit negative prompt remains text-only.

The checked-in `workflows/06_text_to_image.json` demonstrates the direct runtime path:

~~~text
Vision Reference ----+
                     |
UNET + CLIP ---------+
                     v
Krea2 CcC Text To Image
    +--> MODEL ---------> ModelSamplingAuraFlow -> BlackwellAttentionFix -> KSampler
    +--> positive -----------------------------------------------> KSampler
    +--> negative -----------------------------------------------> KSampler
    +--> LATENT -------------------------------------------------> KSampler
                                                                    |
                                                                    v
                                                              VAE Decode
                                                                    |
                                                                    v
                                                                SaveImage
~~~

Krea2 CcC Edit accepts `prompt_augmentation` as an optional explicit input, preserving its fixed-empty negative text contract.

## Ownership

The public architecture is implemented by the nodes registered in ccc_krea2/nodes.py. Internal helper modules may contain implementation support that is not a public ComfyUI node.

External projects that influenced individual ideas or techniques are acknowledged separately in NOTICE.

### Reference geometry ownership

Visual Reference geometry (`native`, `crop`, `contain`, `fit`) is resolved in pixel space before VAE encoding. The Edit forward consumes those prepared reference latents **without cropping or resizing them to the target**. The old `reference_fit`/`ref_fit` runtime flag and latent-space `_fit_latent` fallback have been removed. DiT patch-size padding may still be applied when required; that is not image fit/resizing. RoPE token grids are derived from the reference tensors sent to the transformer.
