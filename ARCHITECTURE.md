# Architecture

## Public Surface

The current package registers **14 public nodes**:

~~~text
Edit:
  Krea2 CcC Attention Region
  Krea2 CcC Visual Reference
  Krea2 CcC Semantic Reference
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
  CcC Krea2 - LoRA Prompt Settings
  CcC Krea2 - LoRA Stack
  CcC Krea2 - Text to Image
~~~

There is no public Reference Cache node family and no public Paint Geometry node. Paint geometry helpers remain internal implementation details used by Paint Prepare and Paint Restore.

## Single Edit Runtime

Krea2 CcC Edit has one edit execution path.

~~~text
Attention Region(s) ------------------------+
IMAGE -> Visual Reference(s) ---------------+
IMAGE -> Semantic Reference(s) -------------+--> Edit --> Krea2 CcC Edit runtime --> Krea2
                                             ^
Size Resolver -> Latent ---------------------+
~~~

No alternate edit runtime is selected based on other installed custom nodes.

### Optional Prompt Creator

Edit Prompt Creator is an optional pre-processing layer, not another edit runtime.

~~~text
Krea2/Qwen3-VL CLIP ----------------+-----------------------> Edit
                                    |
Visual Reference chain ------------>+--> Prompt Creator --> Visual Reference passthrough --> Edit
user_prompt ------------------------>+--> Prompt Creator --> created_prompt ----------------> Edit
reference_edit_image --------------->+
~~~

The Creator uses the same multimodal CLIP generation path as ComfyUI Generate Text. It does not mutate the Visual Reference chain.

reference_edit_image is visible to the Creator only unless the user separately wires that image into another existing path. Therefore create_from_image converts useful scene/action/pose/environment/composition details into explicit text rather than relying on Edit seeing that internal image.

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
          crop | resize | contain | native
          VAE encode
          reference latent
          boost
          RoPE placement
          optional target attention scope
~~~

The Qwen copy does not need to match target pixel geometry. Appearance preparation resolves against the target geometry before VAE encoding.

The positive branch uses configured reference boosts. The grounded negative branch uses the same appearance-image vision blocks, fixed empty negative text, no role annotation text, and neutral boost 1.0.

### Tagged Target Attention Regions

Attention Region nodes form a chain of unique tags with percentage-based target/content boxes.

With empty Latent content, percentages refer directly to the final target canvas. With image content, Latent transports those boxes through stretch, contain, or crop.

Crop is strict: if the retained crop touches or removes any part of a declared region, Latent raises an error instead of clipping or relocating the region.

Edit resolves region_tag for each regional Visual Reference:

- boost in region: reference boost is applied only to target queries inside the region.
- only in region: target queries outside the region are blocked from attending to that reference.

## Semantic Reference

Semantic References are Qwen-only and do not create appearance latents.

semantic_only always uses the complete image. Style modes can process full, 2x2, or 4x4 physical Qwen images before style-span handling.

## Target Geometry

Latent owns target dimensions, /16 alignment, optional target content, target content placement, optional target semantic metadata, and transformed Attention Regions.

Dimension modes are from_image, fixed, and preset.

For image-derived and fixed dimensions:

- nearest_krea_aspect
- preserve_aspect_krea_bounds

Preset dimensions are explicit curated width/height pairs and bypass geometry policy.

When target content is enabled, content fit is independent: crop, contain, or stretch.

Visual Reference fit is a separate operation and supports crop, resize, contain, and native. RoPE placement is independent from Visual Reference pixel preparation.

### Semantic Grounding Cadence

Visual Reference semantic_grounding_px, Semantic Reference grounding_px, and Latent latent_grounding_px use a step of 32. This matches Qwen3-VL's effective spatial cadence.

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

LoRA Stack patches MODEL only and can carry immutable prompt augmentation assembled from LoRA Prompt Settings.

Text to Image consumes optional prompt_augmentation, encodes positive and negative text independently with CLIP, calculates a /16-aligned latent from aspect ratio + megapixels, and returns the unchanged MODEL plus conditioning and LATENT.

Krea2 CcC Edit does not consume LoRA Prompt Settings or prompt_augmentation automatically.

## Ownership

The public architecture is implemented by the nodes registered in ccc_krea2/nodes.py. Internal helper modules may contain implementation support that is not a public ComfyUI node.

External projects that influenced individual ideas or techniques are acknowledged separately in NOTICE.
