# Architecture

## Public Surface

13 public nodes. For the canonical inventory see [NODES.md](NODES.md).

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

## Single Edit Runtime

One runtime, no alternate model edit backend. A unified Reference is tagged
and can contribute a Qwen Vision image, an appearance VAE latent, both, or neither.
References are chained in declared order, with matching tags merged to prevent
duplication. Appearance latents travel through CONDITIONING with their own
`visual_boost` and RoPE placement.

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


## Target Geometry

Size Resolver and Latent resolve target pixel geometry and /16 constraints.
Visual appearance fit and Qwen grounding are independently prepared.
The legacy regional Attention Region input and attention masks are no longer
exposed through public nodes.

## Prompt Creator

Edit Prompt Creator can inspect the Qwen-enabled Visual Reference chain and
produce an enhanced or image-grounded prompt. It is optional and is not a second edit runtime.

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

Text to Image consumes optional prompt_augmentation and optional vision_references, resolves its empty latent through the shared Krea geometry library, and returns the unchanged MODEL plus conditioning and LATENT. Its Vision adapter uses Qwen multimodal generation as an analysis step for Semantic, Visual, Style, and Context, then converts that analysis into text-only Krea conditioning. Direct Qwen vision rows are deliberately not used in T2I because the attempted direct transport was unsafe with the T2I conditioning path. The explicit negative prompt remains text-only.

The checked-in `workflows/01_t2i_one_reference.json` demonstrates the direct runtime path:

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

Krea2 CcC Edit does not consume LoRA Prompt Settings or prompt_augmentation automatically.

## Ownership

The public architecture is implemented by the nodes registered in ccc_krea2/nodes.py. Internal helper modules may contain implementation support that is not a public ComfyUI node.

External projects that influenced individual ideas or techniques are acknowledged separately in NOTICE.
