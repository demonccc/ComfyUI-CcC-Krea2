# Changelog

## Unreleased

### Added

- Chainable `Krea2 CcC Attention Region` nodes with unique tags and percentage-based target boxes.
- Visual Reference regional attention scopes: `global`, `boost in region`, and `only in region`.
- Target-side region transport through Latent geometry, strict crop guardrails, and token-grid projection for regional attention.
- Regional attention example workflow with two tagged subjects.

- Paint Geometry node with reversible pad/crop-only Krea normalization, configurable edge/reflect/neutral/white padding, placement controls, and crop/pad guardrails.
- Paint Restore node for exact depadding or generated-crop compositing back to the requested native canvas.

- Portable visual-reference caches using safetensors.
- Cached raw VAE appearance latents for repeated target-specific edits.
- Cached Qwen3-VL visual features (merged, grid and DeepStack) while preserving prompt-dependent language conditioning.
- Cached Visual Reference node that mixes with normal ordered references.

### Changed

- Paint Prepare now consumes finalized Paint Geometry, owns VAE encoding, and returns the KSampler LATENT with token-aligned noise_mask.
- Paint runtime now consumes the prepared appearance latent and no longer creates the target latent itself.
- AnyPaint test workflow now uses Paint Geometry -> Paint Prepare -> Paint -> decode -> Paint Restore.

- Added Krea geometry policies to Krea2 CcC Latent for fixed and image-derived dimensions: nearest curated Krea aspect or aspect-preserving Krea bounds.
- Replaced Latent image-fit modes with explicit content-fit behavior: crop, contain with white padding, or stretch. Presets bypass geometry resolution.

- Replaced megapixel + aspect-ratio latent preset generation with explicit curated Krea target sizes. Preset labels show exact dimensions, aspect ratio, and approximate megapixels; fixed and image-derived modes remain separate.

- Standardized semantic grounding controls on a 32-pixel cadence to match Qwen3-VL visual grid alignment.

- Consolidated Krea2 CcC Edit around one Krea2 CcC Edit runtime.
- Removed runtime delegation to other installed custom nodes.
- Removed alternate edit backends and compatibility-only execution paths.
- Removed obsolete Subject, Scene, Outfit, Style and generic Reference public-node implementations that were no longer registered.
- Removed dead engine, settings, validation, mask, resolution and reference compatibility modules.
- Simplified the reference specification to the current visual, semantic and style contract.
- Split public editing into Visual Reference, Semantic Reference, Size Resolver, Latent and Edit nodes.
- Visual Reference keeps Qwen preparation independent from VAE geometry.
- Visual Reference exposes crop, resize and native appearance modes plus explicit RoPE placement.
- Reference latents, fit state, boosts and RoPE placement travel through CONDITIONING.
- Positive conditioning uses configured per-reference boosts; grounded negative uses neutral boosts.
- Semantic-only references remain Qwen-only and do not add VAE reference latents.
- Size Resolver outputs only width and height.
- Latent owns final target dimensions, /16 alignment and optional target content.
- Documentation now describes only the current Krea2 CcC Edit architecture. Attribution and project influences remain in NOTICE.
