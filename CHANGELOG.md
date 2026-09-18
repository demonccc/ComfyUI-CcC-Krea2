# Changelog

## Unreleased

### Added

- Portable visual-reference caches using safetensors.
- Cached raw VAE appearance latents for repeated target-specific edits.
- Cached Qwen3-VL visual features (merged, grid and DeepStack) while preserving prompt-dependent language conditioning.
- Cached Visual Reference node that mixes with normal ordered references.

### Changed

- Consolidated Krea2 Edit around one CcC runtime.
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
- Documentation now describes only the current CcC architecture. Attribution and project influences remain in NOTICE.
