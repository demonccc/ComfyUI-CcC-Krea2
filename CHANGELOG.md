# Changelog

## Unreleased

### Added

- Optional Krea2 CcC Edit Prompt Creator with enhance, create_from_image, and create_from_theme modes, reusing the same multimodal Krea2/Qwen3-VL CLIP as Generate Text.
- Chainable Krea2 CcC Attention Region nodes with unique tags and percentage-based target boxes.
- Visual Reference regional attention scopes: global, boost in region, and only in region.
- Target-side region transport through Latent geometry, strict crop guardrails, and token-grid projection for regional attention.
- Regional attention example workflow with two tagged subjects.
- Krea2 CcC Paint Restore for exact depadding or generated-crop compositing.

### Changed

- Prompt Creator preset system_prompt widgets now use ComfyUI's read_only option instead of disabling the widget, so preset text remains visible/greyed; thinking mode now prefills the Qwen3-VL assistant turn with <think> and separates reasoning at </think>.
- Edit Prompt Creator now exposes system_prompt as the effective preset/custom prompt: preset modes display it read-only in the UI, custom enables editing, create_from_image requests substantially richer scene detail, and thinking output is reported separately without suppressing reasoning in the system prompt.
- Edit Prompt Creator adds a custom mode, optional Qwen thinking with a separate thinking output, stricter final-prompt-only system contracts, and Image N reference naming; leaked meta-instructions are filtered/rejected before reaching Edit.
- Edit Prompt Creator now exposes and passes an explicit sampling seed to Qwen3-VL generation, matching ComfyUI Generate Text sampling requirements.
- Merged the former public Paint Geometry stage into Krea2 CcC Paint Prepare. Paint Prepare now owns native image/mask geometry, outpaint expansion, pad/crop normalization, mask processing, semantic-reference preparation, VAE preparation, and returns paint_geometry for Restore.
- Removed the four Reference Cache nodes from the public ComfyUI node registry, workflows, and current documentation. Normal Visual Reference is the single public appearance-reference path.
- Updated AnyPaint example workflow to the current Paint Prepare -> Paint -> decode -> Paint Restore path.
- Character Sheet now uses generic portrait/body inputs, nine explicit layouts, fixed no-crop fit behavior with selectable resize interpolation, and the same curated Krea geometry presets as Latent.
- Explicit outpaint margins keep their selected padding fill as semantic/reference context while remaining fully generable.
- Paint runtime consumes the appearance latent prepared by Paint Prepare and no longer creates the target latent.
- Added Krea geometry policies to Latent for fixed and image-derived dimensions: nearest curated Krea aspect or aspect-preserving Krea bounds.
- Replaced Latent image-fit behavior with explicit content fit: crop, contain with white padding, or stretch.
- Replaced megapixel-derived Edit presets with explicit curated Krea target sizes.
- Standardized semantic grounding controls on a 32-pixel cadence.
- Consolidated Edit around one Krea2 CcC runtime.
- Removed runtime delegation to other installed custom nodes and obsolete public Subject/Scene/Outfit/Style/reference node implementations.
- Split public editing into Visual Reference, Semantic Reference, Size Resolver, Latent, optional Edit Prompt Creator, and Edit.
- Visual Reference keeps Qwen preparation independent from VAE geometry and exposes crop, resize, contain, and native appearance modes plus independent RoPE placement.
- Positive conditioning uses configured per-reference boosts; grounded negative uses neutral boosts and fixed empty negative text.
- Semantic-only references remain Qwen-only and do not add VAE appearance latents.
- Size Resolver outputs only width and height.
- Latent owns final target dimensions, /16 alignment, optional target content, optional target semantic metadata, and transformed Attention Regions.
- Documentation and checked-in workflows now track the current 14-node public registry.
