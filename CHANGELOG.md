# Changelog

## Unreleased

### Duplicate implementation consolidation

- Consolidated Edit Prompt Creator's optional-reference contract into its single implementation; its former public wrapper is now an import-only compatibility alias.
- Kept its ComfyUI node identifier, inputs, outputs and legacy Python import path stable. No workflow JSON files were changed.


### Architecture cleanup

- Removed unused role-based Subject/Scene target latent generation and inactive target-vision metadata. The active Latent node owns target geometry and VAE output.
- Removed legacy Qwen prompt/template/conditioning adapters and fixed-role constants, while preserving the current Qwen3-VL Edit encoder and Vision guidance paths.
- Removed obsolete pixel-fit and sampling transformations; `geometry.resize_tensor` remains the shared pixel resampler and `krea2edit_geometry` owns Visual Reference preparation.
- Removed ignored internal resize controls and corrected diagnostics to reflect fixed bicubic behavior. Active semantic image downscaling no longer depends on a legacy grounding preset module.
- Updated tests to exercise current geometry and target tensor behavior rather than retired helper APIs.
- Restored the previously requested Prompt Creator max token ceiling to 32768 (32-token increments), with generation-path coverage.
- No files under `workflows/` were changed. `03_regional_attention.json` is a legacy example requiring migration.

### Attention masks and simplified Edit

- Visual Reference owns two optional masks: `reference_boost_mask` on the reference image and `target_attention_mask` on the final target; masks remain aligned to their own reference entries.
- Latent adds `target_image` output (post-fit canvas or neutral empty canvas) for aligned mask painting.
- Mask attention modes are `global`, `boost in mask`, and `only in mask`; reference boosts operate on selected source tokens and target scopes operate on selected target queries.
- Removed the tagged Attention Region node, rectangle-percent editor, tag matching, unused legacy mask branches and obsolete post-VAE latent resize.
- Existing example workflows were intentionally not modified in this change.

### Dead-code and documentation audit

- Removed legacy subject/scene/megapixel target latent construction; current Latent is the single owner of target geometry.
- Removed disabled target-vision-context forwarding, unused Qwen grounding preset APIs and duplicate fit/sampling geometry helpers.
- Removed obsolete Qwen encoding/role helpers, outdated constants and their retired tests; retained live reference, VAE, Qwen and Paint paths.
- Corrected documentation against the current 14 registered nodes. Workflow `03_regional_attention.json` is legacy/incompatible and was intentionally left untouched.

### Reference-chain and geometry cleanup

- Unified Visual and Vision references under the `KREA2_REFERENCE_CHAIN` socket, with `references` inputs/outputs across Edit, Prompt Creator, and T2I. `previous_references` accepts mixed chains.
- Prompt Creator analyzes Visual entries only and passes all mixed-chain entries through unchanged; T2I consumes Vision entries only.
- Visual Reference modes are now `native`, `crop`, `contain`, and `fit` (proportional cover + center crop). Image resizes use bicubic with antialiasing.
- Removed reference-cache implementation and unused cache-specific tests, controls, and runtime paths. No workflow JSON changes in this documentation/cleanup pass.


### Added

- Added `workflows/06_text_to_image.json`, a complete native T2I example wiring Krea2 CcC Text To Image into the Krea2 sampling/decode/output path.
- Optional Krea2 CcC Edit Prompt Creator with enhance, create_from_image, and create_from_theme modes, reusing the same multimodal Krea2/Qwen3-VL CLIP as Generate Text.
- Krea2 CcC Paint Restore for exact depadding or generated-crop compositing.

### Changed

- Prompt Creator preset system_prompt widgets now use ComfyUI's read_only option instead of disabling the widget, so preset text remains visible/greyed; thinking mode now prefills the Qwen3-VL assistant turn with <think> and separates reasoning at </think>.
- Edit Prompt Creator now exposes system_prompt as the effective preset/custom prompt: preset modes display it read-only in the UI, custom enables editing, create_from_image requests substantially richer scene detail, and thinking output is reported separately without suppressing reasoning in the system prompt.
- Edit Prompt Creator adds a custom mode, optional Qwen thinking with a separate thinking output, stricter final-prompt-only system contracts, and Image N reference naming; leaked meta-instructions are filtered/rejected before reaching Edit.
- Edit Prompt Creator now exposes and passes an explicit sampling seed to Qwen3-VL generation, matching ComfyUI Generate Text sampling requirements.
- Merged the former public Paint Geometry stage into Krea2 CcC Paint Prepare. Paint Prepare now owns native image/mask geometry, outpaint expansion, pad/crop normalization, mask processing, semantic-reference preparation, VAE preparation, and returns paint_geometry for Restore.
- Removed the unused Reference Cache feature completely; normal Visual Reference is the appearance-reference path.
- Updated AnyPaint example workflow to the current Paint Prepare -> Paint -> decode -> Paint Restore path.
- Character Sheet now uses generic portrait/body inputs, nine explicit layouts, fixed no-crop fit behavior using bicubic antialiasing, and the same curated Krea geometry presets as Latent.
- Explicit outpaint margins keep their selected padding fill as semantic/reference context while remaining fully generable.
- Paint runtime consumes the appearance latent prepared by Paint Prepare and no longer creates the target latent.
- Added Krea geometry policies to Latent for fixed and image-derived dimensions: nearest curated Krea aspect or aspect-preserving Krea bounds.
- Replaced Latent image-fit behavior with explicit content fit: crop, contain with white padding, or stretch.
- Replaced megapixel-derived Edit presets with explicit curated Krea target sizes.
- Standardized semantic grounding controls on a 32-pixel cadence.
- Consolidated Edit around one Krea2 CcC runtime.
- Removed runtime delegation to other installed custom nodes and obsolete public Subject/Scene/Outfit/Style/reference node implementations.
- Split public editing into Visual Reference, Vision Reference, Size Resolver, Latent, optional Edit Prompt Creator, and Edit.
- Visual Reference keeps Qwen preparation independent from VAE geometry and exposes native, crop, contain, and fit appearance modes plus independent RoPE placement.
- Positive conditioning uses configured per-reference boosts; grounded negative uses neutral boosts and fixed empty negative text.
- Semantic-only references remain Qwen-only and do not add VAE appearance latents.
- Size Resolver outputs only width and height.
- Latent owns final target dimensions, /16 alignment, optional target content, optional target semantic metadata, and an aligned target_image mask canvas.
- Documentation describes the current 14-node public registry. Workflow JSONs are intentionally unchanged and some examples still use removed nodes.
