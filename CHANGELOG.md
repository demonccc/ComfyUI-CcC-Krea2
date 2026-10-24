# Changelog

## Unreleased

- Simplified Prompt Creator to a single `instructions` textbox; removed editable/preview `system_prompt` and its frontend graying logic. Six T2I/I2I variants now share three preset templates and a `{kind}` placeholder. Added `none` as the no-preset option (shared Krea2 guidelines still apply), and added the effective assembled Qwen prompt to `creator_info`. Updated docs and tests; existing saved workflow widgets/modes require migration, but workflow files were not edited.

- Split Prompt Creator `create_from_image` into `create_from_image_t2i` and `create_from_image_i2i`, both analyzing optional image references and using the user's prompt to determine which details matter and how to use them. Both compose preset, user prompt, and common Krea2 guidelines; other presets and workflows unchanged. Existing workflows selecting `create_from_image` will need their mode updated.

- Split Prompt Creator `create_from_theme` into `create_from_theme_t2i` and `create_from_theme_i2i`, with the approved Krea2 preset instructions and the common preset → user prompt → guidelines structure. Both T2I and I2I accept zero or more optional images, without assigning creative roles; T2I can use images as inspiration. Updated frontend preset previews and regression tests. Existing saved workflows selecting the old mode must select a new theme mode (workflows not modified).

- Prompt Creator `enhance` now uses the agreed one-line task instruction and assembles `preset → USER PROMPT → KREA2 PROMPT GUIDELINES` in the Qwen generation message. Enhance does not analyze reference images, but forwards the reference chain unchanged. Other presets and their system prompts are unchanged. Added regression tests (not executed in ComfyUI).

- Migrated workflow 05 to the new Reference-chain `reference_edit_image` socket and corrected the order and passthrough to Edit.
- Added workflow 19: Scene + Subject + Prompt Creator. The generated prompt enhances the edit while both reference conditionings remain connected.
- Structurally validated all 19 workflow JSON graphs for link, input/output type, and endpoint consistency. No ComfyUI execution or image-generation quality tests were run.


### Reference image transport and Prompt Creator

- Public Reference VAE/RoPE controls now use the `vae_rope_` prefix instead of `visual_`, with `vae_rope_enable` replacing `visual_enable`.
- Source images remain available even when both VAE/RoPE and Vision are disabled; a raw image-only chain entry carries the source without conditioning.
- Vision references retain the original source pixels independently of Qwen resizing.
- Prompt Creator accepts an Edit Reference chain on `reference_edit_image`, analyzes Edit images before other references, removes paired Visual/Vision duplicates from its analysis image list, and forwards the combined chain to Edit without changing enabled processing paths.
- Existing workflows using the previous `IMAGE` socket on Prompt Creator or old Reference widget names must be migrated. Workflow JSON files are not changed in this patch.
- Added reference transport and Prompt Creator routing tests; runtime/GPU validation remains pending.


### Final Native/Edit contract audit

- Verified the Native Vision mode routes through Edit as unchanged Qwen image rows with no preset-specific directive; Visual still has explicit appearance guidance.
- Removed a silent Qwen tokenizer fallback that could ignore the Krea2Edit-specific grounded template. Incompatible encoders now report an actionable error rather than silently changing grounding behavior.
- Live ComfyUI/GPU validation and comprehensive automated test execution are still pending.


### Native Krea2Edit Vision preset

- Added **Native** to the unified Vision mode options. For Edit it forwards Qwen vision embeddings without statistical Semantic/Style extraction or any forced preset-specific instruction, following original Krea2Edit grounding semantics while preserving user-provided instructions.
- Existing Visual remains direct embeddings plus an explicit visual-guidance instruction, so it is not identical to Native.
- T2I uses an ordinary visual description for Native, because T2I consumes Qwen-derived text rather than injecting Qwen vision rows as Edit does.


### Upstream Krea2Edit V2 parity audit

- Compared our Edit implementation with `lbouaraba/comfyui-krea2edit` V2 reference node and grounding encoder.
- The attempted upstream `fit` replacement was reverted: reference framing modes remain project-defined.
- Edit Qwen encoding now uses the upstream training-grounded system template about image color, shape, texture and spatial relationships, rather than the generic Krea2 template.
- Kept `contain`, `crop` and `native` as distinct public geometry choices; did not change RoPE or patch behavior without validated parity evidence.
- Added grounding-template tests; project geometry remains unchanged. Actual V1.2/V2 GPU image generation and full pytest have not been executed.


### Vision Style resize fidelity

- Fixed pre-tile downscaling for unified Vision Style 2x2/4x4: preserve the original reference until tiled, then bicubic-antialias resize each crop independently if it exceeds vision_max_edge. Both Edit and T2I use the same policy.
- Added regression test for 1536x1536 -> four 768x768 Qwen crops, retaining more fine texture information than whole-image downscale-first.
- Live ComfyUI image generation, complete pytest suite and V1.2/V2 comparative quality checks have **not** been run in this environment.


### Further reference-conditioning audit

- Paired Style with 2x2/4x4 crops now keeps one VAE appearance reference but contributes only 4/16 crop images to **positive** Qwen conditioning, instead of adding an unintended full-size positive image.
- The unconditional Qwen branch still receives the original appearance image, preserving the grounded negative conditioning contract.
- Deferred style crop references until after normal edit references to avoid invalid style-before-edit slot ordering when chaining multiple references.
- Added contract tests for positive/negative image blocks, single VAE frame, multiple reference ordering, and style crop slots.
- Remaining verification: Qwen/LoRA image-quality regression on live ComfyUI.


### Paired Visual/Vision conditioning correction

- Fixed unified Reference to associate the enabled Visual and Vision paths as one logical Edit reference instead of submitting the same full image twice to Qwen and shifting Image N aliases.
- Visual geometry/VAE uses original source pixels, while Qwen uses the Vision image after its optional bicubic-antialias resize and applies its selected mode policy.
- For Style with 2x2 or 4x4 tiling, a distinct style span is still required (one full appearance image plus 4/16 style crops); this is deliberate and preserves tiling.
- Added regression checks for combined and isolated modes, physical Qwen slot numbering, image geometry, and tiled Style spans. GPU-based fidelity tests remain outstanding.


### Five-slot LoRA consistency

- Limited LoRA Stack and LoRA Stack Sliders to five positions; LoRA Stack with Prompt already had five. Updated UI toggle behavior and loader regression tests.
- Verified the existing `prompt_augmentation` optional input and consumption paths in Edit, Paint, and T2I. Paint and Edit apply the positive augmentation without changing their grounded negative branch; T2I applies positive and negative augmentation.
- Existing workflow JSON files are intentionally unchanged. Saved ten-slot node configurations must be migrated manually if slots 6–10 were populated.


### Moodboard embedding correction

- Verified that Subject uses the same whitened-span blend and Style uses the same channel statistics blend as Krea2 Moodboard, rather than merely sharing descriptive prompts.
- Fixed Context: full ordered Qwen vision rows are preserved instead of whitening and subsampling 64 rows. Subsampling is not a pose/interaction extractor and discarded spatial information. Context remains constrained by its mode directive; testing is needed to determine whether direct tokens leak appearance.
- Retained Moodboard sample standard deviation for regular spans; guarded single-token spans against NaNs.
- Added formula-level unit tests and checks for Context's no-compaction contract. No workflow JSON changed; image-quality regression testing is still required.


### Moodboard-aligned Vision updates

- Reviewed RedNodeAI Krea2 Moodboard: subject whitening and style mean/standard-deviation transforms already correspond to the existing conditioning path; indirect mode is a separate token-removal strategy, not an equivalent context extractor.
- Centralized canonical Semantic/Visual/Style/Context directives in `vision_presets.py` and included them in both Edit and T2I guidance.
- Unified Reference now optionally downsizes the Vision-only copy using the shared bicubic antialiasing helper and a 768px max edge; Visual keeps the source image untouched.
- Removed misleading guidance detail and sampling presets from the unified node UI while retaining old-node compatibility.
- Kept Context's existing embedding compaction unchanged pending image-fidelity validation. Runtime tests could not be run in this environment.


### Vision preset audit

- Confirmed that Edit uses mode-specific Qwen embedding transforms (subject whitening, full direct visual rows, style statistics, and compacted context rows), while T2I uses Qwen-generated textual guidance. These are distinct consumer-specific implementations.
- Fixed custom Vision instruction propagation for Semantic, Style and Context in Edit; it was previously applied only in Visual. Added regression tests for all four modes.
- Preset consolidation and full runtime regression validation remain pending; workflow JSON files are unchanged.


### Unified Reference

- Added `Krea2 CcC Reference`, exposing Visual settings first and Vision settings second with independent enable switches.
- Both enabled paths share one source image and append distinct appearance and Qwen entries to the existing chain, avoiding duplicate Qwen semantics from the Visual path.
- Retained the two legacy reference node registrations so existing, unchanged workflow JSONs remain loadable. Added UI disabled-state logic and unified-reference contract tests.
- No workflow JSON changed; end-to-end image fidelity with the Edit LoRA remains to be validated.


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
