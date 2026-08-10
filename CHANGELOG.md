# Changelog

All notable changes to the `ComfyUI-CcC-Krea2` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Surgical stabilization pass for 5-layer modular reference architecture (`feat/modular-reference-pipeline-refactor`).
- Mandatory Qwen visual row processing via `comfy.text_encoders.qwen_vl.process_qwen2vl_images` in production with clear `RuntimeError` failure handling on missing processors or GPU execution failures.
- Strict half-open vision span validation `(start >= 0, end > start, end <= sequence_length, no overlap, physical order)` preventing invalid or out-of-order token stream spans.
- Exact target pixel dimensions for `crop_and_resize` near-match VAE reference geometry.
- Comprehensive per-Style diagnostic metrics in `edit_info` output (actual spans, total rows, rows removed, direct/indirect transfer status).
- `CcC Krea2 - Text to Image` node (`CcCKrea2TextToImage`) for native text-to-image generation using CLIP tokenization, scheduled text encoding, and Empty SD3 Latent generation.
- `CcC Krea2 - LoRA Stack` node (`CcCKrea2LoRAStack`) for model-only LoRA stacking (up to 4 slots) with global/slot strengths, persistent slot loader instances, and prompt augmentation accumulators.
- `CcC Krea2 - LoRA Prompt Settings` node (`CcCKrea2LoRAPromptSettings`) for slot-level positive and negative prompt fragment configuration with `prepend` and `append` positioning.
- Immutable prompt augmentation module (`ccc_krea2/prompt_augmentation.py`) supporting deterministic text merging via double newline `\n\n` separators without mutating input strings.
- Tested Upstream Parity wording updated in ARCHITECTURE.md reflecting exact commits in NOTICE.
- Exact Auto Visual Reference Fit outcomes documented (`exact`, `crop_only`, `crop_and_resize`, `fit`).
- Corrected Indirect Style explanation clarifying total indirect Style visual row removal via keep-mask without mandatory retained Style spans.
- Replaced workflow strategy table with five-column matrix (`Workflow filename`, `Target Content`, `Target Geometry`, `References`, `Main objective`).
- Documented README Favor Subject connection requirement (prepared image to Target Latent `subject_image`).
- Added integrated Qwen token-to-span mapping test (`test_integrated_qwen_token_to_span_mapping`).
- Strengthened workflow contract tests to enforce required non-widget linked socket presence and declared vs serialized type parity.
- Corrected obsolete `crop_and_resize` `/16` dimension comment in `ccc_krea2/krea2edit_geometry.py`.


### Fixed
- Fixed Target Latent VAE output normalization to accept native 5D tensors ([B, C, T, H, W]) alongside 4D tensors ([B, C, H, W]) with dimension-agnostic batch expansion.
- Fixed ComfyUI custom node package import error by converting internal ccc_krea2 absolute imports to package-relative imports.
- Aligned internal dataclass defaults (`masked_identity_anchor = 0.0` in `SubjectReferenceSpec` and `masked_region_anchor = 0.0` in `SceneReferenceSpec`) to match node UI defaults.
- Corrected `edit_info` anchor reporting to accurately output `Anchor Implementation Type: Vision directive only` and list active directive-only controls for Subject, Scene, and Outfit roles.
- Canonicalized modular workflow collection (`01_t2i_basic.json` through `12_full_pipeline_composition.json`) using current socket types (`PREPARED_VISION_IMAGE`, `REFERENCE_CHAIN`, `LATENT`, `CCC_KREA2_PROMPT_AUGMENTATION`) and input names (`reference_chain`, `visual_fit_mode`).
- Reorganized superseded workflow files into `workflows/legacy/` with explanatory migration README.
- Synchronized `NODES.md`, `ARCHITECTURE.md`, `README.md`, and `CHANGELOG.md` with current Python node interface definitions and canonical workflow contracts.

## [0.1.0-alpha] - 2026-07-30

### Added
- Experimental Alpha release of `ComfyUI-CcC-Krea2` suite.
- 4 general editing nodes (`Subject`, `Subject + Outfit`, `Subject + Scene`, `Subject + Scene + Outfit`).
- 3 inpainting nodes (`Inpaint`, `Inpaint Subject + Outfit`, `Inpaint Subject + Scene`).
- 2 advanced settings nodes (`Image Advanced Settings`, `Edit Advanced Settings`).
- `role_resolution_limit_mode` and `role_resolution_max_megapixels` controls in `Edit Advanced Settings` for clamping high-resolution role images while preserving aspect ratio.
- 6 curated example workflows in `workflows/` covering single-, dual-, and triple-reference editing, as well as Qwen-VL Vision Language Model integrations.
- Standard ComfyUI `DIFFUSION_MODEL` wrapper signature `(executor, x, timesteps, context, *wargs, **kwargs)` with closure transport.
- Applied `model.model.process_latent_in(...)` to every reference VAE latent.
- Native pixel-path reference geometry (`reference_fit_mode="fit"`) aligned to multiples of 16 without black canvas padding.
- Explicit inpaint base role selection per node.
- Pure PyTorch mask dilation via `F.max_pool2d` removing `scipy` dependency.
- Model-driven empty latent generation honoring `batch_size`.
- Mandatory VAE input across all custom node definitions.
- Unit test suite in `tests/` covering mock integration, geometry, masks, dynamic prompt templates, resize method wiring, role resolution clamping, workflow files, and patch isolation.
