# CcC Krea2 Technical Architecture and Workflow Strategy

This document provides the canonical technical architecture and workflow strategy for `ComfyUI-CcC-Krea2`. It details the 5-layer modular pipeline, image representation models, Target Latent combinations, reference order resolution, directive-only anchor parity, and practical workflow strategies for production deployment.

---

## 1. Goals and Design Principles

1. **Strict Upstream Parity**: Reproduce upstream `ComfyUI-Krea2Edit` (commit `5f8a02c`) and `ComfyUI-Krea2Moodboard` (commit `a7d83f1`) algorithms, mathematical definitions, and pixel-space geometry down to exact pixel target division, crop rounding, and tile shuffle orders.
2. **Modular 5-Layer Pipeline**: Decouple image prep, target latent creation, declarative reference definitions, immutable chain ordering, and edit orchestration into dedicated, composable nodes.
3. **Triple Image Representations**: Explicitly separate raw source images, Qwen Vision semantic images (`vision_image`), and spatial VAE reference latents (`vae_reference_latent`).
4. **Directive-Only Anchors Parity**: Text anchors modulate Qwen Vision prompt directives only, without modifying attention hooks, VAE latents, or RoPE geometry:
   - **Subject**: Pose Anchor, Outfit Anchor, Masked Identity Anchor
   - **Scene**: Scene Anchor, Masked Region Anchor
   - **Outfit**: Outfit Anchor
5. **Deterministic Reference Ordering**: Single-source slot resolution ensuring all non-Style references precede Style references in physical token streams and VAE spatial frames.

---

## 2. Five-Layer Modular Architecture

```mermaid
graph TD
    L1["Layer 1: Qwen Vision Image Prep<br/>(CcCKrea2QwenVisionImagePrep)"] -->|Prepared Vision Image| L3["Layer 3: Declarative Reference Nodes<br/>(Subject, Scene, Outfit, Style)"]
    L2["Layer 2: Target Latent<br/>(CcCKrea2TargetLatent)"] -->|Target Latent| L5["Layer 5: Edit Orchestrator<br/>(CcCKrea2Edit)"]
    L3 -->|Reference Spec| L4["Layer 4: Reference Chain<br/>(Immutable Specs Chain)"]
    L4 -->|Ordered Reference Chain| L5
    L5 -->|Output| OUT["Patched Model, Positive, Negative, Latent, Edit Info"]
```

- **Layer 1 (Qwen Vision Image Prep)**: Resizes raw image into a lightweight Qwen-optimized derivative (`vision_image`) aligned to patch boundaries (`32x32`), while retaining the untouched raw source image (`original_image`) intact for VAE processing.
- **Layer 2 (Target Latent)**: Resolves the target latent tensor (`[B, 16, H//8, W//8]`), independently configuring Latent Content Source (`empty`, `subject`, `scene`) and Target Geometry (`favor_subject`, `favor_scene`, `fixed`).
- **Layer 3 (Declarative References)**: Defines per-reference specs (`Subject`, `Scene`, `Outfit`, `Style`), configuring visual fit modes (`auto`, `fit`, `crop`), attention boosts, masks, and style processing mode.
- **Layer 4 (Reference Chain)**: Maintains an immutable linked chain of reference specifications (`REFERENCE_CHAIN`). Reference definition nodes accept an optional `reference_chain` input and produce an updated `reference_chain` output.
- **Layer 5 (Edit Orchestrator)**: Consumes the resolved reference chain via its `references` input, resolves physical Qwen indices and VAE frame numbers, executes Qwen text/vision encoding, applies Moodboard statistical transforms, patches diffusion model attention hooks, and formats `edit_info`.

---

## 3. The Three Image Representations

In this architecture, an input image exists in up to three distinct representations:

1. **Original Image**: The unmodified raw pixel tensor (`[B, H, W, C]`) provided by the user. Preserved intact for VAE reference crop/fit processing in Layer 5.
2. **Vision Image**: The semantic derivative prepared by Layer 1 (`CcCKrea2QwenVisionImagePrep`) aligned to Qwen patch factor (`16 * 2 = 32`). Used exclusively for Qwen3-VL tokenization and visual context.
3. **VAE Reference Latent**: The fitted pixel image cropped, scaled, and encoded via VAE into spatial reference latents (`[B, 16, H//8, W//8]`). Used for spatial cross-attention patching in the diffusion model.

> [!NOTE]
> `CcCKrea2QwenVisionImagePrep` does **not** perform grounding itself. Grounding occurs when Qwen processes prepared images, aliases, automatic role directives, Extra Vision Directives, Global Vision Directives, and positive or negative prompts during orchestration in Layer 5.

---

## 4. CLIP Loader & Independent Semantic Image Sizing

- **CLIP Loader**: The Qwen3-VL text encoder must be loaded using ComfyUI's native **CLIPLoader** node configured with the `krea2` model type.
- **Semantic Sizing**: Semantic vision image sizing (`vision_image`) is strictly independent of VAE latent output sizing (`vae_reference_latent`):
  - **Native Mode**: Computes exact Qwen2-VL / Qwen3-VL native aspect-ratio grid geometry adhering to `min_pixels` (3,136) and `max_pixels` (12,845,056).
  - **Adaptive Mode**: Resizes within user-configured minimum (`min_mp`) and maximum (`max_mp`) megapixels while retaining source aspect ratio.
  - **Fixed Mode**: Standardizes vision inputs to a specific fixed megapixel resolution (`fixed_mp`) regardless of source dimensions.

> [!IMPORTANT]
> Semantic image size and output image size are completely decoupled. A higher semantic megapixel setting may improve small-detail Qwen understanding, but it does **not** automatically improve VAE identity preservation or increase VAE latent output resolution.

---

## 5. Target Latent: Content versus Geometry

`CcCKrea2TargetLatent` separates **Target Latent Content** from **Target Geometry**:

- **Target Latent Content**: `empty` (`Empty`, zeros tensor), `subject` (`Subject`, VAE-encoded Subject image), or `scene` (`Scene`, VAE-encoded Scene image).
- **Target Geometry**: `favor_subject` (`Favor Subject`, dimensions derived from Subject reference), `favor_scene` (`Favor Scene`, dimensions derived from Scene reference), or `fixed` (`Fixed`, dimensions derived from fixed megapixel / aspect ratio selection).

Inputs: `target_content`, `geometry_mode`, `target_megapixels`, `fixed_megapixels`, `aspect_ratio`, `batch_size`. Optional inputs: `vae`, `subject_image`, `scene_image`. VAE, Subject image, and Scene image are required at runtime based on the selected content and geometry options.

Output: `target_latent` (`LATENT`).

---

## 6. Complete Target Latent Combination Matrix

The 9 primary Target Latent combinations across Content (`empty`/`Empty`, `subject`/`Subject`, `scene`/`Scene`) and Geometry (`favor_subject`/`Favor Subject`, `favor_scene`/`Favor Scene`, `fixed`/`Fixed`):

| Combination | Content | Geometry | Main Objective | Expected Tendency | Trade-offs | Typical Use Case |
|---|---|---|---|---|---|---|
| 1 | `empty` (`Empty`) | `favor_subject` (`Favor Subject`) | Free generation from noise in Subject aspect | Flexible pose/environment; uses Subject aspect ratio | Requires identity attention guidance without initial Subject pixels | Character generation in new poses or environments |
| 2 | `empty` (`Empty`) | `favor_scene` (`Favor Scene`) | Free generation from noise in Scene aspect | Flexible composition in Scene dimensions | No direct pixel initialization from Scene | Character placement into fresh Scene-proportioned layout |
| 3 | `empty` (`Empty`) | `fixed` (`Fixed`) | Free generation at fixed dimensions | Exact target width/height (e.g. 16:9 banner) | Subject/Scene aspect ratio must be adapted to target | Standard T2I banner/wallpaper generation |
| 4 | `subject` (`Subject`) | `favor_subject` (`Favor Subject`) | Max identity continuity & subtle adjustment | Strongest pixel preservation of Subject | Limited composition changes; stays close to original Subject | Direct Subject retouching, face touchup, subtle edits |
| 5 | `subject` (`Subject`) | `favor_scene` (`Favor Scene`) | Subject pixel edit inside Scene aspect | Subject pixels adapted to Scene aspect ratio | Subject spatial distortion if aspect ratios differ significantly | Subject editing within wide aspect constraints |
| 6 | `subject` (`Subject`) | `fixed` (`Fixed`) | Subject pixel edit at fixed dimensions | Exact target output resolution initialized with Subject | May require crop/fit of Subject base pixels | Fixed-format delivery of Subject modifications |
| 7 | `scene` (`Scene`) | `favor_subject` (`Favor Subject`) | Scene pixel edit fitted to Subject aspect | Scene background initialized; forced into tall Subject aspect | Scene cropping on left/right edges | Restyling Scene into vertical portrait layout |
| 8 | `scene` (`Scene`) | `favor_scene` (`Favor Scene`) | Scene pixel edit in native Scene aspect | Strongest pixel preservation of Scene background | Subject must be inserted into existing Scene pixels | Subject insertion or localized scene inpainting |
| 9 | `scene` (`Scene`) | `fixed` (`Fixed`) | Scene pixel edit at fixed dimensions | Scene background initialized at exact target resolution | Scene pixels scaled/cropped to fit target MP | Fixed-format delivery of Scene edits |


---

## 7. Practical Workflow Strategies & Canonical Workflows

### 7.1 Practical Editing Strategies

- **Maximum Identity Continuity**: Target Content = `subject`, Target Geometry = `favor_subject`. Active References: Subject reference (Slot 1). Preserves fine facial and identity details.
- **Freer Generation**: Target Content = `empty`, Target Geometry = `favor_subject`. Active References: Subject reference (Slot 1). Allows new poses/environments while adhering to Subject aspect constraints.
- **Scene-Preserving Subject Replacement**: Target Content = `scene`, Target Geometry = `favor_scene`. Active References: Scene (Slot 1), Subject (Slot 2).

### 7.2 Canonical Workflows Index

The canonical modular workflow collection in `workflows/`:

| Workflow File | Description | Key Features / Strategy Covered |
|---|---|---|
| `workflows/01_t2i_basic.json` | T2I Basic Generation | empty + fixed Target Latent, native text-to-image pipeline |
| `workflows/02_t2i_lora_stack.json` | T2I + LoRA Stack | empty + fixed Target Latent with 4-slot LoRA Stack |
| `workflows/03_subject_edit.json` | Single Subject Edit | subject + favor_subject (Maximum Identity Continuity) |
| `workflows/04_subject_scene_edit.json` | Subject + Scene Edit | scene + favor_scene (Scene-Preserving Subject Replacement) |
| `workflows/05_subject_outfit_edit.json` | Subject + Outfit Edit | subject + outfit reference competition and outfit transfer |
| `workflows/06_subject_scene_outfit_edit.json` | Subject + Scene + Outfit | Triple-reference editing pipeline |
| `workflows/07_style_moodboard_edit.json` | Style / Moodboard Transfer | subject + style 2x2 indirect moodboard statistical style transfer |
| `workflows/08_inpaint_subject_edit.json` | Subject Inpainting | Inpaint-masked Subject identity editing |
| `workflows/09_inpaint_scene_edit.json` | Scene Inpainting | Inpaint-masked Scene composition insertion |
| `workflows/10_multi_subject_chasing_slots.json` | Multi-Subject Chasing Slots | Multiple Subject references with sequential logical slot resolution |
| `workflows/11_advanced_directives_fit_modes.json` | Directives & Fit Modes | Custom Extra Vision Directives and Auto/Fit/Crop fit modes |
| `workflows/12_full_pipeline_composition.json` | Full Pipeline Composition | Subject + Scene + Outfit + Style + LoRA Stack composite |

Legacy workflows targeting earlier node interfaces are located in `workflows/legacy/`.

---

## 8. Visual Reference Fit Options

- **`auto`**: Evaluates source vs target dimensions:
  - Selects `fit` when aspect ratio matches within tolerance (0.01) AND target dimensions match exactly.
  - Selects `crop_and_resize` when aspect ratio matches but dimensions differ, scaling to cover and center cropping to exact target dimensions.
  - Selects `crop_only` when aspect ratio matches and no resize is needed. `crop_only` is Auto-only.
- **`fit`**: Preserves original reference geometry using `/16`-aligned aspect ratio fitting. Target VAE input dimensions are floored to exact pixel targets divisible by 8/16.
- **`crop`**: Preserves aspect ratio by scaling to cover, then center-cropping to fill exact target dimensions.

> [!NOTE]
> Pixel-space reference fitting occurs **only in Layer 5 (Edit Orchestrator)** when target latent dimensions are known. Reference definition nodes (Layer 3) declare the requested fit mode.

---

## 9. Reference Roles and Single-Source Ordering

The single source of truth for reference ordering is `reference_slots.py`. All non-Style references (**Subject**, **Scene**, **Outfit**) precede all **Style** references.

1. **Non-Style References (Subject, Scene, Outfit)**: Map 1:1 to physical Qwen images and assign sequential 1-based VAE Reference Frames (`Frame 1`, `Frame 2`, etc.).
2. **Style References**: Expand to 1 (full), 4 (2x2), or 16 (4x4) physical Qwen vision images. Style references receive VAE Reference Frame = `none` and do not participate in spatial VAE latent attention patching or negative conditioning.

---

## 10. Moodboard Style Processing

1. **Statistical Style Fidelity**:
   Transforms Style vision conditioning tokens $z_{\text{orig}}$ via:
   $$z_{\text{out}} = \text{fidelity} \cdot z_{\text{orig}} + (1.0 - \text{fidelity}) \cdot z_{\text{target}}$$
   where $z_{\text{target}}$ is constructed by cycling mean $\mu$, $\mu + \sigma$, and $\mu - \sigma$ computed across visual rows.
2. **Indirect Style Transfer**:
   Designated indirect style vision rows are removed in **one single operation** using a consolidated boolean keep-mask. Associated `attention_mask` metadata is sliced `[..., keep]` in lockstep to ensure sequence length alignment. At least one visual span is preserved to ensure total visual row count remains greater than 0 for Qwen token stream stability.

---

## 11. Positive and Negative Conditioning

- **Positive Qwen Context**: Includes Subject, Scene, Outfit, expanded Style images/directives, Global Vision Directive, positive prompt, and prompt augmentation.
- **Negative Qwen Context**: Includes Subject, Scene, Outfit, Global Vision Directive, negative prompt, and negative prompt augmentation. **Strictly excludes all Style images, crops, tiles, Style automatic directives, and Moodboard operations.**

---

## 12. Upstream Parity and Attribution

- Ported from and attributed to `ComfyUI-Krea2Edit` by lbouaraba (Apache-2.0 / GPL-3.0) and `ComfyUI-Krea2Moodboard` by RedNodeAI (GPL-3.0).
