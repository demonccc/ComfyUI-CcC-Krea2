# Krea2 CcC — End-to-end showcase workflows

These are **integrated editing and generation use cases**, rather than individual-node tests. They are derived from the supplied complex Scene + Subject + Prompt Creator + Mask and Layout + SAM + Paint + Restore workflows. The older workflows in the parent directory remain available as technical examples.

| File | Goal | Integrated capabilities |
| --- | --- | --- |
| `01_identity_scene_transfer.json` | Re-create the scene with a different person's identity | Two References, SAM masks, region-specific Boost, Prompt Creator, Edit, sampling |
| `02_outfit_transfer.json` | Transfer clothing from a third photo while retaining a person's face and the scene's pose | Three ordered References, two SAM masks, Prompt Creator, Edit |
| `03_character_consistency.json` | Put a recognizable subject into a different action and environment | Identity Reference, ordered body/face SAM masks, Prompt Creator, Edit |
| `04_scene_composition.json` | Compose an environmental moodboard into one coherent new photograph | Three Layout Cells, Layout Composer, identity Reference, Prompt Creator, Edit |
| `05_local_image_editing.json` | Keep a masked person while replacing the background with a studio canvas | Layout Cell + Composer, two SAM masks, InvertMask, Paint Prepare, Paint, Restore |
| `06_outpainting_reframing.json` | Extend an existing photograph beyond its borders | Paint Prepare with asymmetric expansion, Paint, Restore |
| `07_creative_generation.json` | Create a new image from visual inspiration | Vision Reference, Prompt Creator (T2I), optional LoRA Stack, Text To Image |

## How to try them

1. Open the JSON in ComfyUI and replace **every LoadImage placeholder** with an image from your own inputs. The image assets are **not included** in the repository.
2. Verify the selected UNET, CLIP/Qwen, VAE and LoRA filenames match your installations. Image editing examples use Krea2 Identity Edit v2; Paint workflows require AnyPaint. The SAM examples use `sam3.1_multiplex_fp16.safetensors`; some workflows also use the third-party `BlackwellAttentionFix` node.
3. Keep the fixed sampler and Prompt Creator seeds when comparing settings. Check the generated prompt, Thinking, SAM selections, layout previews and final output in order.
4. Adjust the new **signed Boost** scale: `-5` blocks direct VAE/RoPE reference attention, `0` is neutral, and `+5` is 10×. Intermediate values use `10^(boost/5)`, in increments of `0.1`.
5. Change the reference images, segmentation prompts, composition and editing instruction to match your task. These are **starting configurations**, not verified output-quality presets.

## Important operational details

- The **Mask** node's `image` output retains the source image. `mask_attention` conveys region-specific reference attention. `composite_image` is a visibly altered IMAGE intended for inspection or deliberate artistic compositing: it should **not** replace the unmodified reference image by accident.
- Masks on the same image can be chained using `previous_mask_attention`. **Last Mask Wins** for overlapping regional Boost values. When masks use **different image sources**, they cannot share a MaskChain: each belongs to its matching Reference.
- Masks feeding **Paint** select pixels to regenerate and do not apply VAE/RoPE Boost. The background-replacement workflow inverts the subject mask so Paint edits the background, not the person.
- In workflows 01 and 02, SAM's targeted unload is left disabled because two masks on independent image branches may execute in either order. In workflow 03 and 05, SAM is unloaded in the last **sequentially chained** mask.
- The Layout Composer in workflow 05 intentionally establishes the output canvas. Paint Restore returns to **that composed canvas**, not necessarily the source photo's original dimensions.
- Workflows 01–04 combine Qwen Vision analysis with VAE/RoPE references. Workflow 07 uses a Vision-only reference, creates a text prompt, then generates a new image with T2I.
- Filename placeholders are deliberate; this repository does not distribute the user's uploaded photographs.

## Validation status

The JSON graphs have been checked for valid link endpoints, input/output consistency, reference and mask controls, and the current input order of the main custom nodes. Actual image generation with model weights, Qwen, SAM and ComfyUI remains to be tested on a GPU installation.
