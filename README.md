# ComfyUI-CcC-Krea2

[![CI](https://github.com/demonccc/ComfyUI-CcC-Krea2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/demonccc/ComfyUI-CcC-Krea2/actions/workflows/ci.yml)

I created CcC Krea2 after needing a simpler and more consistent way to perform advanced Krea 2 edits in ComfyUI.

Existing tools made different parts of the workflow possible, but combining subject identity, outfits, scenes, attention control, latent selection and localized inpainting required large and difficult-to-maintain workflows.

CcC Krea2 groups those editing patterns into a small set of purpose-built nodes. Each node prepares the references, applies the model patch, builds the positive and negative conditioning, and returns the latent that should be sent to KSampler.

## Features

- Subject identity editing.
- Outfit and scene references.
- Subject, outfit and scene combinations.
- Localized inpainting.
- Qwen3-VL grounding controls.
- Per-reference attention boosts and masks.
- Empty, subject-based or scene-based sampling latents.
- A model patch applied only to the MODEL returned by the node.
- No global modification of ComfyUI.

## Edit LoRA compatibility

CcC Krea2 requires a Krea 2 edit LoRA trained to work with the in-context editing model patch introduced by ComfyUI-Krea2Edit. A regular Krea 2 text-to-image LoRA does not provide the required editing behavior.

The nodes work particularly well with the Identity Edit LoRA, including Identity Edit v1.2, which is the recommended starting point for subject and subject + scene workflows.

How each reference is interpreted depends on the LoRA training, reference order and prompt. The nodes expose subject, outfit and scene roles, but the loaded edit LoRA ultimately determines how strongly those roles are understood.

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/demonccc/ComfyUI-CcC-Krea2.git
   ```
2. Restart ComfyUI.

## Nodes

See [NODES.md](NODES.md) for the complete node reference, input descriptions, latent modes, grounding controls, attention masks and inpainting behavior.

## Credits

CcC Krea2 builds upon technical work and concepts from:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-Krea2Edit](https://github.com/lbouaraba/ComfyUI-Krea2Edit) by lbouaraba: model patch, dual conditioning and Krea 2 identity-edit architecture.
- [ComfyUI-Krea2Moodboard](https://github.com/RedNodeAI/ComfyUI-Krea2Moodboard) by RedNodeAI: high-level workflow UX, reference-role organization and matched grounding.
- [ComfyUI-EditUtils](https://github.com/lrzjason/ComfyUI-EditUtils) by lrzjason: flexible image/reference preparation and latent workflow concepts.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms.

*Disclaimer: This project is an independent community package and is not affiliated with or endorsed by Krea.ai, Qwen, or the authors of the referenced models and LoRAs.*
