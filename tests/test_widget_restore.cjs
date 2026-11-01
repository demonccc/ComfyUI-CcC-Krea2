/* Regression for positional ComfyUI widget-values migration.
 * Run: node tests/test_widget_restore.cjs
 */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

let extension;
const source = fs.readFileSync(path.join(__dirname, "..", "web", "ccc_krea2.js"), "utf8")
  .replace(/^import .*$/gm, "");
vm.runInNewContext(source, {
  app: { registerExtension(value) { extension = value; } },
  setTimeout() {},
  console,
});

async function verify(comfyClass, names, savedNames, values, expected) {
  const widgets = names.map(name => ({ name, value: null, options: { values: [] } }));
  const node = { comfyClass, widgets, setDirtyCanvas() {} };
  await extension.nodeCreated(node);
  const saved = { widgets_values: savedNames.map(name => values[name]) };
  // Simulate LiteGraph restoring by position before calling onConfigure.
  widgets.forEach((widget, index) => { widget.value = saved.widgets_values[index]; });
  node.onConfigure(saved);
  for (const [name, value] of Object.entries(expected)) {
    assert.deepEqual(widgets.find(widget => widget.name === name)?.value, value,
      `${comfyClass}: wrong restored widget ${name}`);
  }
}

const referenceNames = [
  "vae_rope_enable", "vae_rope_boost", "vae_rope_reference_fit",
  "vae_rope_placement_grid", "vae_rope_grid_horizontal_position",
  "vae_rope_grid_vertical_position", "vae_rope_rope_mode",
  "vae_rope_reference_stack", "vae_rope_rope_geometry",
  "vae_rope_prompt_annotation", "vae_rope_attention_scope",
  "vision_enable", "vision_resize", "vision_max_edge", "vision_mode",
  "vision_instruction", "vision_style_processing", "vision_guidance_seed_mode",
  "vision_guidance_seed", "vision_vision_fidelity",
];
const oldReferenceNames = referenceNames.filter(name => name !== "vae_rope_prompt_annotation");
oldReferenceNames.splice(oldReferenceNames.indexOf("vision_vision_fidelity"), 0, "vision_reference_annotation");
const referenceValues = {
  vae_rope_enable: true, vae_rope_boost: 1, vae_rope_reference_fit: "native",
  vae_rope_placement_grid: "inside", vae_rope_grid_horizontal_position: "center",
  vae_rope_grid_vertical_position: "center", vae_rope_rope_mode: "spatial",
  vae_rope_reference_stack: "overlay", vae_rope_rope_geometry: "source",
  vae_rope_prompt_annotation: "This is the subject image.",
  vision_reference_annotation: "This is the subject image.",
  vae_rope_attention_scope: "global", vision_enable: true, vision_resize: true,
  vision_max_edge: 768, vision_mode: "visual", vision_instruction: "",
  vision_style_processing: "full", vision_guidance_seed_mode: "fixed",
  vision_guidance_seed: 55, vision_vision_fidelity: 0.25,
};
const latentNames = [
  "dimensions", "width", "height", "preset_size", "geometry_policy",
  "content", "content_fit", "latent_vision_enable", "latent_vision_resize",
  "latent_vision_max_edge", "latent_vision_annotation", "latent_vision_mode",
  "latent_vision_instruction", "latent_vision_style_processing",
  "latent_vision_guidance_seed_mode", "latent_vision_guidance_seed",
  "latent_vision_vision_fidelity", "batch_size",
];
const oldLatentNames = latentNames.filter(name => name !== "latent_vision_annotation");
oldLatentNames.splice(oldLatentNames.indexOf("latent_vision_vision_fidelity"), 0,
  "latent_vision_guidance_detail", "latent_vision_vision_guidance",
  "latent_vision_reference_annotation");
const latentValues = {
  dimensions: "preset", width: 1024, height: 1024, preset_size: "1024 x 1024",
  geometry_policy: "preserve_aspect_krea_bounds", content: "from_image",
  content_fit: "contain", latent_vision_enable: true,
  latent_vision_resize: true, latent_vision_max_edge: 768,
  latent_vision_annotation: "This is the scene image.",
  latent_vision_reference_annotation: "This is the scene image.",
  latent_vision_mode: "context", latent_vision_instruction: "",
  latent_vision_style_processing: "full", latent_vision_guidance_seed_mode: "fixed",
  latent_vision_guidance_seed: 7, latent_vision_guidance_detail: "balanced",
  latent_vision_vision_guidance: "balanced", latent_vision_vision_fidelity: 0.25,
  batch_size: 1,
};

(async () => {
  const referenceExpected = {
    vae_rope_prompt_annotation: referenceValues.vae_rope_prompt_annotation,
    vision_mode: "visual", vision_max_edge: 768, vision_guidance_seed: 55,
  };
  await verify("CcCKrea2Reference", referenceNames, oldReferenceNames,
    referenceValues, referenceExpected);
  await verify("CcCKrea2Reference", referenceNames, referenceNames,
    referenceValues, referenceExpected);
  const latentExpected = {
    latent_vision_annotation: latentValues.latent_vision_annotation,
    latent_vision_mode: "context", latent_vision_max_edge: 768,
    latent_vision_guidance_seed: 7, latent_vision_enable: true,
  };
  await verify("CcCKrea2Latent", latentNames, oldLatentNames,
    latentValues, latentExpected);
  await verify("CcCKrea2Latent", latentNames, latentNames,
    latentValues, latentExpected);
  console.log("Reference and Latent widget migration: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
