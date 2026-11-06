/* Reference / Latent: never migrate or rewrite invalid saved widget values.
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

async function verifyUnmodified(comfyClass, entries) {
  const widgets = entries.map(([name, value]) => ({ name, value, options: { values: [] } }));
  const node = { comfyClass, widgets, setDirtyCanvas() {} };
  await extension.nodeCreated(node);
  node.onConfigure({ widgets_values: entries.map(([, value]) => value) });
  for (let i = 0; i < entries.length; i++) {
    assert.deepEqual(widgets[i].value, entries[i][1],
      `${comfyClass}: unexpected rewrite of ${entries[i][0]}`);
  }
}

(async () => {
  await verifyUnmodified("CcCKrea2Reference", [
    ["vae_rope_enable", true],
    ["vae_rope_reference_fit", "contain"],
    ["vae_rope_placement_grid", "outside"],
    ["vae_rope_rope_geometry", "latent"],
    ["vae_rope_prompt_annotation", "This is the subject image."],
    ["vision_enable", true],
    ["vision_resize", true],
    ["vision_max_edge", 1024],
    ["vision_mode", "visual"],
    ["vision_style_processing", "full"],
    ["vision_guidance_seed_mode", "fixed"],
    ["vision_guidance_seed", 42],
    ["vision_vision_fidelity", 0.65],
  ]);
  await verifyUnmodified("CcCKrea2Reference", [
    ["vae_rope_enable", true], ["vision_enable", true],
    ["vision_mode", ""], ["vision_max_edge", "none"],
    ["vision_guidance_seed", "This is the subject image."],
  ]);
  await verifyUnmodified("CcCKrea2Latent", [
    ["dimensions", "fixed"], ["content", "from_image"],
    ["latent_vision_enable", true], ["latent_vision_mode", "context"],
    ["latent_vision_annotation", "This is the scene image."],
    ["latent_vision_vision_fidelity", 0.25], ["batch_size", 1],
  ]);
  await verifyUnmodified("CcCKrea2Latent", [
    ["dimensions", "fixed"], ["content", "from_image"],
    ["latent_vision_enable", true], ["latent_vision_mode", ""],
    ["latent_vision_vision_fidelity", "balanced"], ["batch_size", ""],
  ]);
  console.log("Reference and Latent do not mutate saved widget values: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
