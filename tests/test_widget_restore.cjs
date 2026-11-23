/* Reference / Latent: migrate removed Vision presets while preserving other widget values.
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
    const name = entries[i][0];
    const isVisionMode = ["vision_mode", "latent_vision_mode", "mode"].includes(name);
    const expected = isVisionMode && !["visual", "semantic"].includes(entries[i][1])
      ? "visual"
      : entries[i][1];
    assert.deepEqual(widgets[i].value, expected,
      `${comfyClass}: unexpected value for ${name}`);
    if (isVisionMode) {
      assert.deepEqual(Array.from(widgets[i].options.values), ["visual", "semantic"],
        `${comfyClass}: obsolete Vision modes remain in combo options`);
    }
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
  await verifyUnmodified("CcCKrea2VisionReference", [
    ["mode", "style"], ["instruction", "Preserve the color palette."],
  ]);
  // Saved Mask graphs used unload / composite / invert order. Ensure migration
  // produces the new composite / invert / unload order without losing values.
  {
    const saved = ["sam3", "face", 2, 0.5, true, "#FF00AA", true];
    const names = ["source", "prompt", "boost", "confidence",
      "composite_color", "invert", "unload_sam_model"];
    const widgets = names.map((name, i) => ({ name, value: saved[i] }));
    const node = { comfyClass: "CcCKrea2Mask", widgets, setDirtyCanvas() {} };
    await extension.nodeCreated(node);
    node.onConfigure({ widgets_values: saved });
    assert.equal(widgets[4].value, "#FF00AA");
    assert.equal(widgets[5].value, true);
    assert.equal(widgets[6].value, true);
  }
  // Previous Paint Prepare graphs stored geometry_mode in first position.
  {
    const saved = ["crop", "edge", "center", "center", 0, 0, 0, 0,
      false, 0, "gaussian_sigma", 0, "outside"];
    const names = ["padding_fill", "horizontal_position", "vertical_position",
      "expand_left", "expand_top", "expand_right", "expand_bottom",
      "fill_holes", "mask_grow", "mask_blur_mode", "mask_blur_amount",
      "mask_blur_direction"];
    const widgets = names.map((name, i) => ({ name, value: saved[i] }));
    const node = { comfyClass: "CcCKrea2PaintPrepare", widgets, setDirtyCanvas() {} };
    await extension.nodeCreated(node);
    node.onConfigure({ widgets_values: saved });
    assert.equal(widgets[0].value, "edge");
    assert.equal(widgets[11].value, "outside");
  }
  console.log("Mask unload-last and Paint padding widget migration: PASS");
  console.log("Vision modes limited to Visual / Semantic: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
