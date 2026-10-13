import { app } from "../../scripts/app.js";

// Keep this extension limited to the Latent node: UI-only enable/disable rules.
function install(node) {
  if (node.__cccLatentVisibilityInstalled) return;
  const find = (name) => node.widgets?.find((w) => w.name === name);
  if (!find("dimensions") || !find("geometry_policy") || !find("content")) return;
  node.__cccLatentVisibilityInstalled = true;

  const controlled = [
    "width", "height", "preset_size", "geometry_policy",
    "content_fit", "resize_method",
    "latent_semantic_instruction", "latent_grounding_px",
  ];
  const originals = new Map();
  for (const key of controlled) {
    const widget = find(key);
    if (!widget) continue;
    const draw = widget.draw;
    const mouse = widget.mouse;
    const callback = widget.callback;
    originals.set(key, { draw, mouse, callback });
    widget.draw = function(ctx, ...args) {
      ctx.save();
      if (this.__cccLatentDisabled) ctx.globalAlpha *= 0.32;
      try { return draw?.call(this, ctx, ...args); }
      finally { ctx.restore(); }
    };
    widget.mouse = function(event, ...args) {
      if (this.__cccLatentDisabled) return true;
      return mouse?.call(this, event, ...args);
    };
    widget.__cccLatentVisibilityOriginalCallback = callback;
  }

  function refresh() {
    const dimensions = find("dimensions")?.value;
    const content = find("content")?.value;
    const semantic = Boolean(find("latent_semantic")?.value);
    const active = {
      width: dimensions === "fixed",
      height: dimensions === "fixed",
      preset_size: dimensions === "preset",
      geometry_policy: dimensions !== "preset",
      content_fit: content === "from_image",
      resize_method: content === "from_image",
      latent_semantic_instruction: semantic,
      latent_grounding_px: semantic,
    };
    for (const key of controlled) {
      const widget = find(key);
      if (widget) widget.__cccLatentDisabled = !active[key];
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  }

  for (const key of ["dimensions", "content", "latent_semantic"]) {
    const widget = find(key);
    if (!widget) continue;
    const original = widget.callback;
    widget.callback = function (...args) {
      const result = original?.apply(this, args);
      refresh();
      return result;
    };
  }

  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    refresh();
    return result;
  };
  refresh();
}

function whenReady(node, attempts = 0) {
  if (node.__cccLatentVisibilityInstalled) return;
  if (!node.widgets?.some((w) => w.name === "latent_semantic")) {
    if (attempts < 30) requestAnimationFrame(() => whenReady(node, attempts + 1));
    return;
  }
  try { install(node); }
  catch (error) { console.error("[CcC Krea2] Latent visibility:", error); }
}

app.registerExtension({
  name: "ccc.krea2.latentVisibility",
  nodeCreated(node) {
    if (node.comfyClass === "CcCKrea2Latent" || node.type === "CcCKrea2Latent") whenReady(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CcCKrea2Latent") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      whenReady(this);
      return result;
    };
  },
});
