import { app } from "../../scripts/app.js";

// The original ComfyUI widgets remain the serializable source of truth. This
// extension only changes their presentation (two visible rows per LoRA).
const STEP = 0.05;
const MIN = -10;
const MAX = 10;
const ROW_HEIGHT = 26;

const clamp = (v) => Math.max(MIN, Math.min(MAX, v));
const setStrength = (widget, value, node) => {
  const next = Number(clamp(Math.round(value / STEP) * STEP).toFixed(2));
  widget.value = next;
  widget.callback?.(next);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
};
const isDown = (e) => ["pointerdown", "mousedown"].includes(e.type);
const isMove = (e) => ["pointermove", "mousemove"].includes(e.type);

function install(node) {
  if (node.__cccLoraCompactInstalled) return;
  const find = (name) => node.widgets?.find((w) => w.name === name);
  if (!find("lora_1_name") || !find("lora_1_enabled")) return;
  node.__cccLoraCompactInstalled = true;

  for (let i = 1; i <= 10; i++) {
    const nameWidget = find(`lora_${i}_name`);
    const enabled = find(`lora_${i}_enabled`);
    const strength = find(`lora_${i}_strength`);
    if (!nameWidget || !enabled || !strength) continue;

    // Keep the native LoRA selector. Paint a compact ON/OFF switch over its
    // left edge and intercept clicks there; everywhere else it acts normally.
    const originalDraw = nameWidget.draw?.bind(nameWidget);
    nameWidget.draw = function (ctx, n, width, y, height) {
      ctx.save();
      if (!enabled.value) ctx.globalAlpha = 0.35;
      originalDraw?.(ctx, n, width, y, height);
      ctx.restore();

      const cy = y + height / 2;
      const sx = 18;
      ctx.save();
      ctx.fillStyle = enabled.value ? "#426b57" : "#444";
      ctx.beginPath();
      ctx.roundRect(sx, cy - 10, 57, 20, 10);
      ctx.fill();
      ctx.fillStyle = enabled.value ? "#d8ffe4" : "#aaa";
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(enabled.value ? "ON" : "OFF", sx + 28, cy);
      ctx.restore();

      // Mask the inactive selector, but never its switch.
      if (!enabled.value) {
        ctx.save();
        ctx.fillStyle = "rgba(25,25,25,0.32)";
        ctx.fillRect(79, y + 2, Math.max(0, width - 95), height - 4);
        ctx.restore();
      }
    };

    const originalMouse = nameWidget.mouse?.bind(nameWidget);
    nameWidget.mouse = function (event, pos, n) {
      if (isDown(event) && pos[0] >= 12 && pos[0] <= 80) {
        enabled.value = !enabled.value;
        enabled.callback?.(enabled.value);
        n.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        return true;
      }
      if (!enabled.value) return true; // No dropdown interaction while OFF.
      return originalMouse?.(event, pos, n) ?? false;
    };

    // Remove the original enabled/strength rows without losing serialization.
    for (const widget of [enabled, strength]) {
      widget.type = "converted-widget";
      widget.computeSize = () => [0, -4];
      widget.draw = () => {};
    }

    node.addCustomWidget({
      name: `lora_${i}_compact_strength`,
      type: "ccc_lora_strength",
      serialize: false,
      computeSize: (width) => [width, ROW_HEIGHT],
      draw(ctx, n, width, y, height) {
        const cy = y + height / 2;
        const leftArrow = 19;
        const rightArrow = width - 22;
        const trackStart = 43;
        const trackEnd = width - 99;
        const trackWidth = Math.max(20, trackEnd - trackStart);
        const ratio = (clamp(Number(strength.value) || 0) - MIN) / (MAX - MIN);
        ctx.save();
        ctx.globalAlpha = enabled.value ? 1 : 0.3;
        ctx.fillStyle = "#aaa";
        ctx.font = "13px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("◀", leftArrow, cy);
        ctx.fillText("▶", rightArrow, cy);
        ctx.fillStyle = "#505050";
        ctx.fillRect(trackStart, cy - 3, trackWidth, 6);
        ctx.fillStyle = "#849aad";
        ctx.fillRect(trackStart, cy - 3, ratio * trackWidth, 6);
        ctx.fillStyle = "#eee";
        ctx.fillRect(trackStart + ratio * trackWidth - 3, cy - 8, 6, 16);
        ctx.font = "12px Arial";
        ctx.textAlign = "right";
        ctx.fillText(Number(strength.value).toFixed(2), width - 39, cy);
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (!enabled.value) return true;
        if (!isDown(event) && !(isMove(event) && event.buttons === 1)) return false;
        const x = pos[0];
        const width = n.size[0];
        const start = 43;
        const end = Math.max(start + 20, width - 99);
        if (isDown(event) && x <= 37) {
          setStrength(strength, Number(strength.value) - STEP, n);
        } else if (isDown(event) && x >= width - 34) {
          setStrength(strength, Number(strength.value) + STEP, n);
        } else if (x >= start && x <= end) {
          setStrength(strength, MIN + ((x - start) / (end - start)) * (MAX - MIN), n);
        } else {
          return false;
        }
        return true;
      },
    });
  }

  // Render the selector and arrow-slider together for every LoRA.
  const order = [];
  for (let i = 1; i <= 10; i++) {
    for (const suffix of ["name", "compact_strength", "enabled", "strength"]) {
      const widget = find(`lora_${i}_${suffix}`);
      if (widget) order.push(widget);
    }
  }
  node.widgets = [
    ...node.widgets.filter((w) => !order.includes(w)),
    ...order,
  ];
  node.setSize?.(node.computeSize());
}

// ComfyUI can create the node before all Python-declared widgets are attached.
// Recheck on the next animation frame instead of permanently skipping setup.
function installWhenReady(node, attempts = 0) {
  if (node.__cccLoraCompactInstalled) return;
  if (!node.widgets?.some((w) => w.name === "lora_10_strength")) {
    if (attempts < 12) requestAnimationFrame(() => installWhenReady(node, attempts + 1));
    return;
  }
  try {
    install(node);
  } catch (error) {
    console.error("[CcC Krea2] Compact LoRA controls failed:", error);
  }
}

app.registerExtension({
  name: "ccc.krea2.compactLoraLoader",
  nodeCreated(node) {
    if (node.comfyClass === "CcCKrea2LoRALoader" || node.type === "CcCKrea2LoRALoader") {
      installWhenReady(node);
    }
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CcCKrea2LoRALoader" && nodeData.name !== "Krea2 CcC LoRA Loader") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      // Widgets exist after onNodeCreated; fallback for ComfyUI frontend versions.
      installWhenReady(this);
      return result;
    };
  },
});
