import { app } from "../../scripts/app.js";

// One responsive row: enabled switch | LoRA selector | strength with +/- controls.
const clamp = (v) => Math.max(-10, Math.min(10, v));
const press = (e) => e.type === "pointerdown" || e.type === "mousedown";
function set(w, value, node) {
  const previous = w.value;
  w.value = value;
  w.callback?.(value);
  node.onWidgetChanged?.(w.name, value, previous, w);
  app.graph?.setDirtyCanvas?.(true, true);
}
function hide(w) {
  w.type = "converted-widget";
  w.computeSize = () => [0, -4];
  w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0, maxWidth: 0 });
  w.draw = () => {};
}
function pick(event, node, widget) {
  const raw = typeof widget.options?.values === "function" ? widget.options.values(widget) : widget.options?.values;
  const choices = Array.isArray(raw) ? raw : Object.values(raw || []);
  const menu = globalThis.LiteGraph?.ContextMenu;
  if (menu) {
    new menu(choices, {
      event, title: "Select LoRA",
      callback: (choice) => {
        const value = typeof choice === "object" ? choice?.value ?? choice?.content : choice;
        if (value != null) set(widget, String(value), node);
      },
    });
    return;
  }
  const selector = document.createElement("select");
  for (const choice of choices) {
    const option = document.createElement("option");
    option.value = String(choice);
    option.textContent = String(choice);
    selector.appendChild(option);
  }
  selector.value = String(widget.value);
  selector.style.cssText = "position:fixed;z-index:99999;max-width:500px;background:#292929;color:#eee;padding:8px";
  selector.style.left = Math.max(0, event.clientX || 100) + "px";
  selector.style.top = Math.max(0, event.clientY || 100) + "px";
  document.body.appendChild(selector);
  selector.addEventListener("change", () => { set(widget, selector.value, node); selector.remove(); }, { once: true });
  selector.addEventListener("blur", () => selector.remove(), { once: true });
  selector.focus();
}
function install(node) {
  if (node.__cccCompactStack) return;
  const find = (key) => node.widgets?.find((w) => w.name === key);
  if (!find("lora_10_strength")) return;
  node.__cccCompactStack = true;
  const visible = [];
  for (let i = 1; i <= 10; i++) {
    const name = find(`lora_${i}_name`);
    const enabled = find(`lora_${i}_enabled`);
    const strength = find(`lora_${i}_strength`);
    if (!name || !enabled || !strength) continue;
    [name, enabled, strength].forEach(hide);
    const row = {
      name: `ccc_stack_${i}_row`, type: "custom", serialize: false,
      computeSize: (width) => [width, 30],
      draw(ctx, n, width, y, height) {
        const w = n.size[0], cy = y + height / 2;
        const selectLeft = 73, rightWidth = 106;
        const selectRight = Math.max(selectLeft + 32, w - rightWidth);
        ctx.save();
        ctx.fillStyle = enabled.value ? "#5a788c" : "#454545";
        ctx.beginPath(); ctx.roundRect(17, cy - 10, 47, 20, 10); ctx.fill();
        ctx.fillStyle = enabled.value ? "#e6f4ff" : "#aaa";
        ctx.beginPath(); ctx.arc(enabled.value ? 53 : 27, cy, 8, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = enabled.value ? 1 : 0.32;
        ctx.fillStyle = "#232323"; ctx.strokeStyle = "#777";
        ctx.beginPath(); ctx.roundRect(selectLeft, cy - 11, selectRight - selectLeft, 22, 10); ctx.fill(); ctx.stroke();
        ctx.font = "12px Arial"; ctx.fillStyle = "#ccc"; ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        let label = String(name.value ?? "None");
        const max = Math.max(4, selectRight - selectLeft - 32);
        if (ctx.measureText(label).width > max) {
          while (label.length && ctx.measureText(label + "…").width > max) label = label.slice(0, -1);
          label += "…";
        }
        ctx.save(); ctx.beginPath(); ctx.rect(selectLeft + 8, cy - 10, max, 20); ctx.clip();
        ctx.fillText(label, selectLeft + 8, cy); ctx.restore();
        ctx.textAlign = "right"; ctx.fillText("▾", selectRight - 10, cy);
        // Strength area: decrement, value, increment.
        ctx.fillStyle = "#bbb";
        ctx.textAlign = "center";
        ctx.fillText("◀", w - 94, cy);
        ctx.fillText(Number(strength.value).toFixed(2), w - 59, cy);
        ctx.fillText("▶", w - 19, cy);
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (!press(event)) return false;
        const x = pos[0], w = n.size[0], selectRight = Math.max(105, w - 106);
        if (x < 69) {
          set(enabled, !enabled.value, n); return true;
        }
        if (!enabled.value) return true;
        if (x < selectRight) { pick(event, n, name); return true; }
        if (x < w - 78) { set(strength, Number(clamp(Number(strength.value) - 0.05).toFixed(2)), n); return true; }
        if (x >= w - 34) { set(strength, Number(clamp(Number(strength.value) + 0.05).toFixed(2)), n); return true; }
        // Clicking the number allows exact manual input.
        const answer = window.prompt("LoRA strength (-10 to 10)", String(strength.value));
        if (answer !== null && answer.trim() !== "" && Number.isFinite(Number(answer))) {
          set(strength, Number(clamp(Number(answer)).toFixed(2)), n);
        }
        return true;
      },
    };
    node.addCustomWidget(row);
    visible.push(row);
  }
  node.widgets = [...node.widgets.filter((w) => !visible.includes(w)), ...visible];
  node.setSize?.(node.computeSize());
}
function ready(node, attempt = 0) {
  if (node.__cccCompactStack) return;
  if (!node.widgets?.some((w) => w.name === "lora_10_strength")) {
    if (attempt < 30) requestAnimationFrame(() => ready(node, attempt + 1));
    return;
  }
  try { install(node); } catch (error) { console.error("[CcC Krea2] LoRA Stack UI", error); }
}
app.registerExtension({
  name: "ccc.krea2.compactLoraStack",
  nodeCreated(node) {
    if (node.comfyClass === "CcCKrea2LoRAStack" || node.type === "CcCKrea2LoRAStack") ready(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CcCKrea2LoRAStack") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      ready(this);
      return result;
    };
  },
});
