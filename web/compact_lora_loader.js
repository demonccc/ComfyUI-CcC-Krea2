import { app } from "../../scripts/app.js";

// Two canvas-rendered rows per LoRA. All original widgets remain serializable.
const STEP = 0.05;
const clamp = (v) => Math.min(10, Math.max(-10, v));
const down = (e) => e.type === "pointerdown" || e.type === "mousedown";
const move = (e) => e.type === "pointermove" || e.type === "mousemove";

function update(widget, value, node) {
  const previous = widget.value;
  if (previous === value) return;
  widget.value = value;
  widget.callback?.(value);
  node.onWidgetChanged?.(widget.name, value, previous, widget);
  app.graph?.setDirtyCanvas?.(true, true);
}

function hide(widget) {
  // This is ComfyUI's established hidden-widget convention; do not remove
  // backend widgets, since workflow persistence depends on their values.
  widget.type = "converted-widget";
  widget.draw = () => {};
  widget.computeSize = () => [0, -4];
  widget.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0, maxWidth: 0 });
}

function optionsFor(widget) {
  const values = typeof widget.options?.values === "function"
    ? widget.options.values(widget) : widget.options?.values;
  return Array.isArray(values) ? values : Object.values(values || {});
}

function chooseLora(event, node, widget) {
  const choices = optionsFor(widget);
  const litegraph = globalThis.LiteGraph;
  if (litegraph?.ContextMenu) {
    new litegraph.ContextMenu(choices, {
      event,
      title: "Select LoRA",
      callback: (choice) => {
        const value = typeof choice === "object" ? (choice?.value ?? choice?.content) : choice;
        if (value != null) update(widget, String(value), node);
      },
    });
    return;
  }
  // Fallback for frontends that don't expose LiteGraph globally.
  // Use a native HTML select rather than an unresponsive custom control.
  const select = document.createElement("select");
  for (const choice of choices) {
    const option = document.createElement("option");
    option.value = String(choice);
    option.textContent = String(choice);
    select.appendChild(option);
  }
  select.value = String(widget.value);
  select.style.cssText = "position:fixed;z-index:99999;max-width:500px;background:#292929;color:#eee;padding:8px";
  select.style.left = Math.max(0, event.clientX || 100) + "px";
  select.style.top = Math.max(0, event.clientY || 100) + "px";
  document.body.appendChild(select);
  const finish = () => { update(widget, select.value, node); select.remove(); };
  select.addEventListener("change", finish, { once: true });
  select.addEventListener("blur", () => select.remove(), { once: true });
  select.focus();
}

function install(node) {
  if (node.__cccTenLoraInstalled) return;
  const find = (name) => node.widgets?.find((w) => w.name === name);
  if (!find("lora_10_strength")) return;
  node.__cccTenLoraInstalled = true;
  const groups = [];
  // Reserve a small gap after the MODEL input. On some frontend versions the
  // first custom row otherwise overlaps the input-row selection highlight.
  const firstRowSpacer = {
    name: "ccc_lora_header_spacing",
    type: "custom",
    serialize: false,
    computeSize: (width) => [width, 10],
    draw() {},
  };
  node.addCustomWidget(firstRowSpacer);
  for (let i = 1; i <= 10; i++) {
    const name = find(`lora_${i}_name`);
    const enabled = find(`lora_${i}_enabled`);
    const strength = find(`lora_${i}_strength`);
    if (!name || !enabled || !strength) continue;
    [name, enabled, strength].forEach(hide);

    const select = {
      name: `ccc_lora_${i}_select`, type: "custom", serialize: false,
      value: "", computeSize: (width) => [width, 28],
      draw(ctx, n, width, y, height) {
        const nodeWidth = n.size[0];
        const cy = y + height / 2, right = nodeWidth - 15;
        ctx.save();
        // Compact native-looking boolean toggle on the left.
        ctx.fillStyle = enabled.value ? "#5a788c" : "#454545";
        ctx.beginPath();
        ctx.roundRect(16, cy - 10, 48, 20, 10);
        ctx.fill();
        ctx.fillStyle = enabled.value ? "#e6f4ff" : "#aaa";
        ctx.beginPath();
        ctx.arc(enabled.value ? 53 : 27, cy, 8, 0, Math.PI * 2);
        ctx.fill();
        // LoRA dropdown on the same row, dimmed and locked when disabled.
        ctx.globalAlpha = enabled.value ? 1 : 0.32;
        ctx.fillStyle = "#232323";
        ctx.strokeStyle = "#777";
        ctx.beginPath();
        ctx.roundRect(72, cy - 11, Math.max(20, nodeWidth - 87), 22, 11);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#ccc";
        ctx.font = "12px Arial";
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        const value = String(name.value ?? "None");
        const maxWidth = Math.max(20, nodeWidth - 135);
        let label = value;
        if (ctx.measureText(label).width > maxWidth) {
          while (label.length && ctx.measureText(label + "…").width > maxWidth) {
            label = label.slice(0, -1);
          }
          label += "…";
        }
        ctx.save();
        ctx.beginPath();
        ctx.rect(82, cy - 10, maxWidth, 20);
        ctx.clip();
        ctx.fillText(label, 82, cy);
        ctx.restore();
        ctx.textAlign = "right";
        ctx.fillText("▾", right - 9, cy);
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (!down(event)) return false;
        if (pos[0] <= 68) {
          update(enabled, !enabled.value, n);
        } else if (enabled.value) {
          chooseLora(event, n, name);
        }
        return true;
      },
    };
    const slider = {
      name: `ccc_lora_${i}_slider`, type: "custom", serialize: false,
      value: "", computeSize: (width) => [width, 26],
      draw(ctx, n, width, y, height) {
        const cy = y + height / 2;
        const nodeWidth = n.size[0];
        const left = 42, right = nodeWidth - 100;
        const length = Math.max(20, right - left);
        const ratio = (clamp(Number(strength.value) || 0) + 10) / 20;
        ctx.save();
        ctx.globalAlpha = enabled.value ? 1 : 0.27;
        ctx.font = "12px Arial";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ccc";
        ctx.textAlign = "center";
        ctx.fillText("◀", 24, cy);
        ctx.fillText("▶", nodeWidth - 19, cy);
        ctx.fillStyle = "#4b4b4b";
        ctx.fillRect(left, cy - 3, length, 6);
        ctx.fillStyle = "#809bb4";
        ctx.fillRect(left, cy - 3, length * ratio, 6);
        ctx.fillStyle = "#eee";
        ctx.fillRect(left + length * ratio - 3, cy - 8, 6, 16);
        ctx.textAlign = "right";
        ctx.fillText(Number(strength.value).toFixed(2), nodeWidth - 43, cy);
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (!enabled.value || (!down(event) && !(move(event) && event.buttons === 1))) return true;
        const x = pos[0], width = n.size[0];
        const left = 42, right = Math.max(left + 20, width - 100);
        const original = Number(strength.value) || 0;
        let next;
        if (down(event) && x <= 36) next = original - STEP;
        else if (down(event) && x >= width - 34) next = original + STEP;
        else if (x >= left && x <= right) next = -10 + (x - left) * 20 / (right - left);
        else return false;
        update(strength, Number((Math.round(clamp(next) / STEP) * STEP).toFixed(2)), n);
        return true;
      },
    };
    node.addCustomWidget(select);
    node.addCustomWidget(slider);
    groups.push({ name, enabled, strength, select, slider });
  }
  // Native hidden values first, then interleaved visible rows.
  const widgets = node.widgets;
  const visible = [firstRowSpacer, ...groups.flatMap((g) => [g.select, g.slider])];
  node.widgets = [...widgets.filter((w) => !visible.includes(w)), ...visible];
  node.setSize?.(node.computeSize());
}

function ready(node, tries = 0) {
  if (node.__cccTenLoraInstalled) return;
  if (!node.widgets?.some((w) => w.name === "lora_10_strength")) {
    if (tries < 30) requestAnimationFrame(() => ready(node, tries + 1));
    return;
  }
  try { install(node); }
  catch (error) { console.error("[CcC Krea2] LoRA UI:", error); }
}

app.registerExtension({
  name: "ccc.krea2.tenLoraControls",
  nodeCreated(node) {
    if (node.comfyClass === "CcCKrea2LoRALoader" || node.type === "CcCKrea2LoRALoader")
      ready(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CcCKrea2LoRALoader") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      ready(this);
      return result;
    };
  },
});
