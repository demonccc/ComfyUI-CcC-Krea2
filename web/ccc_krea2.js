import { app } from "../../scripts/app.js";

const PROMPT_CREATOR_PRESETS = {
    enhance: `Analyze the user's instructions and the available visual context, then create a clear English I2I prompt for Krea2 Edit.

Treat the user's instructions as the authoritative guide for the desired result. Use the available visual context to understand and describe that result accurately.

Preserve the user's intent and constraints while improving clarity. Do not add unnecessary assumptions or unrelated details.

Write the result as a direct description of the desired final image, not as an explanation of the editing process.

Return only the final I2I prompt as one plain-text paragraph.`,

    create_from_image: `Analyze the reference edit image and the user's instructions, then create a clear English I2I prompt for Krea2 Edit.

Treat the reference edit image as the authoritative visual blueprint for the desired result. Build the prompt from what is visibly present in that image.

Use the user's instructions to determine what should be preserved, changed, emphasized, or adapted.

Write the result as a direct description of the desired final image, not as an explanation of the editing process.

Return only the final I2I prompt as one plain-text paragraph.`,

    create_from_theme: `Analyze the user's theme or idea and the available visual context, then create a clear English I2I prompt for Krea2 Edit.

Treat the user's theme or idea as the authoritative creative direction. Use the available visual context to build a coherent desired result that fits it.

Keep the prompt focused on the intended final image. Add only the visual detail needed to make the result clear and useful.

Write the result as a direct description of the desired final image, not as an explanation of the editing process.

Return only the final I2I prompt as one plain-text paragraph.`,
};


// Consolidated from the existing compactLoader frontend behavior.
const compactLoader = (() => {
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
        const left = 42, right = nodeWidth - 124;
        const length = Math.max(20, right - left);
        const ratio = (clamp(Number(strength.value) || 0) + 10) / 20;
        ctx.save();
        ctx.globalAlpha = enabled.value ? 1 : 0.27;
        ctx.font = "12px Arial";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ccc";
        ctx.textAlign = "center";
        ctx.fillText("◀", 24, cy);
        ctx.fillText("▶", nodeWidth - 91, cy);
        ctx.fillStyle = "#4b4b4b";
        ctx.fillRect(left, cy - 3, length, 6);
        ctx.fillStyle = "#809bb4";
        ctx.fillRect(left, cy - 3, length * ratio, 6);
        ctx.fillStyle = "#eee";
        ctx.fillRect(left + length * ratio - 3, cy - 8, 6, 16);
        ctx.textAlign = "right";
        ctx.fillText(Number(strength.value).toFixed(2), nodeWidth - 18, cy);
        ctx.restore();
      },
      mouse(event, pos, n) {
        if (!enabled.value || (!down(event) && !(move(event) && event.buttons === 1))) return true;
        const x = pos[0], width = n.size[0];
        const left = 42, right = Math.max(left + 20, width - 124);
        const original = Number(strength.value) || 0;
        let next;
        if (down(event) && x <= 36) next = original - STEP;
        else if (down(event) && x >= width - 111 && x <= width - 73) next = original + STEP;
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

return {
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
};

})();


// Consolidated from the existing compactStack frontend behavior.
const compactStack = (() => {
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
function cleanupLegacySockets(node) {
  for (const key of ["prompt_augmentation", "lora_prompt_settings"]) {
    let index;
    while ((index = node.inputs?.findIndex((s) => s.name === key)) >= 0) node.removeInput(index);
  }
  let index;
  while ((index = node.outputs?.findIndex((s) => s.name === "prompt_augmentation")) >= 0) {
    node.removeOutput(index);
  }
}

function restoreNames(node, info) {
  const values = info?.widgets_values;
  if (!Array.isArray(values)) return;
  const find = (key) => node.widgets?.find((w) => w.name === key);
  // Previously stored widget values had global enabled + global strength
  // followed by 4 triples of enabled/name/strength.
  if (values.length === 14) {
    for (let i = 1; i <= 4; i++) {
      const base = 2 + (i - 1) * 3;
      const enabled = find(`lora_${i}_enabled`);
      const name = find(`lora_${i}_name`);
      const strength = find(`lora_${i}_strength`);
      if (enabled) enabled.value = Boolean(values[base]);
      if (name && typeof values[base + 1] === "string") name.value = values[base + 1];
      if (strength && Number.isFinite(Number(values[base + 2]))) strength.value = Number(values[base + 2]);
    }
  }
}

function install(node) {
  if (node.__cccCompactStack) return;
  const find = (key) => node.widgets?.find((w) => w.name === key);
  const withPrompt = node.comfyClass === "CcCKrea2LoRAStackWithPrompt" || node.type === "CcCKrea2LoRAStackWithPrompt";
  const slots = withPrompt ? 5 : 10;
  if (!find(`lora_${slots}_strength`)) return;
  node.__cccCompactStack = true;
  if (!withPrompt) cleanupLegacySockets(node);

  const visible = [];
  for (let i = 1; i <= slots; i++) {
    const name = find(`lora_${i}_name`);
    const enabled = find(`lora_${i}_enabled`);
    const strength = find(`lora_${i}_strength`);
    if (!name || !enabled || !strength) continue;
    // Old workflows used different widget indexes; a numeric name is invalid.
    const choices = typeof name.options?.values === "function" ? name.options.values(name) : name.options?.values;
    if (!Array.isArray(choices) || !choices.includes(name.value)) {
      name.value = "None";
    }
    if (!Number.isFinite(Number(strength.value))) strength.value = 1.0;
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
        // Edit in place on top of the numeric area, not a blocking prompt.
        const canvas = app.canvas?.canvas;
        if (!canvas) return true;
        const editor = document.createElement("input");
        editor.type = "number"; editor.min = "-10"; editor.max = "10"; editor.step = "0.05";
        editor.value = String(strength.value);
        editor.style.cssText = "position:fixed;z-index:100000;width:65px;height:24px;box-sizing:border-box;text-align:center;background:#252525;color:#eee;border:1px solid #7b9ab0;border-radius:4px;";
        // LiteGraph may pass coordinates only in `pos`, without browser clientX.
        const canvasRect = canvas.getBoundingClientRect();
        const scaleX = canvasRect.width / canvas.width;
        const scaleY = canvasRect.height / canvas.height;
        const ds = app.canvas?.ds;
        const offset = ds?.offset || [0, 0];
        const zoom = ds?.scale || 1;
        const nodeX = (n.pos[0] + offset[0]) * zoom;
        const nodeY = (n.pos[1] + offset[1]) * zoom;
        const widgetY = pos[1];
        const clientX = Number.isFinite(event.clientX) ? event.clientX : canvasRect.left + (nodeX + w - 59) * scaleX;
        const clientY = Number.isFinite(event.clientY) ? event.clientY : canvasRect.top + (nodeY + widgetY) * scaleY;
        editor.style.left = Math.max(0, clientX - 32) + "px";
        editor.style.top = Math.max(0, clientY - 12) + "px";
        document.body.appendChild(editor);
        let completed = false;
        const commit = () => {
          if (completed) return;
          completed = true;
          const value = Number(editor.value);
          if (editor.value.trim() && Number.isFinite(value)) set(strength, Number(clamp(value).toFixed(2)), n);
          editor.remove();
        };
        editor.addEventListener("keydown", (e) => {
          e.stopPropagation();
          if (e.key === "Enter") commit();
          if (e.key === "Escape") { completed = true; editor.remove(); }
        });
        editor.addEventListener("blur", commit, { once: true });
        editor.focus(); editor.select();
        return true;
      },
    };
    node.addCustomWidget(row);
    visible.push(row);
    if (withPrompt) {
      const position = find(`lora_${i}_prompt_position`);
      const prompt = find(`lora_${i}_prompt`);
      if (position && prompt) {
        [position, prompt].forEach(hide);
        const promptRow = {
          name: `ccc_stack_${i}_prompt_row`, type: "custom", serialize: false,
          computeSize: (width) => [width, 76],
          draw(ctx, n, width, y, height) {
            const w = n.size[0], left = 14, comboWidth = 86, fieldX = 106;
            ctx.save();
            ctx.globalAlpha = enabled.value ? 1 : 0.3;
            ctx.fillStyle = "#252525"; ctx.strokeStyle = "#737373";
            ctx.beginPath(); ctx.roundRect(left, y + 3, comboWidth, 24, 6); ctx.fill(); ctx.stroke();
            ctx.beginPath(); ctx.roundRect(fieldX, y + 3, Math.max(60, w - fieldX - 14), 69, 5); ctx.fill(); ctx.stroke();
            ctx.fillStyle = "#ddd"; ctx.textBaseline = "middle"; ctx.font = "12px Arial";
            ctx.textAlign = "left"; ctx.fillText(String(position.value || "append"), left + 8, y + 15);
            ctx.textAlign = "right"; ctx.fillText("▾", left + comboWidth - 8, y + 15);
            ctx.textAlign = "left"; ctx.textBaseline = "top";
            const text = String(prompt.value || "");
            const maxChars = Math.max(8, Math.floor((w - fieldX - 28) / 7));
            const lines = text.split("\n").flatMap((line) => {
              const result = [];
              while (line.length > maxChars) { result.push(line.slice(0, maxChars)); line = line.slice(maxChars); }
              result.push(line);
              return result;
            });
            (lines.length ? lines : ["Prompt / trigger…"]).slice(0, 3).forEach((line, index) => {
              ctx.fillStyle = text ? "#ddd" : "#888";
              ctx.fillText(line, fieldX + 8, y + 10 + index * 20);
            });
            ctx.restore();
          },
          mouse(event, pos, n) {
            if (!press(event)) return false;
            if (!enabled.value) return true;
            const canvas = app.canvas?.canvas;
            if (!canvas) return true;
            const element = pos[0] < 104 ? document.createElement("select") : document.createElement("textarea");
            const editingPosition = pos[0] < 104;
            if (editingPosition) {
              for (const value of ["append", "prepend"]) {
                const option = document.createElement("option");
                option.value = value; option.textContent = value; element.appendChild(option);
              }
              element.value = position.value;
            } else {
              element.value = String(prompt.value || "");
              element.rows = 3;
              element.placeholder = "Positive prompt / trigger";
            }
            const rect = canvas.getBoundingClientRect();
            const clientX = Number.isFinite(event.clientX) ? event.clientX : rect.left + 120;
            const clientY = Number.isFinite(event.clientY) ? event.clientY : rect.top + 120;
            element.style.cssText = "position:fixed;z-index:100000;box-sizing:border-box;background:#252525;color:#eee;border:1px solid #7895af;border-radius:4px;padding:5px;font:12px Arial;";
            element.style.left = Math.max(0, clientX - (editingPosition ? 40 : 60)) + "px";
            element.style.top = Math.max(0, clientY - 10) + "px";
            element.style.width = editingPosition ? "105px" : Math.max(130, n.size[0] - 125) + "px";
            if (!editingPosition) element.style.height = "70px";
            document.body.appendChild(element);
            let finished = false;
            const commit = () => {
              if (finished) return;
              finished = true;
              set(editingPosition ? position : prompt, element.value, n);
              element.remove();
            };
            element.addEventListener("change", () => { if (editingPosition) commit(); });
            element.addEventListener("keydown", (e) => {
              e.stopPropagation();
              if (e.key === "Escape") { finished = true; element.remove(); }
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) commit();
            });
            element.addEventListener("blur", commit, { once: true });
            element.focus();
            return true;
          },
        };
        node.addCustomWidget(promptRow);
        visible.push(promptRow);
      }
    }
  }
  node.widgets = [...node.widgets.filter((w) => !visible.includes(w)), ...visible];
  node.setSize?.(node.computeSize());
}
function ready(node, attempt = 0) {
  if (node.__cccCompactStack) return;
  const slots = (node.comfyClass === "CcCKrea2LoRAStackWithPrompt" || node.type === "CcCKrea2LoRAStackWithPrompt") ? 5 : 10;
  if (!node.widgets?.some((w) => w.name === `lora_${slots}_strength`)) {
    if (attempt < 30) requestAnimationFrame(() => ready(node, attempt + 1));
    return;
  }
  try { install(node); } catch (error) { console.error("[CcC Krea2] LoRA Stack UI", error); }
}
return {
  name: "ccc.krea2.compactLoraStack",
  nodeCreated(node) {
    if (["CcCKrea2LoRAStack", "CcCKrea2LoRAStackWithPrompt"].includes(node.comfyClass || node.type)) ready(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["CcCKrea2LoRAStack", "CcCKrea2LoRAStackWithPrompt"].includes(nodeData.name)) return;
    const configure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = configure?.call(this, info);
      for (const obsolete of ["prompt_augmentation", "lora_prompt_settings"]) {
        const index = this.inputs?.findIndex((input) => input.name === obsolete) ?? -1;
        if (index >= 0) this.removeInput(index);
      }
      if (nodeData.name === "CcCKrea2LoRAStack") {
        restoreNames(this, info);
        cleanupLegacySockets(this);
      }
      return result;
    };
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = original?.apply(this, args);
      ready(this);
      return result;
    };
  },
};

})();

function setWidgetDisabled(node, name, disabled) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (widget) widget.disabled = disabled;
}

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;
        compactLoader.nodeCreated?.(node);
        compactStack.nodeCreated?.(node);

        if (node.comfyClass === "CcCKrea2VisualReference") {
            const semanticWidget = node.widgets?.find((w) => w.name === "semantic");
            const semanticResizeWidget = node.widgets?.find((w) => w.name === "semantic_resize");
            const fitWidget = node.widgets?.find((w) => w.name === "reference_fit");

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                const resizeEnabled = enabled && semanticResizeWidget?.value === true;
                setWidgetDisabled(node, "semantic_resize", !enabled);
                setWidgetDisabled(node, "semantic_grounding_px", !resizeEnabled);
                setWidgetDisabled(node, "semantic_resize_method", !resizeEnabled);
                setWidgetDisabled(node, "prompt_annotation", !enabled);
            };

            const updateFitState = () => {
                const mode = fitWidget?.value ?? "native";
                const crop = mode === "crop";
                setWidgetDisabled(node, "placement_grid", crop);
                setWidgetDisabled(node, "resize_method", mode !== "resize");
                if (crop) {
                    const placement = node.widgets?.find((w) => w.name === "placement_grid");
                    if (placement) placement.value = "inside";
                }
            };

            for (const [widget, callback] of [
                [semanticWidget, updateSemanticState],
                [semanticResizeWidget, updateSemanticState],
                [fitWidget, updateFitState],
            ]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    callback();
                };
            }

            setTimeout(() => {
                updateSemanticState();
                updateFitState();
            }, 20);
        }

        if (node.comfyClass === "CcCKrea2VisionReference") {
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            const processingWidget = node.widgets?.find((w) => w.name === "style_processing");

            const updateVisionReferenceState = () => {
                const styleMode = (modeWidget?.value ?? "semantic") === "style";
                if (processingWidget) {
                    processingWidget.disabled = !styleMode;
                    if (!styleMode) processingWidget.value = "full";
                }
            };

            if (modeWidget) {
                const originalCallback = modeWidget.callback;
                modeWidget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updateVisionReferenceState();
                };
            }

            setTimeout(updateVisionReferenceState, 20);
        }

        if (node.comfyClass === "CcCKrea2TextToImage") {
            const seedModeWidget = node.widgets?.find((w) => w.name === "guidance_seed_mode");
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");

            const updateGuidanceSeedState = () => {
                const mode = seedModeWidget?.value ?? "fixed";
                setWidgetDisabled(node, "guidance_seed", mode !== "fixed");
            };

            const updateDimensionsState = () => {
                const mode = dimensionsWidget?.value ?? "preset";
                setWidgetDisabled(node, "preset_size", mode !== "preset");
                setWidgetDisabled(node, "width", mode !== "fixed");
                setWidgetDisabled(node, "height", mode !== "fixed");
                setWidgetDisabled(node, "geometry_policy", mode !== "fixed");
            };

            for (const [widget, callback] of [
                [seedModeWidget, updateGuidanceSeedState],
                [dimensionsWidget, updateDimensionsState],
            ]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    const result = originalCallback?.apply(this, arguments);
                    callback();
                    return result;
                };
            }

            setTimeout(() => {
                updateGuidanceSeedState();
                updateDimensionsState();
            }, 20);
        }

        if (node.comfyClass === "CcCKrea2Latent") {
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
            const contentWidget = node.widgets?.find((w) => w.name === "content");
            const contentFitWidget = node.widgets?.find((w) => w.name === "content_fit");
            const semanticWidget = node.widgets?.find((w) => w.name === "latent_semantic");

            const updateDimensionsState = () => {
                const mode = dimensionsWidget?.value ?? "preset";
                setWidgetDisabled(node, "width", mode !== "fixed");
                setWidgetDisabled(node, "height", mode !== "fixed");
                setWidgetDisabled(node, "preset_size", mode !== "preset");
                setWidgetDisabled(node, "geometry_policy", mode === "preset");
            };

            const updateContentState = () => {
                const fromImage = contentWidget?.value === "from_image";
                setWidgetDisabled(node, "content_fit", !fromImage);
                setWidgetDisabled(node, "resize_method", !fromImage);
            };

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                setWidgetDisabled(node, "latent_semantic_instruction", !enabled);
                setWidgetDisabled(node, "latent_grounding_px", !enabled);
            };

            for (const [widget, callback] of [
                [dimensionsWidget, updateDimensionsState],
                [contentWidget, updateContentState],
                [contentFitWidget, updateContentState],
                [semanticWidget, updateSemanticState],
            ]) {
                if (!widget) continue;
                const originalCallback = widget.callback;
                widget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    callback();
                };
            }

            setTimeout(() => {
                updateDimensionsState();
                updateContentState();
                updateSemanticState();
            }, 20);
        }


        if (node.comfyClass === "CcCKrea2EditPromptCreator") {
            // Existing prompt-creator title behavior consolidated here.
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            if (modeWidget) {
                const updateTitle = () => {
                    const mode = modeWidget.value ?? "enhance";
                    const current = String(node.title ?? "");
                    if (!current || current === "Krea2 CcC Edit Prompt Creator" ||
                        current.startsWith("Prompt Creator - ")) {
                        node.title = `Prompt Creator - ${mode}`;
                    }
                    node.setDirtyCanvas?.(true, true);
                };
                const originalCallback = modeWidget.callback;
                modeWidget.callback = function (...args) {
                    const result = originalCallback?.apply(this, args);
                    updateTitle();
                    return result;
                };
                setTimeout(updateTitle, 120);
            }
        }

        if (node.comfyClass === "CcCKrea2EditPromptCreator") {
            const modeWidget = node.widgets?.find((w) => w.name === "mode");
            const systemPromptWidget = node.widgets?.find((w) => w.name === "system_prompt");

            if (!modeWidget || !systemPromptWidget) return;

            let previousMode = modeWidget.value ?? "enhance";
            let customSystemPrompt =
                previousMode === "custom" ? String(systemPromptWidget.value ?? "") : "";

            const setSystemPromptValue = (value) => {
                if (systemPromptWidget.value !== value) {
                    systemPromptWidget.value = value;
                }

                const inputEl = systemPromptWidget.inputEl ?? systemPromptWidget.element;
                if (inputEl && "value" in inputEl && inputEl.value !== value) {
                    inputEl.value = value;
                }

                systemPromptWidget.triggerDraw?.();
            };

            const setSystemPromptEditable = (editable) => {
                systemPromptWidget.options ??= {};
                systemPromptWidget.options.read_only = !editable;
                systemPromptWidget.disabled = false;

                const inputEl = systemPromptWidget.inputEl ?? systemPromptWidget.element;
                if (inputEl) {
                    inputEl.readOnly = !editable;
                    inputEl.style.opacity = editable ? "" : "0.65";
                }

                systemPromptWidget.triggerDraw?.();
            };

            const updatePromptCreatorState = () => {
                const mode = modeWidget.value ?? "enhance";

                if (previousMode === "custom" && mode !== "custom") {
                    customSystemPrompt = String(systemPromptWidget.value ?? "");
                }

                if (mode === "custom") {
                    if (!customSystemPrompt.trim()) {
                        customSystemPrompt =
                            PROMPT_CREATOR_PRESETS[previousMode] ??
                            PROMPT_CREATOR_PRESETS.enhance;
                    }
                    setSystemPromptValue(customSystemPrompt);
                    setSystemPromptEditable(true);
                } else {
                    setSystemPromptValue(
                        PROMPT_CREATOR_PRESETS[mode] ?? PROMPT_CREATOR_PRESETS.enhance
                    );
                    setSystemPromptEditable(false);
                }

                previousMode = mode;
                node.setDirtyCanvas?.(true, true);
            };

            const originalModeCallback = modeWidget.callback;
            modeWidget.callback = function () {
                if (originalModeCallback) originalModeCallback.apply(this, arguments);
                updatePromptCreatorState();
            };

            const originalSystemPromptCallback = systemPromptWidget.callback;
            systemPromptWidget.callback = function () {
                if (originalSystemPromptCallback) {
                    originalSystemPromptCallback.apply(this, arguments);
                }
                if ((modeWidget.value ?? "enhance") === "custom") {
                    customSystemPrompt = String(systemPromptWidget.value ?? "");
                }
            };

            setTimeout(updatePromptCreatorState, 100);
        }

    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        compactLoader.beforeRegisterNodeDef?.(nodeType, nodeData);
        compactStack.beforeRegisterNodeDef?.(nodeType, nodeData);
    },
});
