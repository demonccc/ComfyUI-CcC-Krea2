import { app } from "../../scripts/app.js";

// Keep model name on the first row and enabled + strength together on the second.
app.registerExtension({
  name: "ccc.krea2.compactLoraLoader",
  nodeCreated(node) {
    if (node.comfyClass !== "CcCKrea2LoRALoader") return;
    const find = (name) => node.widgets?.find((widget) => widget.name === name);
    for (let i = 1; i <= 10; i++) {
      const enabled = find(`lora_${i}_enabled`);
      const strength = find(`lora_${i}_strength`);
      if (!enabled || !strength) continue;

      // Retain original Python-backed widget values and serialized workflow state.
      for (const widget of [enabled, strength]) {
        widget.type = "ccc_hidden";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
      }

      const height = 26;
      const row = {
        name: `lora_${i}_controls`,
        type: "ccc_lora_controls",
        serialize: false,
        computeSize: (width) => [width, height],
        draw(ctx, n, width, y, widgetHeight) {
          const x = 15;
          const checkboxX = x + 5;
          const cy = y + widgetHeight / 2;
          const left = x + 91;
          const right = width - 17;
          const trackWidth = Math.max(30, right - left - 48);
          ctx.save();
          ctx.fillStyle = "#aaa";
          ctx.font = "12px Arial";
          ctx.textBaseline = "middle";
          ctx.fillText(`#${i}`, x, cy);
          ctx.strokeStyle = "#aaa";
          ctx.strokeRect(checkboxX + 25, cy - 6, 12, 12);
          if (enabled.value) {
            ctx.fillStyle = "#ddd";
            ctx.fillText("✓", checkboxX + 26, cy + 1);
          }
          ctx.fillStyle = "#aaa";
          ctx.fillText("Enable", checkboxX + 42, cy);
          ctx.fillStyle = "#555";
          ctx.fillRect(left, cy - 3, trackWidth, 6);
          const ratio = (Math.max(-10, Math.min(10, Number(strength.value) || 0)) + 10) / 20;
          ctx.fillStyle = "#9ab";
          ctx.fillRect(left, cy - 3, trackWidth * ratio, 6);
          ctx.fillStyle = "#eee";
          ctx.fillRect(left + trackWidth * ratio - 3, cy - 7, 6, 14);
          ctx.textAlign = "right";
          ctx.fillText(Number(strength.value).toFixed(2), right, cy);
          ctx.restore();
        },
        mouse(event, pos, n) {
          if (event.type !== "pointerdown" && event.type !== "pointermove" && event.type !== "pointerup") return false;
          const width = n.size[0];
          const x = pos[0];
          const left = 106;
          const trackWidth = Math.max(30, width - 17 - left - 48);
          if (event.type === "pointerdown" && x < left) {
            enabled.value = !enabled.value;
            app.graph?.setDirtyCanvas(true, true);
            return true;
          }
          if (x >= left && x <= left + trackWidth && (event.type === "pointerdown" || (event.type === "pointermove" && event.buttons === 1))) {
            const ratio = Math.max(0, Math.min(1, (x - left) / trackWidth));
            strength.value = Math.round((-10 + ratio * 20) * 20) / 20;
            app.graph?.setDirtyCanvas(true, true);
            return true;
          }
          return false;
        },
      };
      node.addCustomWidget(row);
    }
    // Interleave visible rows; keep backing widgets hidden but serializable.
    const modelWidget = find("model");
    const sorted = modelWidget ? [modelWidget] : [];
    for (let i = 1; i <= 10; i++) {
      for (const name of [`lora_${i}_name`, `lora_${i}_controls`, `lora_${i}_enabled`, `lora_${i}_strength`]) {
        const widget = find(name);
        if (widget) sorted.push(widget);
      }
    }
    node.widgets = [...sorted, ...node.widgets.filter((widget) => !sorted.includes(widget))];
    node.setSize(node.computeSize());
  },
});
