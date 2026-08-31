import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;

        if (node.comfyClass === "CcCKrea2VisualReference") {
            const semanticWidget = node.widgets?.find((w) => w.name === "semantic");

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                for (const name of ["semantic_role", "instruction", "grounding_px"]) {
                    const widget = node.widgets?.find((w) => w.name === name);
                    if (widget) widget.disabled = !enabled;
                }
            };

            if (semanticWidget) {
                const originalCallback = semanticWidget.callback;
                semanticWidget.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    updateSemanticState();
                };
            }

            setTimeout(updateSemanticState, 20);
        }
    },
});
