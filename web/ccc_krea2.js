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

        if (node.comfyClass === "CcCKrea2Latent") {
            const dimensionsWidget = node.widgets?.find((w) => w.name === "dimensions");
            const contentWidget = node.widgets?.find((w) => w.name === "content");
            const imageFitWidget = node.widgets?.find((w) => w.name === "image_fit");
            const semanticWidget = node.widgets?.find((w) => w.name === "latent_semantic");

            const setDisabled = (name, disabled) => {
                const widget = node.widgets?.find((w) => w.name === name);
                if (widget) widget.disabled = disabled;
            };

            const updateDimensionsState = () => {
                const mode = dimensionsWidget?.value ?? "preset";
                setDisabled("width", mode !== "fixed");
                setDisabled("height", mode !== "fixed");
                setDisabled("resolution", mode !== "preset");
                setDisabled("aspect_ratio", mode !== "preset");
            };

            const updateContentState = () => {
                const fromImage = contentWidget?.value === "from_image";
                setDisabled("image_fit", !fromImage);
                const native = imageFitWidget?.value === "native";
                setDisabled("resize_method", !fromImage || native);
            };

            const updateSemanticState = () => {
                const enabled = semanticWidget?.value === true;
                setDisabled("latent_semantic_instruction", !enabled);
                setDisabled("latent_grounding_px", !enabled);
            };

            for (const [widget, callback] of [
                [dimensionsWidget, updateDimensionsState],
                [contentWidget, updateContentState],
                [imageFitWidget, updateContentState],
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
    },
});
