import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CcC.Krea2.PromptCreatorModeTitle",
    async nodeCreated(node) {
        if (!node || node.comfyClass !== "CcCKrea2EditPromptCreator") return;

        const modeWidget = node.widgets?.find((widget) => widget.name === "mode");
        if (!modeWidget) return;

        const updateTitle = () => {
            const mode = modeWidget.value ?? "enhance";
            const currentTitle = String(node.title ?? "");

            if (
                !currentTitle ||
                currentTitle === "Krea2 CcC Edit Prompt Creator" ||
                currentTitle.startsWith("Prompt Creator - ")
            ) {
                node.title = `Prompt Creator - ${mode}`;
            }

            node.setDirtyCanvas?.(true, true);
        };

        const originalCallback = modeWidget.callback;
        modeWidget.callback = function () {
            if (originalCallback) originalCallback.apply(this, arguments);
            updateTitle();
        };

        setTimeout(updateTitle, 120);
    },
});
