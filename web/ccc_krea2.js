import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "CcC.Krea2",
    async nodeCreated(node) {
        if (!node || !node.comfyClass) return;

        // Dynamic widget graying for CcCKrea2ReferenceImage
        if (node.comfyClass === "CcCKrea2ReferenceImage") {
            const refPathWidget = node.widgets?.find(w => w.name === "reference_path");
            if (refPathWidget) {
                const updateRefPathState = () => {
                    const mode = refPathWidget.value;
                    const isEdit = (mode === "edit");

                    const editWidgets = ["attention_boost", "masked_attention_boost", "visual_reference_fit"];
                    const styleWidgets = ["style_fidelity", "style_processing", "indirect_style_transfer", "style_directive"];

                    node.widgets?.forEach(w => {
                        if (editWidgets.includes(w.name)) {
                            w.disabled = !isEdit;
                        }
                        if (styleWidgets.includes(w.name)) {
                            w.disabled = isEdit;
                        }
                    });
                };

                const origCallback = refPathWidget.callback;
                refPathWidget.callback = function (val) {
                    if (origCallback) origCallback.apply(this, arguments);
                    updateRefPathState();
                };
                setTimeout(updateRefPathState, 20);
            }
        }

        // Dynamic widget graying for CcCKrea2TargetLatent
        if (node.comfyClass === "CcCKrea2TargetLatent") {
            const contentWidget = node.widgets?.find(w => w.name === "target_content");
            const geomWidget = node.widgets?.find(w => w.name === "geometry_mode");

            const updateTargetState = () => {
                const isGeomFixed = (geomWidget?.value === "fixed");

                const targetMpWidget = node.widgets?.find(w => w.name === "target_megapixels");
                const fixedMpWidget = node.widgets?.find(w => w.name === "fixed_megapixels");
                const aspectWidget = node.widgets?.find(w => w.name === "aspect_ratio");

                if (fixedMpWidget) fixedMpWidget.disabled = !isGeomFixed;
                if (targetMpWidget) targetMpWidget.disabled = isGeomFixed;
                if (aspectWidget) aspectWidget.disabled = !isGeomFixed;
            };

            [contentWidget, geomWidget].forEach(w => {
                if (w) {
                    const origCb = w.callback;
                    w.callback = function () {
                        if (origCb) origCb.apply(this, arguments);
                        updateTargetState();
                    };
                }
            });
            setTimeout(updateTargetState, 20);
        }

        // Dynamic widget graying for CcCKrea2Edit & CcCKrea2EasyEdit
        if (node.comfyClass === "CcCKrea2Edit" || node.comfyClass === "CcCKrea2EasyEdit") {
            const methodWidget = node.widgets?.find(w => w.name === "reference_method");
            const patchWidget = node.widgets?.find(w => w.name === "apply_model_patch");
            const kvCacheWidget = node.widgets?.find(w => w.name === "ostris_kv_cache");

            const updateEditState = () => {
                const method = methodWidget?.value || "krea2_edit";

                // Native does not support model patching
                if (method === "native") {
                    if (patchWidget) patchWidget.disabled = true;
                } else {
                    if (patchWidget) patchWidget.disabled = false;
                }

                // ostris_kv_cache is currently unsupported in runtime environment
                if (kvCacheWidget) {
                    kvCacheWidget.disabled = true;
                }
            };

            if (methodWidget) {
                const origCb = methodWidget.callback;
                methodWidget.callback = function () {
                    if (origCb) origCb.apply(this, arguments);
                    updateEditState();
                };
            }
            setTimeout(updateEditState, 20);
        }
    }
});
