import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import assert from "assert";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const jsPath = path.resolve(__dirname, "../web/ccc_krea2.js");
let jsCode = fs.readFileSync(jsPath, "utf-8");

// Stub out export / import
jsCode = jsCode.replace(/import\s+\{\s*app\s*\}\s+from\s+["'].*?["'];?/, "");

let registeredExtension = null;
const app = {
    registerExtension(ext) {
        registeredExtension = ext;
    }
};

// Global timers for sync testing
global.requestAnimationFrame = (cb) => {
    cb();
    return 1;
};
global.cancelAnimationFrame = (id) => {};

// Evaluate script in function scope with stubbed app
const fn = new Function("app", jsCode);
fn(app);

assert(registeredExtension, "Extension was not registered");

function createMockNode(comfyClass = "CcCKrea2EasyEdit") {
    const node = {
        comfyClass,
        inputs: [
            { name: "subject", link: null },
            { name: "scene", link: null },
            { name: "outfit", link: null },
            { name: "style", link: null }
        ],
        widgets: [],
        setDirtyCanvasCalls: 0,
        setDirtyCanvas(a, b) {
            this.setDirtyCanvasCalls++;
        }
    };

    // Actual ComfyUI INPUT_TYPES widget ordering
    const posPromptWidget = { name: "positive_prompt", value: "", inputEl: { readOnly: false, disabled: false, title: "" } };
    const useDefaultWidget = { name: "use_default_prompt", value: true };
    const presetWidget = { name: "preset", value: "balanced", options: { values: [] } };
    const refSubjWidget = { name: "reference_subject", value: "main subject" };
    const subjDescWidget = { name: "subject_description", value: "main subject" };
    const outfitSourceWidget = { name: "outfit_source", value: "outfit image" };
    const styleSourceWidget = { name: "style_source", value: "style image" };
    const patchWidget = { name: "apply_krea2_edit_patch", value: true };

    node.widgets = [
        posPromptWidget,
        useDefaultWidget,
        presetWidget,
        refSubjWidget,
        subjDescWidget,
        outfitSourceWidget,
        styleSourceWidget,
        patchWidget
    ];

    registeredExtension.nodeCreated(node);

    const setPreset = (val) => {
        presetWidget.value = val;
        if (presetWidget.callback) {
            presetWidget.callback(val);
        }
    };

    return {
        node,
        posPromptWidget,
        useDefaultWidget,
        presetWidget,
        refSubjWidget,
        subjDescWidget,
        outfitSourceWidget,
        styleSourceWidget,
        patchWidget,
        setPreset
    };
}

async function runTests() {
    // 1. Test Identity Transfer Baseline
    {
        const { node, posPromptWidget, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("identity_transfer");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(posPromptWidget.disabled, true);
        assert.strictEqual(posPromptWidget.readOnly, true);
        assert.strictEqual(posPromptWidget.inputEl.readOnly, true);
        assert.strictEqual(posPromptWidget.inputEl.disabled, true);
        assert(posPromptWidget.value.includes("Replace only the identity"), "Prompt should contain canonical identity transfer text");
    }

    // 2. Test 3 (transfer_identity_test_3)
    {
        const { node, presetWidget, posPromptWidget, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("[Experimental] Identity Test 3 — Scene 4 / Subject 7");
        await new Promise(r => setTimeout(r, 60));
        assert.strictEqual(presetWidget.value, "transfer_identity_test_3");

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert(posPromptWidget.value.includes("Replace only the identity"), "Test 3 canonical prompt preserved");
    }

    // 3. Test 5 (transfer_identity_test_5)
    {
        const { node, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_5");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Disabled]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 4. Group A & B
    {
        const { node, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_a_4_4");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 5. Group C
    {
        const { node, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_c_4_4");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(styleSourceWidget.label, "Style Source [Auto: Scene Outfit]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 6. Group D
    {
        const { node, outfitSourceWidget, styleSourceWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;

        setPreset("transfer_identity_test_d_s2_5_o2_5");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(outfitSourceWidget.label, "Outfit Source [Auto: Scene]");
        assert.strictEqual(outfitSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "outfit").disabled, true);
        assert.strictEqual(styleSourceWidget.label, "Style Source [Disabled]");
        assert.strictEqual(styleSourceWidget.disabled, true);
        assert.strictEqual(node.inputs.find(i => i.name === "style").disabled, true);
    }

    // 7. Normal presets
    {
        const { outfitSourceWidget, setPreset } = createMockNode();
        setPreset("preserve_scene");
        await new Promise(r => setTimeout(r, 60));
        assert.strictEqual(outfitSourceWidget.disabled, true, "preserve_scene disables outfitSourceWidget");

        setPreset("outfit_transfer");
        await new Promise(r => setTimeout(r, 60));
        assert.strictEqual(outfitSourceWidget.disabled, false, "outfit_transfer enables outfitSourceWidget");
    }

    // 8. Workflow restoration sequence with real widget ordering
    {
        const { node, useDefaultWidget, posPromptWidget } = createMockNode();
        // Old Schema 1 workflow: ["prompt text", "identity_transfer", "outfit image", "style image", true]
        const infoOld = {
            widgets_values: ["", "identity_transfer", "outfit image", "style image", true]
        };
        node.onConfigure(infoOld);
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(useDefaultWidget.value, true);
        assert.strictEqual(node._isPromptSystemManaged, true);
        assert.strictEqual(infoOld.widgets_values[0], "");
        assert.strictEqual(infoOld.widgets_values[1], true);
        assert.strictEqual(infoOld.widgets_values[2], "identity_transfer");
        assert.strictEqual(infoOld.widgets_values[3], "main subject");
        assert.strictEqual(infoOld.widgets_values[4], "main subject");
        assert.strictEqual(infoOld.widgets_values[5], "outfit image");
        assert.strictEqual(infoOld.widgets_values[6], "style image");

        // Links appear asynchronously
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;
        node.onConnectionsChange();
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(useDefaultWidget.value, true);
        assert.strictEqual(node._isPromptSystemManaged, true);
        assert.strictEqual(posPromptWidget.disabled, true);
        assert.strictEqual(posPromptWidget.readOnly, true);
        assert(posPromptWidget.value.includes("Replace only the identity"));
    }

    // 9. Manual prompt mode
    {
        const { node, useDefaultWidget, posPromptWidget, setPreset } = createMockNode();
        node.inputs.find(i => i.name === "subject").link = 1;
        node.inputs.find(i => i.name === "scene").link = 2;
        node.onConnectionsChange();
        await new Promise(r => setTimeout(r, 60));

        useDefaultWidget.value = false;
        useDefaultWidget.callback();

        posPromptWidget.value = "User custom prompt text";

        setPreset("transfer_identity_test_a_4_4");
        await new Promise(r => setTimeout(r, 60));

        assert.strictEqual(posPromptWidget.value, "User custom prompt text");
        assert.strictEqual(posPromptWidget.disabled, false);
        assert.strictEqual(posPromptWidget.readOnly, false);
    }

    // 10. Comprehensive migration testing (Old, Intermediate, Current, Repeated)
    {
        // 10a. Old schema without use_default_prompt and without subject fields
        const { node: node1 } = createMockNode();
        const infoOld = {
            widgets_values: ["custom prompt", "identity_transfer", "outfit image", "style image", true]
        };
        node1.onConfigure(infoOld);
        assert.strictEqual(infoOld.widgets_values[0], "custom prompt");
        assert.strictEqual(infoOld.widgets_values[1], true);
        assert.strictEqual(infoOld.widgets_values[2], "identity_transfer");
        assert.strictEqual(infoOld.widgets_values[3], "main subject");
        assert.strictEqual(infoOld.widgets_values[4], "main subject");
        assert.strictEqual(infoOld.widgets_values[5], "outfit image");
        assert.strictEqual(infoOld.widgets_values[6], "style image");
        assert.strictEqual(infoOld.widgets_values.length, 8);

        // Repeated call idempotency on Old Schema
        const lengthAfterFirst = infoOld.widgets_values.length;
        node1.onConfigure(infoOld);
        assert.strictEqual(infoOld.widgets_values.length, lengthAfterFirst);
        assert.strictEqual(infoOld.widgets_values[1], true);

        // 10b. Intermediate schema with explicit use_default_prompt boolean false but without subject fields
        const { node: node2 } = createMockNode();
        const infoIntermediate = {
            widgets_values: ["custom prompt", false, "identity_transfer", "outfit image", "style image", true]
        };
        node2.onConfigure(infoIntermediate);
        assert.strictEqual(infoIntermediate.widgets_values[0], "custom prompt");
        assert.strictEqual(infoIntermediate.widgets_values[1], false, "Explicit boolean false must be preserved!");
        assert.strictEqual(infoIntermediate.widgets_values[2], "identity_transfer");
        assert.strictEqual(infoIntermediate.widgets_values[3], "main subject");
        assert.strictEqual(infoIntermediate.widgets_values[4], "main subject");
        assert.strictEqual(infoIntermediate.widgets_values[5], "outfit image");
        assert.strictEqual(infoIntermediate.widgets_values[6], "style image");
        assert.strictEqual(infoIntermediate.widgets_values.length, 8);

        // Repeated call idempotency on Intermediate Schema
        node2.onConfigure(infoIntermediate);
        assert.strictEqual(infoIntermediate.widgets_values[1], false);
        assert.strictEqual(infoIntermediate.widgets_values.length, 8);

        // 10c. Current schema with explicit boolean, subject fields, and all widgets
        const { node: node3 } = createMockNode();
        const infoCurrent = {
            widgets_values: ["custom prompt", false, "identity_transfer", "target subject", "subject desc", "outfit image", "style image", true]
        };
        node3.onConfigure(infoCurrent);
        assert.strictEqual(infoCurrent.widgets_values[0], "custom prompt");
        assert.strictEqual(infoCurrent.widgets_values[1], false);
        assert.strictEqual(infoCurrent.widgets_values[2], "identity_transfer");
        assert.strictEqual(infoCurrent.widgets_values[3], "target subject");
        assert.strictEqual(infoCurrent.widgets_values[4], "subject desc");
        assert.strictEqual(infoCurrent.widgets_values[5], "outfit image");
        assert.strictEqual(infoCurrent.widgets_values[6], "style image");
        assert.strictEqual(infoCurrent.widgets_values.length, 8);

        // Repeated call idempotency on Current Schema
        node3.onConfigure(infoCurrent);
        assert.strictEqual(infoCurrent.widgets_values.length, 8);
    }

    // 11. Preset display ordering test
    {
        const { presetWidget } = createMockNode();
        assert(presetWidget.options && typeof presetWidget.options.values === "function");
        const displayedValues = presetWidget.options.values();

        const expectedStableFirst = [
            "Flexible",
            "Balanced",
            "Consistent",
            "Preserve Identity",
            "Max Identity",
            "Identity Transfer",
            "Subject Transfer",
            "Preserve Scene",
            "Outfit Transfer",
            "Style Transfer",
            "Scene Reinterpretation"
        ];

        // Assert the first 11 items match the required stable presets in exact order
        const actualStableFirst = displayedValues.slice(0, 11);
        assert.deepStrictEqual(actualStableFirst, expectedStableFirst, "First 11 presets must be the stable presets in exact order");

        // Assert experimental presets appear after index 10
        const experimentalValues = displayedValues.slice(11);
        assert(experimentalValues.length > 0, "Experimental presets must appear after stable presets");
        assert(experimentalValues.some(v => v.startsWith("[Experimental]")), "Experimental calibration presets must follow stable presets");
        assert(experimentalValues.some(v => v.startsWith("[Experimental A]")), "Group A presets must follow stable presets");
    }

    console.log("All JavaScript frontend behavior & migration tests PASSED successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
