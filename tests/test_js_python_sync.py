"""Static contract test to ensure Python and JavaScript preset definitions, display labels, and experimental groups stay strictly synchronized dynamically."""

import re
from pathlib import Path
from ccc_krea2.easy_routing import (
    EASY_PRESET_CAPABILITIES,
    EASY_PRESET_DISPLAY_LABELS,
    IDENTITY_TEST_PRESETS,
)
from ccc_krea2.modular_nodes.easy_edit_node import (
    EASY_EDIT_PRESETS,
    CcCKrea2EasyEdit,
    CcCKrea2EasyEditOstris,
)


def _extract_js_array(js_code: str, var_name: str) -> list[str]:
    pattern = rf"const\s+{var_name}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, js_code, re.DOTALL)
    assert match is not None, f"Could not find array '{var_name}' in web/ccc_krea2.js"
    raw_content = match.group(1)
    # Extract string literals
    items = re.findall(r'["\']([^"\']+)["\']', raw_content)
    return items


def _extract_js_dict(js_code: str, var_name: str) -> dict[str, str]:
    pattern = rf"const\s+{var_name}\s*=\s*\{{(.*?)\}};"
    match = re.search(pattern, js_code, re.DOTALL)
    assert match is not None, f"Could not find object '{var_name}' in web/ccc_krea2.js"
    raw_content = match.group(1)
    pairs = re.findall(r'["\']([^"\']+)["\']\s*:\s*["\']([^"\']+)["\']', raw_content)
    return dict(pairs)


def test_js_python_preset_synchronization():
    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    assert js_path.exists(), f"JavaScript file not found at {js_path}"

    js_code = js_path.read_text(encoding="utf-8")

    js_identity_presets = _extract_js_array(js_code, "IDENTITY_TEST_PRESETS")
    js_group_a = _extract_js_array(js_code, "GROUP_A_PRESETS")
    js_group_b = _extract_js_array(js_code, "GROUP_B_PRESETS")
    js_group_c = _extract_js_array(js_code, "GROUP_C_PRESETS")
    js_group_d = _extract_js_array(js_code, "GROUP_D_PRESETS")
    js_display_labels = _extract_js_dict(js_code, "PRESET_DISPLAY_LABELS")

    # 1. EASY_PRESET_CAPABILITIES is non-empty dynamic registry
    assert len(EASY_PRESET_CAPABILITIES) > 0, "EASY_PRESET_CAPABILITIES must not be empty"

    # 2. Assert IDENTITY_TEST_PRESETS match between JS and Python
    assert set(js_identity_presets) == set(IDENTITY_TEST_PRESETS), (
        f"Mismatch between JS and Python IDENTITY_TEST_PRESETS.\nJS: {js_identity_presets}\nPy: {IDENTITY_TEST_PRESETS}"
    )

    # 3. Assert all IDs in JS IDENTITY_TEST_PRESETS exist in Python registry
    for item in js_identity_presets:
        assert item in IDENTITY_TEST_PRESETS
        assert item in EASY_PRESET_CAPABILITIES

    # 4. Assert Group A/B/C/D exact experiment IDs and counts
    expected_group_a = ["transfer_identity_test_a_4_4", "transfer_identity_test_a_4_5", "transfer_identity_test_a_4_6"]
    expected_group_b = ["transfer_identity_test_b_2_5_4", "transfer_identity_test_b_2_5_5", "transfer_identity_test_b_2_5_6"]
    expected_group_c = [
        "transfer_identity_test_c_4_4",
        "transfer_identity_test_c_4_5",
        "transfer_identity_test_c_4_6",
        "transfer_identity_test_c_4_7",
        "transfer_identity_test_c_2_5_4",
        "transfer_identity_test_c_2_5_5",
        "transfer_identity_test_c_2_5_6",
        "transfer_identity_test_c_2_5_9",
    ]
    expected_group_d = [
        "transfer_identity_test_d_s2_5_o2_5",
        "transfer_identity_test_d_s2_5_o4",
        "transfer_identity_test_d_s4_o4",
        "transfer_identity_test_d_s5_o4",
        "transfer_identity_test_d_s6_o4",
        "transfer_identity_test_d_s7_o4",
    ]

    assert js_group_a == expected_group_a
    assert js_group_b == expected_group_b
    assert js_group_c == expected_group_c
    assert js_group_d == expected_group_d

    # 5. Assert all Group A/B/C/D IDs exist in Python and are contained in IDENTITY_TEST_PRESETS
    for grp in [js_group_a, js_group_b, js_group_c, js_group_d]:
        for item in grp:
            assert item in EASY_PRESET_CAPABILITIES
            assert item in IDENTITY_TEST_PRESETS

    # 6. Assert dynamic display label synchronization across Python and JavaScript
    for preset_id in EASY_PRESET_CAPABILITIES.keys():
        assert preset_id in EASY_PRESET_DISPLAY_LABELS, f"Missing Python display label for preset '{preset_id}'"
        assert preset_id in js_display_labels, f"Missing JavaScript display label for preset '{preset_id}'"
        py_label = EASY_PRESET_DISPLAY_LABELS[preset_id]
        js_label = js_display_labels[preset_id]
        assert py_label == js_label, (
            f"Display label mismatch for preset '{preset_id}': Python='{py_label}' vs JS='{js_label}'"
        )

    # 7. Assert Easy Edit and Easy Edit Ostris derive their preset list from canonical Python registry
    easy_edit_presets = CcCKrea2EasyEdit.INPUT_TYPES()["required"]["preset"][0]
    ostris_edit_presets = CcCKrea2EasyEditOstris.INPUT_TYPES()["required"]["preset"][0]
    assert easy_edit_presets == EASY_EDIT_PRESETS
    assert ostris_edit_presets == EASY_EDIT_PRESETS
    assert list(EASY_PRESET_CAPABILITIES.keys()) == list(EASY_EDIT_PRESETS)


def test_js_structure_regression_protection():
    """Verify structural integrity of web/ccc_krea2.js declarations to prevent syntax regressions."""
    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    js_code = js_path.read_text(encoding="utf-8")

    assert "const LEGACY_PRESETS = [" in js_code, "Missing 'const LEGACY_PRESETS = [' declaration"
    assert (
        js_code.count("const migrateLegacyEasyEditWidgets = (info) => {") == 1
    ), "Expected exactly one 'migrateLegacyEasyEditWidgets' declaration"


def test_js_syntax_validation(pytestconfig):
    """Validate JS file syntax with Node.js if available."""
    import shutil
    import subprocess
    import tempfile
    import pytest

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js binary not found in PATH")

    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    with tempfile.NamedTemporaryFile(suffix=".mjs", delete=False) as tmp:
        tmp.write(js_path.read_bytes())
        tmp_path = tmp.name

    res = subprocess.run([node_bin, "--check", tmp_path], capture_output=True, text=True)
    Path(tmp_path).unlink(missing_ok=True)
    assert res.returncode == 0, f"Node.js syntax check failed:\n{res.stderr}"


def test_js_prompt_ui_behavior():
    """Verify frontend logic contract for prompt widget disabling and event triggers."""
    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    js_code = js_path.read_text(encoding="utf-8")

    # 1. Assert posPromptWidget disabled = true when default prompt active
    assert "posPromptWidget.disabled = true;" in js_code

    # 2. Assert posPromptWidget disabled = false when editable / custom
    assert "posPromptWidget.disabled = false;" in js_code

    # 3. Assert updatePromptState triggers on preset, subject widgets, use_default, connections, and lifecycle
    assert "updatePromptState();" in js_code
    assert "[presetWidget, refSubjWidget, subjDescWidget, outfitSourceWidget, styleSourceWidget]" in js_code
    assert "onConnectionsChange" in js_code
    assert "onConfigure" in js_code
    assert "schedulePromptStateUpdate();" in js_code


def test_js_no_use_default_value_mutation_in_update_prompt_state():
    """Verify that updatePromptState() does not mutate useDefaultWidget.value = false when inputs are unavailable."""
    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    js_code = js_path.read_text(encoding="utf-8")

    # Extract updatePromptState function block
    match = re.search(r"const updatePromptState = \(\) => \{(.*?)\};\n\n", js_code, re.DOTALL)
    assert match is not None, "Could not locate updatePromptState function definition in web/ccc_krea2.js"
    fn_body = match.group(1)

    # Invariant: updatePromptState must never overwrite useDefaultWidget.value
    assert "useDefaultWidget.value = false" not in fn_body, (
        "updatePromptState() must not overwrite user's useDefaultWidget.value preference"
    )


def test_experimental_preset_display_labels():
    """Verify all experimental calibration presets have [Experimental...] display label prefixes and map to internal IDs."""
    from ccc_krea2.easy_routing import (
        IDENTITY_TEST_PRESETS,
        EASY_PRESET_DISPLAY_LABELS,
    )
    for preset_id in IDENTITY_TEST_PRESETS:
        if preset_id == "identity_transfer":
            continue
        label = EASY_PRESET_DISPLAY_LABELS[preset_id]
        assert label.startswith("[Experimental"), f"Preset '{preset_id}' label '{label}' must start with '[Experimental'"


def test_legacy_preset_migration_defaults_true():
    """Verify legacy workflow migration splices use_default_prompt = true (matching node default)."""
    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    js_code = js_path.read_text(encoding="utf-8")

    assert "vals.splice(1, 0, true);" in js_code, "Legacy widget migration must default use_default_prompt to true"


def test_js_dom_readonly_controls():
    """Verify readOnly and disabled properties are applied to DOM input controls."""
    js_path = Path(__file__).parent.parent / "web" / "ccc_krea2.js"
    js_code = js_path.read_text(encoding="utf-8")

    assert "posPromptWidget.inputEl.readOnly = true;" in js_code
    assert "posPromptWidget.inputEl.disabled = true;" in js_code



