"""Static contract test to ensure Python and JavaScript preset definitions, display labels, and experimental groups stay strictly synchronized."""

import re
from pathlib import Path
from ccc_krea2.easy_routing import (
    EASY_PRESET_CAPABILITIES,
    EASY_PRESET_DISPLAY_LABELS,
    IDENTITY_TEST_PRESETS,
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

    # 1. Assert IDENTITY_TEST_PRESETS match exactly
    assert set(js_identity_presets) == set(IDENTITY_TEST_PRESETS), (
        f"Mismatch between JS and Python IDENTITY_TEST_PRESETS.\nJS: {js_identity_presets}\nPy: {IDENTITY_TEST_PRESETS}"
    )

    # 2. Assert Group counts match exact counts
    assert len(js_group_a) == 3
    assert len(js_group_b) == 3
    assert len(js_group_c) == 8
    assert len(js_group_d) == 6

    # 3. Assert all experimental groups are contained within IDENTITY_TEST_PRESETS
    for grp in [js_group_a, js_group_b, js_group_c, js_group_d]:
        for item in grp:
            assert item in IDENTITY_TEST_PRESETS
            assert item in EASY_PRESET_CAPABILITIES

    # 4. Assert total preset count is 36
    assert len(EASY_PRESET_CAPABILITIES) == 36

    # 5. Assert display label synchronization across Python and JavaScript for all 36 presets
    for preset_id in EASY_PRESET_CAPABILITIES.keys():
        assert preset_id in EASY_PRESET_DISPLAY_LABELS, f"Missing Python display label for preset '{preset_id}'"
        assert preset_id in js_display_labels, f"Missing JavaScript display label for preset '{preset_id}'"
        py_label = EASY_PRESET_DISPLAY_LABELS[preset_id]
        js_label = js_display_labels[preset_id]
        assert py_label == js_label, (
            f"Display label mismatch for preset '{preset_id}': Python='{py_label}' vs JS='{js_label}'"
        )
