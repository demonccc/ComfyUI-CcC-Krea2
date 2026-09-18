"""Frontend contract for conditional Visual Reference semantic controls."""

from pathlib import Path


def test_visual_reference_semantic_widgets_are_conditionally_disabled():
    js = Path("web/ccc_krea2.js").read_text(encoding="utf-8")
    assert 'node.comfyClass === "CcCKrea2VisualReference"' in js
    assert 'w.name === "semantic"' in js
    assert 'setDisabled("semantic_resize", !enabled)' in js
    assert 'setDisabled("semantic_grounding_px", !resizeEnabled)' in js
    assert 'setDisabled("semantic_resize_method", !resizeEnabled)' in js
    assert 'setDisabled("prompt_annotation", !enabled)' in js


def test_js_syntax_validation():
    import shutil
    import subprocess
    import tempfile

    node_bin = shutil.which("node")
    if not node_bin:
        return

    js_path = Path("web/ccc_krea2.js")
    with tempfile.NamedTemporaryFile(suffix=".mjs", delete=False) as tmp:
        tmp.write(js_path.read_bytes())
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run([node_bin, "--check", str(tmp_path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    finally:
        tmp_path.unlink(missing_ok=True)
