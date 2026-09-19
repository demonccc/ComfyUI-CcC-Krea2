"""Regression tests for package-relative imports and entrypoint package loading."""

import ast
from pathlib import Path
import importlib.util
import sys


def test_no_absolute_internal_package_imports():
    """Scan production Python files under ccc_krea2/ using AST to ensure no absolute sibling imports exist."""
    repo_root = Path(__file__).resolve().parent.parent
    ccc_krea2_dir = repo_root / "ccc_krea2"

    assert ccc_krea2_dir.is_dir(), f"Directory not found: {ccc_krea2_dir}"

    offending_imports = []

    for py_file in ccc_krea2_dir.rglob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Reject level == 0 (absolute) imports targeting ccc_krea2 or ccc_krea2.*
                if (
                    node.level == 0
                    and node.module
                    and (node.module == "ccc_krea2" or node.module.startswith("ccc_krea2."))
                ):
                    rel_path = py_file.relative_to(repo_root)
                    imported_names = ", ".join(alias.name for alias in node.names)
                    stmt = f"from {node.module} import {imported_names}"
                    offending_imports.append((str(rel_path), node.lineno, stmt))

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ccc_krea2" or alias.name.startswith("ccc_krea2."):
                        rel_path = py_file.relative_to(repo_root)
                        stmt = f"import {alias.name}"
                        offending_imports.append((str(rel_path), node.lineno, stmt))

    if offending_imports:
        msg_lines = ["Absolute internal package imports found in production code:"]
        for file_path, lineno, stmt in offending_imports:
            msg_lines.append(f"  - {file_path}:{lineno} -> {stmt}")
        msg_lines.append("Production code inside ccc_krea2 package must use package-relative imports.")
        assert False, "\n".join(msg_lines)


def test_repository_entrypoint_package_loading():
    """Smoke test ensuring repository root __init__.py can be loaded as a package and exports node mappings."""
    repo_root = Path(__file__).resolve().parent.parent
    init_py = repo_root / "__init__.py"

    assert init_py.is_file(), f"Root __init__.py not found at {init_py}"

    package_name = "test_custom_node_package"
    spec = importlib.util.spec_from_file_location(
        package_name, str(init_py), submodule_search_locations=[str(repo_root)]
    )
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module

    try:
        spec.loader.exec_module(module)

        assert hasattr(module, "NODE_CLASS_MAPPINGS"), "Entrypoint package missing NODE_CLASS_MAPPINGS"
        assert hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"), "Entrypoint package missing NODE_DISPLAY_NAME_MAPPINGS"
        assert isinstance(module.NODE_CLASS_MAPPINGS, dict), "NODE_CLASS_MAPPINGS must be a dict"
        assert isinstance(module.NODE_DISPLAY_NAME_MAPPINGS, dict), "NODE_DISPLAY_NAME_MAPPINGS must be a dict"
        assert len(module.NODE_CLASS_MAPPINGS) > 0, "NODE_CLASS_MAPPINGS must not be empty"
    finally:
        sys.modules.pop(package_name, None)
        # Clean up any loaded submodules starting with package_name
        submodules = [k for k in sys.modules if k.startswith(f"{package_name}.")]
        for sub in submodules:
            sys.modules.pop(sub, None)
