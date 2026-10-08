"""Layer rules from CLAUDE.md §3: core/ and plots/ work without Qt and never use pyplot."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "larval_explorer"
PURE_LAYERS = ["core", "plots"]

# Runs in a fresh interpreter: Qt is made unimportable, every module in the
# layer is imported, and pyplot must not have been pulled in transitively.
PROBE = """
import importlib, pkgutil, sys

class BlockQt:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"PyQt5", "PyQt6", "PySide2", "PySide6"}:
            raise ImportError(f"{name} is blocked in the layering test")
        return None

sys.meta_path.insert(0, BlockQt())
package = importlib.import_module(sys.argv[1])
for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    importlib.import_module(info.name)
assert "matplotlib.pyplot" not in sys.modules, "matplotlib.pyplot was imported"
"""


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_layer_imports_without_qt_or_pyplot(layer):
    result = subprocess.run(
        [sys.executable, "-c", PROBE, f"larval_explorer.{layer}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_layer_source_has_no_pyplot_import(layer):
    offenders = []
    for path in sorted((PACKAGE_ROOT / layer).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module}.{alias.name}" for alias in node.names] + [node.module or ""]
            else:
                continue
            if any(name.startswith("matplotlib.pyplot") for name in names):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"pyplot imported in: {offenders}"
