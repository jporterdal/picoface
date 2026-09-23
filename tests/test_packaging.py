"""The student-facing package stays independent of Dataset Forge."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_importing_every_picoface_module_imports_no_dataset_forge_code():
    code = (
        "import importlib, pkgutil, sys, picoface\n"
        "for m in pkgutil.walk_packages(picoface.__path__, 'picoface.'):\n"
        "    importlib.import_module(m.name)\n"
        "print(sorted(n for n in sys.modules if n.split('.')[0] == 'dataset_forge'))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_picoface_does_not_declare_the_forges_dependencies():
    # A plain text check: tomllib needs Python 3.11, and picoface supports 3.10.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text().lower()

    assert "pillow" not in pyproject
