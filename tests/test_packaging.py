"""The student-facing package stays independent of Dataset Forge."""

import re
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


def _normalized(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement).lower()


def test_picoface_does_not_carry_the_forges_exact_pins():
    # A plain text parse: tomllib needs Python 3.11, and picoface supports 3.10.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    dependencies = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.M | re.S).group(1)
    declared = {_normalized(entry) for entry in re.findall(r'"([^"]+)"', dependencies)}
    requirements = (REPO_ROOT / "dataset_forge" / "requirements.txt").read_text()
    pins = {
        _normalized(line)
        for line in requirements.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }

    assert declared, "no dependencies found in pyproject.toml"
    assert pins, "no exact pins found in dataset_forge/requirements.txt"
    assert not declared & pins
