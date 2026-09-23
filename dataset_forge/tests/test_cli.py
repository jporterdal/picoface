"""The command-line entry points, run as an instructor would run them."""

import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import subprocess  # noqa: E402
import sys  # noqa: E402

from dataset_forge.export import PACKAGE_DIR, export  # noqa: E402
from dataset_forge.tests._configs import default_config, tiny_config  # noqa: E402

REPO_ROOT = PACKAGE_DIR.parent


def _run(*args):
    return subprocess.run(
        [sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_export_command_writes_a_validated_export(tmp_path):
    # All seven classes: a few classes at a handful of images each is too
    # small a sample for the mean-brightness check to pass reliably.
    config_path = tmp_path / "config.json"
    default_config(train_per_class=120, test_per_class=60).save(config_path)
    out_dir = tmp_path / "out"

    result = _run("-m", "dataset_forge", "--config", str(config_path), "--out", str(out_dir))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validation PASSED" in result.stdout
    assert {p.name for p in out_dir.iterdir()} >= {"train.npz", "test.npz", "classes.json"}


def test_export_command_rejects_an_unknown_class_before_writing(tmp_path):
    config_path = tmp_path / "config.json"
    tiny_config(class_names=("circle", "hexagon")).save(config_path)

    result = _run(
        "-m", "dataset_forge", "--config", str(config_path), "--out", str(tmp_path / "out")
    )

    assert result.returncode != 0
    assert "hexagon" in result.stderr
    assert not (tmp_path / "out").exists()


def test_validate_command_exits_non_zero_on_a_failed_check(tmp_path):
    config = tiny_config(
        class_names=("circle", "ring"),
        train_per_class=40,
        test_per_class=20,
        background_range=(40, 40),
        min_contrast=210,
        noise_sigma=0.0,
    )
    out_dir = export(config, seed=0, out_dir=tmp_path)

    result = _run("-m", "dataset_forge.validate", str(out_dir))

    assert result.returncode == 1
    assert "[FAIL] mean_brightness" in result.stdout


def test_the_forge_tests_skip_cleanly_without_pillow():
    # Hide Pillow from a fresh interpreter and run this folder's tests there.
    code = (
        "import sys; sys.modules['PIL'] = None; import pytest; "
        "sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', 'dataset_forge/tests']))"
    )

    result = _run("-c", code)

    # Every module skips, so pytest collects no tests; in a full run, the
    # student-package tests still run and pass.
    assert result.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, result.stdout
    assert "skipped" in result.stdout
    assert "error" not in result.stdout.lower() and "failed" not in result.stdout.lower()


@pytest.mark.parametrize("module", ["dataset_forge.contact_sheet"])
def test_contact_sheet_command_writes_an_image(tmp_path, module):
    config_path = tmp_path / "config.json"
    tiny_config().save(config_path)

    result = _run(
        "-m", module, "--config", str(config_path), "--out", str(tmp_path / "sheet.png")
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "sheet.png").stat().st_size > 0
