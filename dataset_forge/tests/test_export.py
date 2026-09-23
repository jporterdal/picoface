import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

from dataset_forge.config import ForgeConfig  # noqa: E402
from dataset_forge.export import (  # noqa: E402
    OUTPUT_DIR,
    PACKAGE_DIR,
    default_out_dir,
    export,
    read_manifest,
)
from dataset_forge.tests._configs import tiny_config  # noqa: E402
from picoface.datasets import load_dataset  # noqa: E402


def _arrays(out_dir, split):
    with np.load(out_dir / f"{split}.npz") as data:
        return data["images"], data["labels"]


def test_export_writes_both_splits_a_shared_classes_file_and_a_manifest(tmp_path):
    out_dir = export(tiny_config(), seed=0, out_dir=tmp_path / "out")

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "classes.json",
        "manifest.json",
        "test.npz",
        "train.npz",
    ]


def test_both_splits_load_with_the_configured_shape_classes_and_label_order(tmp_path):
    config = tiny_config()
    out_dir = export(config, seed=0, out_dir=tmp_path)

    for split, per_class in (("train", 8), ("test", 4)):
        data = load_dataset(out_dir / f"{split}.npz")
        assert data.images.shape == (3 * per_class, 28, 28, 1)
        assert data.images.dtype == np.uint8
        assert data.class_names == ["circle", "ring", "square"]
        assert data.labels.tolist() == [c for c in range(3) for _ in range(per_class)]


def test_a_subset_of_classes_is_numbered_from_zero_in_listed_order(tmp_path):
    out_dir = export(tiny_config(class_names=("star", "square")), seed=0, out_dir=tmp_path)

    data = load_dataset(out_dir / "train.npz")

    assert data.class_names == ["star", "square"]
    assert sorted(set(data.labels.tolist())) == [0, 1]


def test_unknown_classes_fail_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="hexagon"):
        export(tiny_config(class_names=("circle", "hexagon")), seed=0, out_dir=tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_same_config_and_seed_give_identical_exports(tmp_path):
    a = export(tiny_config(), seed=3, out_dir=tmp_path / "a")
    b = export(tiny_config(), seed=3, out_dir=tmp_path / "b")

    for split in ("train", "test"):
        assert load_dataset(a / f"{split}.npz") == load_dataset(b / f"{split}.npz")


def test_different_seeds_give_different_images(tmp_path):
    a = export(tiny_config(), seed=0, out_dir=tmp_path / "a")
    b = export(tiny_config(), seed=1, out_dir=tmp_path / "b")

    assert not np.array_equal(_arrays(a, "train")[0], _arrays(b, "train")[0])


def test_changing_the_training_count_leaves_the_test_split_unchanged(tmp_path):
    a = export(tiny_config(train_per_class=8), seed=0, out_dir=tmp_path / "a")
    b = export(tiny_config(train_per_class=12), seed=0, out_dir=tmp_path / "b")

    assert np.array_equal(_arrays(a, "test")[0], _arrays(b, "test")[0])


def test_adding_a_class_leaves_existing_classes_images_unchanged(tmp_path):
    a = export(tiny_config(class_names=("circle", "ring")), seed=0, out_dir=tmp_path / "a")
    b = export(tiny_config(class_names=("circle", "ring", "star")), seed=0, out_dir=tmp_path / "b")

    assert np.array_equal(_arrays(a, "train")[0], _arrays(b, "train")[0][:16])


def test_the_manifest_records_config_seed_and_versions(tmp_path):
    config = tiny_config()
    out_dir = export(config, seed=5, out_dir=tmp_path)

    manifest = read_manifest(out_dir)

    assert ForgeConfig.from_dict(manifest["config"]) == config
    assert manifest["seed"] == 5
    assert set(manifest["versions"]) == {"python", "numpy", "pillow"}
    assert "commit" in manifest["git"]
    assert json.loads((out_dir / "classes.json").read_text()) == {
        "0": "circle",
        "1": "ring",
        "2": "square",
    }


def test_the_default_output_location_is_ignored_by_git():
    out_dir = default_out_dir(tiny_config(), seed=0)

    assert out_dir.parent == OUTPUT_DIR
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(out_dir / "train.npz")], cwd=PACKAGE_DIR
    )
    assert result.returncode == 0
