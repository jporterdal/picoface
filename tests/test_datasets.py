import json

import numpy as np

from picoface.datasets import Dataset, load_dataset
from picoface._internals.stub_data import make_stub_dataset


def _write_bundle(tmp_path, dataset: Dataset, name: str = "bundle"):
    npz_path = tmp_path / f"{name}.npz"
    classes_path = tmp_path / "classes.json"

    np.savez(npz_path, images=dataset.images, labels=dataset.labels)
    classes_by_index = {str(i): name for i, name in enumerate(dataset.class_names)}
    classes_path.write_text(json.dumps(classes_by_index))

    return npz_path


def test_round_trip(tmp_path):
    stub = make_stub_dataset(n_per_class=4, height=8, width=8, channels=3)
    npz_path = _write_bundle(tmp_path, stub)

    loaded = load_dataset(npz_path)

    assert np.array_equal(loaded.images, stub.images)
    assert np.array_equal(loaded.labels, stub.labels)
    assert loaded.class_names == stub.class_names


def test_size_and_class_agnostic(tmp_path):
    small = make_stub_dataset(
        n_per_class=3, height=8, width=8, channels=1, class_names=["a", "b"]
    )
    small_dir = tmp_path / "small"
    small_dir.mkdir()
    small_npz = _write_bundle(small_dir, small)

    large = make_stub_dataset(
        n_per_class=5,
        height=32,
        width=24,
        channels=3,
        class_names=["x", "y", "z"],
    )
    large_dir = tmp_path / "large"
    large_dir.mkdir()
    large_npz = _write_bundle(large_dir, large)

    loaded_small = load_dataset(small_npz)
    loaded_large = load_dataset(large_npz)

    assert loaded_small.images.shape == (6, 8, 8, 1)
    assert loaded_small.class_names == ["a", "b"]

    assert loaded_large.images.shape == (15, 32, 24, 3)
    assert loaded_large.class_names == ["x", "y", "z"]


def test_repr_shows_shape_not_pixels(tmp_path):
    stub = make_stub_dataset(n_per_class=4, height=8, width=8, channels=3)
    npz_path = _write_bundle(tmp_path, stub)
    loaded = load_dataset(npz_path)

    text = repr(loaded)

    assert "8" in text and "3" in text
    assert str(loaded.images.dtype) in text
    for value in np.unique(loaded.images):
        assert f"[{value}" not in text


def test_load_dataset_and_stub_are_interchangeable(tmp_path):
    stub = make_stub_dataset(n_per_class=4, height=8, width=8, channels=3)
    npz_path = _write_bundle(tmp_path, stub)
    loaded = load_dataset(npz_path)

    assert type(loaded) is type(stub) is Dataset
    for field in ("images", "labels", "class_names"):
        assert hasattr(loaded, field)
        assert hasattr(stub, field)
