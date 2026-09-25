import json
import warnings

import numpy as np
import pytest

from picoface.datasets import Dataset, DatasetError, DatasetWarning, load_dataset
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


# --- validation ----------------------------------------------------------------

_IMAGES = np.zeros((4, 8, 8, 1), np.uint8)
_LABELS = np.array([0, 1, 0, 1])
_CLASSES = {"0": "a", "1": "b"}


def _write_raw(tmp_path, arrays, classes=_CLASSES, classes_text=None):
    """Write an arbitrary (possibly malformed) bundle and return the .npz path."""
    npz_path = tmp_path / "bundle.npz"
    np.savez(npz_path, **arrays)
    if classes_text is not None:
        (tmp_path / "classes.json").write_text(classes_text)
    elif classes is not None:
        (tmp_path / "classes.json").write_text(json.dumps(classes))
    return npz_path


def _load_error(tmp_path, arrays=None, **kwargs):
    arrays = {"images": _IMAGES, "labels": _LABELS} if arrays is None else arrays
    npz_path = _write_raw(tmp_path, arrays, **kwargs)
    with pytest.raises(DatasetError) as info:
        load_dataset(npz_path)
    return str(info.value)


def test_error_types_are_value_error_and_user_warning():
    assert issubclass(DatasetError, ValueError)
    assert issubclass(DatasetWarning, UserWarning)


# Dataset invariants, constructed directly -------------------------------------


@pytest.mark.parametrize(
    "images,labels,class_names,expected",
    [
        (np.zeros((4, 8, 8), np.uint8), _LABELS, ["a", "b"], "(4, 8, 8)"),
        (np.zeros((4, 8, 8, 4), np.uint8), _LABELS, ["a", "b"], "4 channels"),
        (np.zeros((4, 8, 8, 1), np.float32), _LABELS, ["a", "b"], "float32"),
        (np.zeros((0, 8, 8, 1), np.uint8), _LABELS[:0], ["a", "b"], "no images"),
        (_IMAGES, _LABELS[:3], ["a", "b"], "3 entries"),
        (_IMAGES, _LABELS[:, None], ["a", "b"], "(4, 1)"),
        (_IMAGES, _LABELS.astype(float), ["a", "b"], "float64"),
        (_IMAGES, [0, 1, 0, 1], ["a", "b"], "numpy array"),
        (_IMAGES, np.array([0, 1, 2, 5]), ["a", "b"], "0 to 1"),
        (_IMAGES, np.array([0, 1, -1, 1]), ["a", "b"], "-1"),
        (_IMAGES, np.zeros(4, int), ["a"], "at least 2"),
        (_IMAGES, _LABELS, ["a", "a"], "'a' more than once"),
        (_IMAGES, _LABELS, ["a", ""], "non-empty string"),
        (_IMAGES, _LABELS, "ab", "list of class-name strings"),
    ],
)
def test_dataset_rejects_invalid_arrays(images, labels, class_names, expected):
    with pytest.raises(DatasetError) as info:
        Dataset(images=images, labels=labels, class_names=class_names)

    assert expected in str(info.value)


def test_out_of_range_label_reports_value_count_and_range():
    labels = np.array([0, 5, 5, 1])

    with pytest.raises(DatasetError) as info:
        Dataset(images=_IMAGES, labels=labels, class_names=["a", "b", "c"])

    message = str(info.value)
    assert "5 (in 2 labels)" in message
    assert "0 to 2" in message


def test_directly_built_dataset_missing_a_class_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(images=_IMAGES, labels=np.zeros(4, np.int64), class_names=["a", "b"])


def test_height_width_and_num_classes():
    stub = make_stub_dataset(
        n_per_class=2, height=28, width=20, channels=1, class_names=["x", "y", "z"]
    )

    assert stub.height == 28
    assert stub.width == 20
    assert stub.num_classes == 3


def test_properties_are_read_from_a_loaded_bundle(tmp_path):
    stub = make_stub_dataset(
        n_per_class=2, height=28, width=28, channels=1, class_names=["x", "y", "z"]
    )
    loaded = load_dataset(_write_bundle(tmp_path, stub))

    assert (loaded.height, loaded.width, loaded.num_classes) == (28, 28, 3)


def test_repr_reports_the_labels_own_shape():
    data = Dataset(images=_IMAGES, labels=_LABELS.astype(np.int32), class_names=["a", "b"])

    assert "labels: int32[4]" in repr(data)
    assert repr(data) == (
        "Dataset(images: uint8[4,8,8,1], labels: int32[4], classes=['a', 'b'])"
    )


# load_dataset() file checks ----------------------------------------------------


def test_missing_npz_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        load_dataset(tmp_path / "nope.npz")

    assert "nope.npz" in str(info.value)


def test_missing_classes_json_names_the_folder(tmp_path):
    npz_path = _write_raw(tmp_path, {"images": _IMAGES, "labels": _LABELS}, classes=None)

    with pytest.raises(FileNotFoundError) as info:
        load_dataset(npz_path)

    message = str(info.value)
    assert "classes.json" in message
    assert "same folder" in message
    assert str(tmp_path.absolute()) in message


def test_npy_file_is_rejected(tmp_path):
    npy_path = tmp_path / "bundle.npy"
    np.save(npy_path, _IMAGES)
    (tmp_path / "classes.json").write_text(json.dumps(_CLASSES))

    with pytest.raises(DatasetError) as info:
        load_dataset(npy_path)

    assert "bundle.npy" in str(info.value)
    assert ".npz" in str(info.value)


def test_non_numpy_file_is_rejected(tmp_path):
    path = tmp_path / "bundle.npz"
    path.write_text("not numpy")

    with pytest.raises(DatasetError) as info:
        load_dataset(path)

    assert "bundle.npz" in str(info.value)


def test_missing_array_is_named_and_present_arrays_listed(tmp_path):
    message = _load_error(tmp_path, {"images": _IMAGES, "lables": _LABELS})

    assert "bundle.npz" in message
    assert "'labels'" in message
    assert "'lables'" in message


def test_ragged_images_explain_same_size_rule(tmp_path):
    ragged = np.empty(2, dtype=object)
    ragged[0], ragged[1] = np.zeros((8, 8, 1)), np.zeros((9, 9, 1))

    message = _load_error(tmp_path, {"images": ragged, "labels": _LABELS[:2]})

    assert "bundle.npz" in message
    assert "same height, width, and channel count" in message


@pytest.mark.parametrize(
    "classes,expected",
    [
        ({"0": "a", "2": "b"}, "Missing: '1'"),
        ({"0": "a", "1": "b", "x": "c"}, "'x'"),
        (["a", "b"], "JSON list"),
        ({"0": "a", "1": 5}, "non-empty string"),
        ({"0": "a", "1": "a"}, "'a' more than once"),
        ({"0": "a"}, "at least 2"),
    ],
)
def test_classes_json_problems_name_the_file(tmp_path, classes, expected):
    message = _load_error(tmp_path, classes=classes)

    assert "classes.json" in message
    assert expected in message


def test_invalid_json_is_reported(tmp_path):
    message = _load_error(tmp_path, classes_text='{"0": "a", "1": }')

    assert "classes.json" in message
    assert "not valid JSON" in message


def test_array_errors_name_the_npz_and_keep_the_cause(tmp_path):
    npz_path = _write_raw(
        tmp_path, {"images": np.zeros((4, 8, 8), np.uint8), "labels": _LABELS}
    )

    with pytest.raises(DatasetError) as info:
        load_dataset(npz_path)

    message = str(info.value)
    assert "bundle.npz" in message
    assert "(4, 8, 8)" in message
    assert "np.newaxis" in message
    assert isinstance(info.value.__cause__, DatasetError)


def test_errors_are_caught_as_value_error(tmp_path):
    npz_path = _write_raw(tmp_path, {"images": _IMAGES.astype(float), "labels": _LABELS})

    with pytest.raises(ValueError):
        load_dataset(npz_path)


# load_dataset() conversions ------------------------------------------------------


def test_integer_images_within_range_convert_to_uint8(tmp_path):
    images = np.arange(4 * 8 * 8).reshape(4, 8, 8, 1) % 256
    npz_path = _write_raw(
        tmp_path, {"images": images.astype(np.int64), "labels": _LABELS.astype(np.uint8)}
    )

    loaded = load_dataset(npz_path)

    assert loaded.images.dtype == np.uint8
    assert np.array_equal(loaded.images, images)
    assert loaded.labels.dtype == np.int64


def test_integer_images_out_of_range_report_min_and_max(tmp_path):
    images = np.zeros((4, 8, 8, 1), np.int16)
    images[0, 0, 0, 0] = 300
    images[1, 0, 0, 0] = -2

    message = _load_error(tmp_path, {"images": images, "labels": _LABELS})

    assert "-2" in message and "300" in message


def test_float_labels_are_rejected_not_converted(tmp_path):
    message = _load_error(tmp_path, {"images": _IMAGES, "labels": _LABELS.astype(float)})

    assert "float64" in message


# load_dataset() empty-class warning ------------------------------------------------


def test_class_with_no_images_warns_but_loads(tmp_path):
    npz_path = _write_raw(
        tmp_path, {"images": _IMAGES, "labels": _LABELS}, classes={"0": "a", "1": "b", "2": "c"}
    )

    with pytest.warns(DatasetWarning, match="'c'"):
        loaded = load_dataset(npz_path)

    assert loaded.class_names == ["a", "b", "c"]


def test_well_formed_bundle_loads_without_warnings(tmp_path):
    npz_path = _write_raw(tmp_path, {"images": _IMAGES, "labels": _LABELS})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_dataset(npz_path)
