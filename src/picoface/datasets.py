"""Dataset interchange format: loading and representing image datasets.

A dataset bundle on disk is an `.npz` file (`images`: uint8 N×H×W×C with C
either 1 for grayscale or 3 for RGB, `labels`: int N) plus a companion
`classes.json` in the same folder, mapping each label number ("0", "1", ...)
to its class name. `load_dataset()` is the sole public entry point for reading
one; it never assumes an image size or class count.

Every `Dataset` checks its own arrays when it is created, however it is
created, and raises `DatasetError` if they break the format. `load_dataset()`
also checks the files themselves and names the file each error is about.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from picoface._internals.image_checks import check_image_array

__all__ = ["Dataset", "DatasetError", "DatasetWarning", "load_dataset"]

_MIN_CLASSES = 2
_MAX_BAD_LABELS_SHOWN = 3


class DatasetError(ValueError):
    """Raised when a dataset bundle or `Dataset` breaks the interchange format.

    The message says what was expected, what was found, and how to fix it.
    """


class DatasetWarning(UserWarning):
    """Issued by `load_dataset()` when a class in `classes.json` has no images."""


@dataclass(frozen=True)
class Dataset:
    """A ready-to-use image dataset: images, integer labels, and class names.

    `images` is a uint8 image array shaped (count, height, width, channels),
    with 1 channel for grayscale or 3 for RGB. `labels` holds one class index
    per image, and `class_names[i]` is the name of class `i`. `height`,
    `width`, and `num_classes` read those sizes without any indexing.
    """

    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]

    def __post_init__(self) -> None:
        check_image_array(self.images, name="images", batch=True, error_cls=DatasetError)
        n = self.images.shape[0]
        if n == 0:
            raise DatasetError(
                f"images has shape {self.images.shape}, which holds no images; "
                f"a dataset needs at least one image."
            )
        _check_labels_shape(self.labels, n)
        _check_class_names(self.class_names, "class_names")
        _check_labels_in_range(self.labels, self.class_names)

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.images.shape[1])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.images.shape[2])

    @property
    def num_classes(self) -> int:
        """How many classes the dataset has (the length of `class_names`)."""
        return len(self.class_names)

    def __repr__(self) -> str:
        n, h, w, c = self.images.shape
        labels_shape = ",".join(str(size) for size in self.labels.shape)
        return (
            f"Dataset(images: {self.images.dtype}[{n},{h},{w},{c}], "
            f"labels: {self.labels.dtype}[{labels_shape}], "
            f"classes={self.class_names!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dataset):
            return NotImplemented
        return (
            np.array_equal(self.images, other.images)
            and np.array_equal(self.labels, other.labels)
            and self.class_names == other.class_names
        )


def _check_labels_shape(labels, n: int) -> None:
    """Labels must be a 1-D integer array with one entry per image."""
    if not isinstance(labels, np.ndarray):
        raise DatasetError(
            f"labels must be a numpy array, but got {type(labels).__name__}. "
            f"Convert it with labels = np.asarray(labels)"
        )
    if labels.ndim != 1:
        message = (
            f"labels has shape {labels.shape}, but it must be one-dimensional, "
            f"with exactly one label per image: shape ({n},)."
        )
        if labels.size == n:
            message += " Flatten it with labels = labels.reshape(-1)"
        raise DatasetError(message)
    if labels.shape[0] != n:
        raise DatasetError(
            f"labels has {labels.shape[0]:,} entries but images has {n:,}; "
            f"each image needs exactly one label."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        message = (
            f"labels has dtype {labels.dtype}, but labels must be whole-number "
            f"class indices (an integer dtype)."
        )
        if np.issubdtype(labels.dtype, np.floating):
            message += (
                " If every value is a whole number, convert with "
                "labels = labels.astype(np.int64)"
            )
        raise DatasetError(message)


def _check_class_names(class_names, source: str) -> None:
    """Class names must be at least two distinct, non-empty strings."""
    if not isinstance(class_names, (list, tuple)):
        raise DatasetError(
            f"{source} must be a list of class-name strings, but got "
            f"{type(class_names).__name__}."
        )
    for index, name in enumerate(class_names):
        if not isinstance(name, str) or not name:
            raise DatasetError(
                f"{source} entry {index} is {name!r}, but every class name must be "
                f"a non-empty string."
            )
    if len(class_names) < _MIN_CLASSES:
        raise DatasetError(
            f"{source} has {len(class_names)} class(es) ({list(class_names)!r}), but "
            f"a dataset needs at least {_MIN_CLASSES} classes to tell apart."
        )
    indices_by_name: dict[str, list[int]] = {}
    for index, name in enumerate(class_names):
        indices_by_name.setdefault(name, []).append(index)
    for name, indices in indices_by_name.items():
        if len(indices) > 1:
            where = ", ".join(str(i) for i in indices)
            raise DatasetError(
                f"{source} lists the class {name!r} more than once (label numbers "
                f"{where}); "
                f"every class needs a different name."
            )


def _check_labels_in_range(labels: np.ndarray, class_names) -> None:
    """Every label must be the index of one of the classes."""
    k = len(class_names)
    out_of_range = (labels < 0) | (labels >= k)
    if not out_of_range.any():
        return
    bad_values, counts = np.unique(labels[out_of_range], return_counts=True)
    shown = ", ".join(
        f"{value} (in {count:,} label{'s' if count != 1 else ''})"
        for value, count in zip(
            bad_values[:_MAX_BAD_LABELS_SHOWN].tolist(),
            counts[:_MAX_BAD_LABELS_SHOWN].tolist(),
        )
    )
    if len(bad_values) > _MAX_BAD_LABELS_SHOWN:
        shown += f", and {len(bad_values) - _MAX_BAD_LABELS_SHOWN} other value(s)"
    raise DatasetError(
        f"labels contains {shown}, but there are {k} classes "
        f"({list(class_names)!r}), so every label must be 0 to {k - 1}."
    )


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset bundle from `path`.

    `path` is the `.npz` file; a companion `classes.json` is expected in the
    same folder. Image and label arrays with a whole-number dtype are converted
    to uint8 and int64 when no value changes; nothing else is converted.

    Raises `FileNotFoundError` if either file is missing, and `DatasetError`
    (a `ValueError`) naming the file and the problem if a file breaks the
    format. Issues a `DatasetWarning` if a class has no images.
    """
    npz_path = Path(path)
    images, labels = _read_npz(npz_path)
    class_names = _read_classes_json(npz_path)

    images = _to_uint8_if_lossless(images, npz_path)
    if np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64, copy=False)

    try:
        dataset = Dataset(images=images, labels=labels, class_names=class_names)
    except DatasetError as err:
        raise DatasetError(f"{npz_path}: {err}") from err

    _warn_about_empty_classes(dataset, npz_path)
    return dataset


def _read_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `images` and `labels` from the bundle's `.npz` file."""
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"load_dataset() could not find a file at {npz_path}. Check the path "
            f"and the file name, including the .npz ending."
        )

    try:
        loaded = np.load(npz_path)
    except (ValueError, OSError) as err:
        raise DatasetError(
            f"{npz_path}: could not be read as an .npz dataset bundle ({err}). "
            f"Save a bundle with np.savez(path, images=images, labels=labels)"
        ) from err

    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise DatasetError(
            f"{npz_path}: holds a single array (an .npy file), but a dataset bundle "
            f"is an .npz file holding both images and labels. Save one with "
            f"np.savez(path, images=images, labels=labels)"
        )

    with loaded as data:
        present = list(data.files)
        for key in ("images", "labels"):
            if key not in present:
                raise DatasetError(
                    f"{npz_path}: has no {key!r} array; a dataset bundle needs both "
                    f"'images' and 'labels'. The arrays in this file are: {present!r}."
                )
        arrays = []
        for key in ("images", "labels"):
            try:
                arrays.append(data[key])
            except ValueError as err:
                raise DatasetError(
                    f"{npz_path}: {key!r} is not a regular array of numbers (numpy "
                    f"saved it as separate Python objects). This usually means the "
                    f"items have different sizes. Every image must have the same "
                    f"height, width, and channel count."
                ) from err

    return arrays[0], arrays[1]


def _read_classes_json(npz_path: Path) -> list[str]:
    """Read and check the companion `classes.json`, returning class names in label order."""
    classes_path = npz_path.with_name("classes.json")
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"load_dataset() needs a classes.json file in the same folder as "
            f"{npz_path.name} ({npz_path.parent.absolute()}), but there isn't one. "
            f'classes.json maps each label number to its class name, like '
            f'{{"0": "circle", "1": "square"}}.'
        )

    try:
        with open(classes_path, encoding="utf-8") as f:
            classes_by_index = json.load(f)
    except json.JSONDecodeError as err:
        raise DatasetError(
            f"{classes_path}: is not valid JSON ({err.msg} at line {err.lineno}, "
            f"column {err.colno})."
        ) from err
    except UnicodeDecodeError as err:
        raise DatasetError(f"{classes_path}: is not a UTF-8 text file.") from err

    if not isinstance(classes_by_index, dict):
        raise DatasetError(
            f"{classes_path}: holds a JSON {_json_type_name(classes_by_index)}, but it "
            f'must be a JSON object mapping label numbers to class names, like '
            f'{{"0": "circle", "1": "square"}}.'
        )

    k = len(classes_by_index)
    expected = [str(i) for i in range(k)]
    found = list(classes_by_index)
    if set(found) != set(expected):
        missing = [key for key in expected if key not in classes_by_index]
        extra = [key for key in found if key not in expected]
        message = (
            f'{classes_path}: expected keys "0" to "{k - 1}" (one per class), but '
            f"found {', '.join(repr(key) for key in found)}."
        )
        if missing:
            message += f" Missing: {', '.join(repr(key) for key in missing)}."
        if extra:
            message += f" Not expected: {', '.join(repr(key) for key in extra)}."
        message += " Number the classes 0, 1, 2, ... with no gaps."
        raise DatasetError(message)

    class_names = [classes_by_index[key] for key in expected]
    _check_class_names(class_names, str(classes_path))
    return class_names


def _json_type_name(value) -> str:
    """The JSON name of a parsed JSON value's type."""
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "true/false value"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _to_uint8_if_lossless(images: np.ndarray, npz_path: Path) -> np.ndarray:
    """Convert whole-number images within 0-255 to uint8; leave anything else as is."""
    if images.dtype == np.uint8 or not np.issubdtype(images.dtype, np.integer):
        return images
    if images.size and (images.min() < 0 or images.max() > 255):
        raise DatasetError(
            f"{npz_path}: images holds whole numbers from {images.min()} to "
            f"{images.max()}, but pixel values must be from 0 to 255."
        )
    return images.astype(np.uint8)


def _warn_about_empty_classes(dataset: Dataset, npz_path: Path) -> None:
    """Warn, naming each class that no label in the bundle refers to."""
    counts = np.bincount(dataset.labels, minlength=dataset.num_classes)
    empty = [name for name, count in zip(dataset.class_names, counts) if count == 0]
    if empty:
        warnings.warn(
            DatasetWarning(
                f"{npz_path}: no images are labeled {', '.join(repr(n) for n in empty)}, "
                f"although classes.json lists "
                f"{'it' if len(empty) == 1 else 'them'}. Models trained on this "
                f"data will never learn {'that class' if len(empty) == 1 else 'those classes'}."
            ),
            stacklevel=3,
        )
