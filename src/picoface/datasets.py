"""Dataset interchange format: loading and representing image datasets.

A dataset bundle on disk is an `.npz` file (`images`: uint8 N×H×W×C,
`labels`: int N) plus a companion `classes.json` mapping label index to
class name. `load_dataset()` is the sole public entry point for reading
one; it never assumes an image size or class count.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dataset:
    """A ready-to-use image dataset: images, integer labels, and class names."""

    images: np.ndarray
    labels: np.ndarray
    class_names: list[str]

    def __repr__(self) -> str:
        n, h, w, c = self.images.shape
        return (
            f"Dataset(images: {self.images.dtype}[{n},{h},{w},{c}], "
            f"labels: {self.labels.dtype}[{n}], "
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


def load_dataset(path: str | Path) -> Dataset:
    """Load a dataset bundle from `path`.

    `path` is the `.npz` file; a companion `classes.json` is expected
    in the same directory.
    """
    npz_path = Path(path)
    classes_path = npz_path.with_name("classes.json")

    with np.load(npz_path) as data:
        images = data["images"]
        labels = data["labels"]

    with open(classes_path) as f:
        classes_by_index = json.load(f)

    class_names = [classes_by_index[str(i)] for i in range(len(classes_by_index))]

    return Dataset(images=images, labels=labels, class_names=class_names)
