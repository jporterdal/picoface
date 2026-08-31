"""Internal, non-public synthetic stub dataset generator.

Used by this project's own development and test code (Phases 2-4) to
exercise classifier/generator/linkage plumbing before the real dataset
(Phase 5) exists. Not part of the public API — import only from
project-internal code and tests, never re-exported from `picoface`.
"""

import numpy as np

from picoface.datasets import Dataset


def make_stub_dataset(
    n_per_class: int = 8,
    height: int = 16,
    width: int = 16,
    channels: int = 3,
    class_names: list[str] | None = None,
    seed: int = 0,
) -> Dataset:
    """Generate a small, in-memory, arbitrary-shaped synthetic dataset.

    Produces `n_per_class` images per class, filled with a class-distinct
    constant value plus noise, so classes are trivially separable without
    encoding any real content decisions.
    """
    if class_names is None:
        class_names = ["class_a", "class_b"]
    if len(class_names) < 2:
        raise ValueError("make_stub_dataset requires at least two classes")

    rng = np.random.default_rng(seed)
    num_classes = len(class_names)
    n = n_per_class * num_classes

    images = np.empty((n, height, width, channels), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)

    for class_idx in range(num_classes):
        base_value = int(255 * (class_idx + 1) / (num_classes + 1))
        noise = rng.integers(-20, 21, size=(n_per_class, height, width, channels))
        class_images = np.clip(base_value + noise, 0, 255).astype(np.uint8)

        start = class_idx * n_per_class
        end = start + n_per_class
        images[start:end] = class_images
        labels[start:end] = class_idx

    return Dataset(images=images, labels=labels, class_names=class_names)
