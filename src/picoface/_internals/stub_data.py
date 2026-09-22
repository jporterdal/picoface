"""Internal, non-public synthetic stub dataset generator.

Used by this project's own development and test code (Phases 2-4) to
exercise classifier/generator/linkage plumbing before the real dataset
(Phase 5) exists. Not part of the public API — import only from
project-internal code and tests, never re-exported from `picoface`.
"""

import numpy as np

from picoface.datasets import Dataset

# Background/foreground gray levels for `kind="shapes"`, and the fraction of
# pixels every figure covers. Every shapes image has exactly the same number
# of foreground pixels, so its pixel-value distribution — and with it mean
# brightness, contrast, and any other order-free statistic — is the same for
# every class; only the pixels' arrangement tells the classes apart.
_SHAPE_BACKGROUND = 64
_SHAPE_FOREGROUND = 192
_SHAPE_AREA_FRACTION = 0.2

_NOISE_AMPLITUDE = 20


def _shape_scores(name: str, dy: np.ndarray, dx: np.ndarray, radius: float) -> np.ndarray:
    """Per-pixel score whose lowest-scoring pixels form figure `name`.

    `dy`/`dx` are offsets from the figure's centre and `radius` its nominal
    size; taking the k lowest scores rasterizes the figure at exactly k pixels.
    """
    ady, adx = np.abs(dy), np.abs(dx)
    tie_break = 1e-3 * (ady + adx)  # grow each figure outward from its centre
    if name == "square":
        return np.maximum(ady, adx) + tie_break
    if name == "cross":
        return np.minimum(ady, adx) + 1e-2 * np.maximum(ady, adx)
    if name == "ring":
        return np.abs(np.hypot(dy, dx) - radius) + tie_break
    if name == "hbar":
        return ady + 1e-2 * adx
    if name == "vbar":
        return adx + 1e-2 * ady
    if name == "diamond":
        return ady + adx + 1e-3 * np.maximum(ady, adx)
    if name == "circle":
        return np.hypot(dy, dx) + 1e-3 * np.maximum(ady, adx)
    raise ValueError(f"unknown stub shape {name!r}")


# Ordered so that small class counts get the most distinct figures.
_SHAPE_ORDER = ["square", "cross", "ring", "hbar", "vbar", "diamond", "circle"]


def _render_shapes(
    rng: np.random.Generator, num_classes: int, n_per_class: int, height: int, width: int
) -> np.ndarray:
    """(num_classes, n_per_class, height, width) uint8 figures at randomly
    jittered positions, each covering exactly the same number of pixels."""
    area = max(1, round(_SHAPE_AREA_FRACTION * height * width))
    radius = 0.25 * min(height, width)
    rows, cols = np.mgrid[0:height, 0:width].astype(float)
    figures = np.full((num_classes, n_per_class, height, width), _SHAPE_BACKGROUND, np.uint8)

    for class_idx in range(num_classes):
        name = _SHAPE_ORDER[class_idx]
        for i in range(n_per_class):
            centre_y = (height - 1) / 2 + rng.uniform(-height / 8, height / 8)
            centre_x = (width - 1) / 2 + rng.uniform(-width / 8, width / 8)
            scores = _shape_scores(name, rows - centre_y, cols - centre_x, radius)
            foreground = np.argsort(scores, axis=None, kind="stable")[:area]
            figures[class_idx, i].flat[foreground] = _SHAPE_FOREGROUND

    return figures


def make_stub_dataset(
    n_per_class: int = 8,
    height: int = 16,
    width: int = 16,
    channels: int = 3,
    class_names: list[str] | None = None,
    seed: int = 0,
    kind: str = "brightness",
) -> Dataset:
    """Generate a small, in-memory, arbitrary-shaped synthetic dataset.

    `kind="brightness"` (the default) fills each class's images with a
    class-distinct constant value plus noise, so classes are trivially
    separable — for plumbing tests that do not need difficulty.

    `kind="shapes"` draws one simple figure per class (square, cross, ring,
    ...) at a randomly jittered position, with every image covering the same
    number of foreground pixels at the same gray levels, plus noise. Mean
    brightness and other global statistics are therefore uninformative;
    only spatial arrangement separates the classes. Supports up to seven
    classes and commits to no real shape taxonomy (Phase 5 owns that).
    """
    if class_names is None:
        class_names = ["class_a", "class_b"]
    if len(class_names) < 2:
        raise ValueError("make_stub_dataset requires at least two classes")
    if kind not in ("brightness", "shapes"):
        raise ValueError(f"kind must be 'brightness' or 'shapes', got {kind!r}")
    if kind == "shapes" and len(class_names) > len(_SHAPE_ORDER):
        raise ValueError(
            f"kind='shapes' supports at most {len(_SHAPE_ORDER)} classes, "
            f"got {len(class_names)}"
        )

    rng = np.random.default_rng(seed)
    num_classes = len(class_names)
    n = n_per_class * num_classes

    images = np.empty((n, height, width, channels), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)

    if kind == "shapes":
        figures = _render_shapes(rng, num_classes, n_per_class, height, width)

    for class_idx in range(num_classes):
        if kind == "shapes":
            base = figures[class_idx][..., np.newaxis].astype(np.int64)
        else:
            base = int(255 * (class_idx + 1) / (num_classes + 1))
        noise = rng.integers(
            -_NOISE_AMPLITUDE,
            _NOISE_AMPLITUDE + 1,
            size=(n_per_class, height, width, channels),
        )
        class_images = np.clip(base + noise, 0, 255).astype(np.uint8)

        start = class_idx * n_per_class
        end = start + n_per_class
        images[start:end] = class_images
        labels[start:end] = class_idx

    return Dataset(images=images, labels=labels, class_names=class_names)
