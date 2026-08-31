"""Build, train, evaluate, and run inference with a small CNN image classifier.

No `nn.Module` authoring, no hand-written training loop: `build_classifier()`
(or `build_classifier_from_shape()`) constructs a model sized to your data,
`train()` runs the full training loop, and `evaluate()`/`predict()` report
results — all via plain function calls.
"""

import numpy as np
import torch

from picoface._internals.classifier_internals import (
    TrainingHistory,
    _build_classifier,
    _forward,
    _train_loop,
)
from picoface.datasets import Dataset

__all__ = [
    "ShapeError",
    "TrainingHistory",
    "build_classifier",
    "build_classifier_from_shape",
    "train",
    "evaluate",
    "predict",
]


class ShapeError(ValueError):
    """Raised when an input shape or class count doesn't match what's expected."""


def build_classifier(data: Dataset):
    """Build a CNN classifier sized for `data`'s class count and image shape.

    Returns a trainable model object — no `nn.Module` code, no manual shape
    derivation required.
    """
    num_classes = len(data.class_names)
    input_shape = tuple(data.images.shape[1:])
    model = _build_classifier(num_classes, input_shape, ShapeError)
    model.class_names = list(data.class_names)
    return model


def build_classifier_from_shape(num_classes: int, input_shape: tuple[int, int, int]):
    """Build a CNN classifier from an explicit class count and image shape.

    Use this when no `Dataset` is yet available. Equivalent to what
    `build_classifier(data)` would return for a `Dataset` with matching
    class count and image shape.
    """
    return _build_classifier(num_classes, tuple(input_shape), ShapeError)


def _check_class_count(model, data: Dataset) -> None:
    if len(data.class_names) != model.num_classes:
        raise ShapeError(
            f"data has {len(data.class_names)} class(es) "
            f"({data.class_names!r}), but this model expects "
            f"{model.num_classes} class(es)."
        )


def train(
    model,
    data: Dataset,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> TrainingHistory:
    """Train `model` on `data` for `epochs` epochs, returning a training history.

    Runs the full training loop (forward pass, loss, backward pass, optimizer
    step, epoch iteration) internally — no training loop to write.
    """
    _check_class_count(model, data)
    model.class_names = list(data.class_names)
    return _train_loop(
        model,
        data.images,
        data.labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shape_error_cls=ShapeError,
    )


def evaluate(model, data: Dataset) -> float:
    """Return classification accuracy (0 to 1) of `model` on `data`."""
    _check_class_count(model, data)
    model.eval()

    with torch.no_grad():
        logits = _forward(model, data.images, ShapeError)
        predicted = logits.argmax(dim=1).numpy()

    labels = np.asarray(data.labels)
    return float((predicted == labels).mean())


def predict(model, image: np.ndarray) -> str:
    """Return the predicted class name for a single `image`."""
    model.eval()

    with torch.no_grad():
        logits = _forward(model, image, ShapeError)
        predicted_index = int(logits.argmax(dim=1).item())

    return model.class_names[predicted_index]
