"""Build, train, evaluate, and run inference with a small CNN image classifier.

No `nn.Module` authoring, no hand-written training loop: `build_classifier()`
(or `build_classifier_from_shape()`) constructs a model sized to your data,
`train()` runs the full training loop, and `evaluate()`/`predict()` report
results — all via plain function calls. `train()`, `evaluate()`, and
`predict()` are the same model-agnostic functions used by the generator arm,
so they also work on models built by `picoface.generator`.
"""

from picoface._internals.classifier_internals import _build_classifier
from picoface._internals.errors import BaseShapeError, CapabilityError
from picoface._internals.model_api import (
    TrainingHistory,
    _require_dataset,
    evaluate,
    predict,
    train,
)
from picoface.datasets import Dataset

__all__ = [
    "BaseShapeError",
    "CapabilityError",
    "ShapeError",
    "TrainingHistory",
    "build_classifier",
    "build_classifier_from_shape",
    "train",
    "evaluate",
    "predict",
]


class ShapeError(BaseShapeError):
    """Raised when an image's shape or format, or a class count, doesn't match what's expected."""


def build_classifier(data: Dataset):
    """Build a CNN classifier sized for `data`'s class count and image shape.

    `data` is a dataset from `load_dataset()`. Returns a trainable model object — no `nn.Module` code, no manual shape
    derivation required.
    """
    _require_dataset(data, "build_classifier")
    num_classes = data.num_classes
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
