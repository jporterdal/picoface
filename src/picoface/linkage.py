"""Tie the classifier and the generator together: the capstone exercise.

`classify_generated()` asks an independently trained classifier to judge
images the generator made for each class, and `activation_maximize()` shows
what a classifier has learned to look for in a class. Both take trained model
objects straight from the same session — nothing is saved to or loaded from
disk.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from picoface._internals.errors import BaseShapeError, CapabilityError, GeneratorError
from picoface._internals.linkage_internals import (
    _agreement,
    _ascend,
    _class_latent_clusters,
    _predicted_class_names,
    _sample_near_clusters,
)
from picoface._internals.model_api import _require_capability, _to_uint8_images
from picoface.datasets import Dataset

__all__ = [
    "BaseShapeError",
    "CapabilityError",
    "GeneratorError",
    "ShapeError",
    "GeneratedImagesReport",
    "classify_generated",
    "activation_maximize",
]


class ShapeError(BaseShapeError):
    """Raised when models or data used together disagree on image shape or class names."""


@dataclass
class GeneratedImagesReport:
    """What `classify_generated()` found.

    `per_class` maps each class name to the fraction of the images generated
    for that class that the classifier also called that class; `overall` is
    the same fraction across every generated image. `images[i]` was generated
    for class `intended[i]` and classified as `predicted[i]`.
    """

    per_class: dict[str, float]
    overall: float
    images: np.ndarray = field(repr=False)
    intended: list[str] = field(repr=False)
    predicted: list[str] = field(repr=False)


def _check_shared_setup(classifier_model, generator_model, data: Dataset) -> None:
    if tuple(classifier_model.input_shape) != tuple(generator_model.input_shape):
        raise ShapeError(
            f"the classifier expects images of shape {tuple(classifier_model.input_shape)}, "
            f"but the generator makes images of shape {tuple(generator_model.input_shape)}."
        )
    if set(classifier_model.class_names) != set(generator_model.class_names):
        raise ShapeError(
            f"the classifier's classes {classifier_model.class_names!r} do not match the "
            f"generator's classes {generator_model.class_names!r}; both must be trained on "
            "the same set of classes."
        )
    if set(data.class_names) != set(generator_model.class_names):
        raise ShapeError(
            f"data's classes {data.class_names!r} do not match the generator's classes "
            f"{generator_model.class_names!r}; pass the data the generator was trained on."
        )


def classify_generated(
    classifier_model, generator_model, data: Dataset, n: int = 10
) -> GeneratedImagesReport:
    """Generate `n` images for each class and check whether the classifier agrees.

    `generator_model` (from `build_vae()`) makes each image from the part of
    its latent space where `data`'s images of that class sit, so every image
    has an intended class. `classifier_model` — trained separately, so it is
    an independent judge — then classifies them. `data` is the labeled data
    the generator was trained on.
    """
    _require_capability(classifier_model, "classify", "classify_generated")
    _require_capability(
        generator_model, "latent_access", "classify_generated", error_cls=GeneratorError
    )
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    generator_model.check_data(data)
    _check_shared_setup(classifier_model, generator_model, data)

    clusters = _class_latent_clusters(generator_model, data)
    images, intended = _sample_near_clusters(generator_model, clusters, n)
    predicted = _predicted_class_names(classifier_model, images)
    per_class, overall = _agreement(intended, predicted, data.class_names)

    return GeneratedImagesReport(
        per_class=per_class,
        overall=overall,
        images=images,
        intended=intended,
        predicted=predicted,
    )


def activation_maximize(model, target_class: str) -> np.ndarray:
    """Make an image that `model` scores as strongly as it can for `target_class`.

    Starts from random noise and repeatedly nudges the pixels in whatever
    direction raises the class's score, so the result shows what the model
    has learned to look for in that class. Works with any model that
    classifies. Returns one image in the model's image shape.
    """
    _require_capability(model, "classify", "activation_maximize")
    if target_class not in model.class_names:
        raise ValueError(
            f"target_class {target_class!r} is not one of this model's classes: "
            f"{model.class_names!r}."
        )

    height, width, channels = model.input_shape
    start = torch.rand(1, channels, height, width)
    pixels = _ascend(model, model.class_names.index(target_class), start)
    return _to_uint8_images(pixels)[0]
