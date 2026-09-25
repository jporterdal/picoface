"""Build and train one supervised model that both classifies and generates images.

No `nn.Module` authoring, no hand-written training loop: `build_vae()`
constructs a model sized to your data, a single `train()` call teaches it
from labeled images, and the trained model then serves two workflows —
`evaluate()`/`predict()` classify images, and `generate()` samples new ones.
`train()`, `evaluate()`, `predict()`, and `generate()` are the same
model-agnostic functions used elsewhere in the library.

`build_autoencoder()` is an optional, reconstruction-only model kept for
completeness; it is not a step on the way to `build_vae()`. GAN-based
generation is a possible future extension, not part of this library.
"""

from picoface._internals.errors import BaseShapeError, CapabilityError, GeneratorError
from picoface._internals.generator_internals import _build_autoencoder, _build_vae
from picoface._internals.model_api import (
    TrainingHistory,
    _require_dataset,
    evaluate,
    generate,
    predict,
    train,
)
from picoface.datasets import Dataset

__all__ = [
    "BaseShapeError",
    "CapabilityError",
    "ShapeError",
    "GeneratorError",
    "TrainingHistory",
    "build_autoencoder",
    "build_vae",
    "train",
    "evaluate",
    "predict",
    "generate",
]


class ShapeError(BaseShapeError):
    """Raised when an image's shape or format, or a class count, doesn't match what's expected."""


def build_autoencoder(data: Dataset):
    """Build a plain (non-variational) encoder/decoder model sized for `data`.

    `data` is a dataset from `load_dataset()`.

    Optional, and not a prerequisite for `build_vae()`. It only learns to
    reconstruct images: `train()` ignores `data`'s labels for it, and it can
    neither classify (`evaluate()`/`predict()`) nor `generate()`.
    """
    _require_dataset(data, "build_autoencoder")
    input_shape = tuple(data.images.shape[1:])
    return _build_autoencoder(input_shape, ShapeError)


def build_vae(data: Dataset):
    """Build a supervised variational autoencoder (VAE) sized for `data`.

    `data` is a dataset from `load_dataset()`. One `train()` call on labeled data teaches it to classify and to
    generate: its classifier reads the same probabilistic latent space its
    generator samples from, so the trained model works with both
    `evaluate()`/`predict()` and `generate()`.
    """
    _require_dataset(data, "build_vae")
    input_shape = tuple(data.images.shape[1:])
    model = _build_vae(input_shape, data.num_classes, ShapeError)
    model.class_names = list(data.class_names)
    return model
