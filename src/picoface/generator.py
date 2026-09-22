"""Build, train, and generate from an autoencoder-to-VAE progression of image models.

No `nn.Module` authoring, no hand-written training loop: `build_autoencoder()`
or `build_vae()` constructs a model sized to your data, `train()` runs the
full training loop, and `generate()` samples new images from a trained VAE —
all via plain function calls. `train()` and `generate()` are the same
model-agnostic functions used elsewhere in the library. GAN-based generation
is a possible future extension, not part of this library.
"""

from picoface._internals.errors import BaseShapeError, CapabilityError, GeneratorError
from picoface._internals.generator_internals import _build_autoencoder, _build_vae
from picoface._internals.model_api import TrainingHistory, generate, train
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
    "generate",
]


class ShapeError(BaseShapeError):
    """Raised when an input/output image shape doesn't match what's expected."""


def build_autoencoder(data: Dataset):
    """Build a plain (non-variational) encoder/decoder model sized for `data`.

    A pedagogical stepping stone toward `build_vae()`: trainable via the same
    `train()` call, but has no probabilistic latent space to `generate()` from.
    Like the VAE, it also learns to classify (see `evaluate()`/`predict()`).
    """
    input_shape = tuple(data.images.shape[1:])
    model = _build_autoencoder(input_shape, len(data.class_names), ShapeError)
    model.class_names = list(data.class_names)
    return model


def build_vae(data: Dataset):
    """Build a variational autoencoder (VAE) sized for `data`.

    Same call shape as `build_autoencoder(data)` — swap one for the other and
    re-run `train()` unchanged. Unlike a plain autoencoder, a trained VAE can
    be sampled from with `generate()`. It is trained jointly to reconstruct,
    to organize its latent space, and to classify (see `evaluate()`/`predict()`).
    """
    input_shape = tuple(data.images.shape[1:])
    model = _build_vae(input_shape, len(data.class_names), ShapeError)
    model.class_names = list(data.class_names)
    return model
