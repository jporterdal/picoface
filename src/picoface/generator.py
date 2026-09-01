"""Build, train, and generate from an autoencoder-to-VAE progression of image models.

No `nn.Module` authoring, no hand-written training loop: `build_autoencoder()`
or `build_vae()` constructs a model sized to your data, `train()` runs the
full training loop, and `generate()` samples new images from a trained VAE —
all via plain function calls. GAN-based generation is a possible future
extension, not part of this library.
"""

import numpy as np

from picoface._internals.generator_internals import (
    TrainingHistory,
    _build_autoencoder,
    _build_vae,
    _sample_generate,
    _train_loop,
)
from picoface.datasets import Dataset

__all__ = [
    "ShapeError",
    "GeneratorError",
    "TrainingHistory",
    "build_autoencoder",
    "build_vae",
    "train",
    "generate",
]


class ShapeError(ValueError):
    """Raised when an input/output image shape doesn't match what's expected."""


class GeneratorError(ValueError):
    """Raised when a function requiring a `build_vae()` model is given a
    `build_autoencoder()` model instead.
    """


def build_autoencoder(data: Dataset):
    """Build a plain (non-variational) encoder/decoder model sized for `data`.

    A pedagogical stepping stone toward `build_vae()`: trainable via the same
    `train()` call, but has no probabilistic latent space to `generate()` from.
    """
    input_shape = tuple(data.images.shape[1:])
    return _build_autoencoder(input_shape, ShapeError)


def build_vae(data: Dataset):
    """Build a variational autoencoder (VAE) sized for `data`.

    Same call shape as `build_autoencoder(data)` — swap one for the other and
    re-run `train()` unchanged. Unlike a plain autoencoder, a trained VAE can
    be sampled from with `generate()`.
    """
    input_shape = tuple(data.images.shape[1:])
    return _build_vae(input_shape, ShapeError)


def train(
    model,
    data: Dataset,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> TrainingHistory:
    """Train `model` (from `build_autoencoder()` or `build_vae()`) on `data`.

    Runs the full training loop internally — no training loop to write. Works
    unchanged for either model type: an autoencoder trains against
    reconstruction loss alone, a VAE against reconstruction + KL-divergence loss.
    """
    return _train_loop(
        model,
        data.images,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shape_error_cls=ShapeError,
    )


def generate(vae_model, n: int) -> np.ndarray:
    """Sample `n` new images from a trained `build_vae()` model's latent space.

    Raises `GeneratorError` if `vae_model` was built by `build_autoencoder()`
    instead — a plain autoencoder has no probabilistic prior to sample from.
    """
    if not getattr(vae_model, "is_variational", False):
        raise GeneratorError(
            "generate() requires a model built by build_vae(); got a "
            "build_autoencoder() model — build_autoencoder() models have no "
            "probabilistic prior to sample from."
        )
    return _sample_generate(vae_model, n)
