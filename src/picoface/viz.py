"""Visualization helpers: plot results without writing matplotlib code directly."""

import matplotlib.pyplot as plt
import numpy as np

from picoface._internals.classifier_internals import TrainingHistory
from picoface._internals.generator_internals import _encode_mu
from picoface.datasets import Dataset
from picoface.generator import GeneratorError, ShapeError

__all__ = ["plot_training_history", "show_latent_space"]


def plot_training_history(history: TrainingHistory):
    """Plot loss per epoch from a classifier's `TrainingHistory`.

    Returns the matplotlib `Figure`.
    """
    fig, ax = plt.subplots()
    epochs = range(1, len(history.loss) + 1)
    ax.plot(epochs, history.loss, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    return fig


def show_latent_space(vae_model, data: Dataset):
    """Scatter-plot a trained VAE's 2D latent encoding of `data`, colored by class.

    Plots the encoder's mean (`mu`), not a stochastic sample, for a stable,
    reproducible plot. Raises `GeneratorError` if `vae_model` was built by
    `build_autoencoder()` instead of `build_vae()`.
    """
    if not getattr(vae_model, "is_variational", False):
        raise GeneratorError(
            "show_latent_space() requires a model built by build_vae(); got a "
            "build_autoencoder() model."
        )

    points = _encode_mu(vae_model, data.images, ShapeError)
    labels = np.asarray(data.labels)

    fig, ax = plt.subplots()
    for class_idx, class_name in enumerate(data.class_names):
        mask = labels == class_idx
        ax.scatter(points[mask, 0], points[mask, 1], label=class_name)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_title("Latent Space")
    ax.legend()
    return fig
