"""Visualization helpers: plot results without writing matplotlib code directly."""

import matplotlib.pyplot as plt
import numpy as np

from picoface._internals.model_api import TrainingHistory, _latent_mean
from picoface.datasets import Dataset

__all__ = ["plot_training_history", "show_latent_space"]


def plot_training_history(history: TrainingHistory):
    """Plot loss per epoch from a `TrainingHistory` returned by `train()`.

    If the history also recorded classification accuracy (models that
    classify alongside generating), a second panel shows accuracy per epoch.

    Returns the matplotlib `Figure`.
    """
    epochs = range(1, len(history.loss) + 1)

    if history.accuracy:
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex=True)
        ax_acc.plot(range(1, len(history.accuracy) + 1), history.accuracy, marker="o")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_title("Training Accuracy")
    else:
        fig, ax_loss = plt.subplots()
        ax_loss.set_xlabel("Epoch")

    ax_loss.plot(epochs, history.loss, marker="o")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    return fig


def show_latent_space(vae_model, data: Dataset):
    """Scatter-plot a trained VAE's 2D latent encoding of `data`, colored by class.

    Plots the encoder's mean (`mu`), not a stochastic sample, for a stable,
    reproducible plot. Raises `GeneratorError` if `vae_model` was built by
    `build_autoencoder()` instead of `build_vae()`.
    """
    points = _latent_mean(vae_model, data.images)
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
