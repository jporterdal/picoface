"""Visualization helpers: plot results without writing matplotlib code directly."""

import matplotlib.pyplot as plt

from picoface._internals.classifier_internals import TrainingHistory

__all__ = ["plot_training_history"]


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
