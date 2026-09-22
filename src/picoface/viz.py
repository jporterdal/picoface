"""Visualization helpers: plot results without writing matplotlib code directly."""

import matplotlib.pyplot as plt

from picoface._internals.model_api import TrainingHistory

__all__ = ["plot_training_history"]


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

