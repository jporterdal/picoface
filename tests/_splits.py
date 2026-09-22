"""Held-out splits of the stub dataset, for every accuracy measurement.

Accuracy is always measured on a second stub dataset generated with a different
seed, never on the training images (picoface-phase3c tasks.md, 7.2).
"""

from picoface._internals.stub_data import make_stub_dataset

# Accuracy tests need enough optimizer steps for learning to show. The default
# stub (8 images per class) is one step per epoch at batch size 16, and a VAE
# stays at chance after 10 such steps; 64 per class gives 8-12 steps per epoch,
# and reached held-out accuracy well above chance in 10/10 seeds within the
# default 10 epochs (picoface-phase3c diagnostics.md).
ACCURACY_N_PER_CLASS = 64

SHAPE_CLASSES = ["square", "cross", "ring"]


def train_and_held_out(**stub_kwargs):
    """A training set (seed 0) and a held-out set (seed 1), otherwise identical."""
    return make_stub_dataset(seed=0, **stub_kwargs), make_stub_dataset(seed=1, **stub_kwargs)
