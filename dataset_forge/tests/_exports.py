"""Reading and editing a written export's bundles, to plant what validation should catch."""

import numpy as np

from dataset_forge.export import SPLITS


def load_split(out_dir, split):
    with np.load(out_dir / f"{split}.npz") as data:
        return data["images"].copy(), data["labels"].copy()


def rewrite_split(out_dir, split, images, labels):
    np.savez_compressed(out_dir / f"{split}.npz", images=images, labels=labels)


def offset_class(out_dir, label, amount):
    """Lighten (or, if negative, darken) every image of one class, in both splits."""
    for split in SPLITS:
        images, labels = load_split(out_dir, split)
        shifted = images[labels == label].astype(np.int16) + amount
        images[labels == label] = np.clip(shifted, 0, 255).astype(np.uint8)
        rewrite_split(out_dir, split, images, labels)
