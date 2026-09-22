import numpy as np
import pytest
import torch
from _splits import ACCURACY_N_PER_CLASS, SHAPE_CLASSES, train_and_held_out

from picoface._internals.stub_data import make_stub_dataset
from picoface.classifier import build_classifier, evaluate, train

ALL_SHAPE_CLASSES = ["square", "cross", "ring", "hbar", "vbar", "diamond", "circle"]


def _mean_brightness(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).mean(axis=1)


@pytest.mark.parametrize("class_names", [SHAPE_CLASSES, ALL_SHAPE_CLASSES])
def test_shapes_are_not_separable_by_mean_brightness(class_names):
    # Nearest class-mean classifier on per-image mean brightness, fit on one
    # seed and scored on another. Seeded, so the margin is deterministic;
    # observed 0.35 (3 classes) and 0.17-0.19 (7 classes).
    data, held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS, kind="shapes", class_names=class_names
    )
    brightness = _mean_brightness(data.images)
    centroids = np.array([brightness[data.labels == c].mean() for c in range(len(class_names))])

    predicted = np.abs(_mean_brightness(held_out.images)[:, None] - centroids).argmin(axis=1)
    accuracy = np.mean(predicted == held_out.labels)

    assert accuracy < 1 / len(class_names) + 0.1


def test_shapes_are_recoverable_from_spatial_arrangement():
    torch.manual_seed(0)
    data, held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS, kind="shapes", class_names=SHAPE_CLASSES
    )
    model = build_classifier(data)

    train(model, data)

    assert evaluate(model, held_out) > 0.8  # 3 classes; observed 1.0


def test_shapes_honor_arbitrary_dimensions_and_class_names():
    data = make_stub_dataset(
        n_per_class=3, height=24, width=12, channels=1, class_names=["p", "q"], kind="shapes"
    )

    assert data.images.shape == (6, 24, 12, 1)
    assert data.class_names == ["p", "q"]


def test_shapes_reject_more_classes_than_figures():
    with pytest.raises(ValueError):
        make_stub_dataset(kind="shapes", class_names=[str(i) for i in range(8)])


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        make_stub_dataset(kind="stripes")
