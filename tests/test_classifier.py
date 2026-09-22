import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from _splits import ACCURACY_N_PER_CLASS, train_and_held_out

from picoface._internals.stub_data import make_stub_dataset
from picoface.classifier import (
    ShapeError,
    build_classifier,
    build_classifier_from_shape,
    evaluate,
    predict,
    train,
)
from picoface.viz import plot_training_history


def test_build_train_evaluate_end_to_end():
    torch.manual_seed(0)
    data, held_out = train_and_held_out(n_per_class=ACCURACY_N_PER_CLASS)

    model = build_classifier(data)
    history = train(model, data)
    accuracy = evaluate(model, held_out)

    assert len(history.loss) == 10
    assert accuracy > 0.8  # 2 classes; observed 1.0


def test_build_classifier_from_shape_equivalent_to_build_classifier():
    torch.manual_seed(0)
    data, held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS, height=16, width=16, channels=3, class_names=["a", "b"]
    )

    model = build_classifier_from_shape(num_classes=len(data.class_names), input_shape=(16, 16, 3))

    history = train(model, data)
    accuracy = evaluate(model, held_out)

    assert len(history.loss) == 10
    assert accuracy > 0.8  # 2 classes; observed 1.0


def test_predict_returns_class_name():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    model = build_classifier(data)
    train(model, data, epochs=1)

    label = predict(model, data.images[0])

    assert label in data.class_names


def test_training_wall_clock_under_generous_ceiling():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    model = build_classifier(data)

    history = train(model, data)

    assert history.wall_clock_seconds < 60.0


@pytest.mark.parametrize(
    "height,width,channels,class_names",
    [
        (16, 16, 3, ["a", "b"]),
        (24, 12, 1, ["x", "y", "z"]),
    ],
)
def test_shape_agnostic(height, width, channels, class_names):
    torch.manual_seed(0)
    data, held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS,
        height=height,
        width=width,
        channels=channels,
        class_names=class_names,
    )

    model = build_classifier(data)
    history = train(model, data)
    accuracy = evaluate(model, held_out)

    assert len(history.loss) == 10
    assert accuracy > 1 / len(class_names) + 0.3


def test_build_classifier_from_shape_rejects_too_small_input():
    with pytest.raises(ShapeError):
        build_classifier_from_shape(num_classes=2, input_shape=(4, 4, 3))

    with pytest.raises(ShapeError):
        build_classifier_from_shape(num_classes=2, input_shape=(2, 2, 3))


def test_class_count_mismatch_raises():
    data = make_stub_dataset(
        n_per_class=6, height=16, width=16, channels=3, class_names=["a", "b"]
    )
    model = build_classifier_from_shape(num_classes=3, input_shape=(16, 16, 3))

    with pytest.raises(ShapeError):
        train(model, data)

    with pytest.raises(ShapeError):
        evaluate(model, data)


def test_image_shape_mismatch_raises():
    data = make_stub_dataset(n_per_class=6, height=16, width=16, channels=3)
    mismatched = make_stub_dataset(n_per_class=6, height=8, width=8, channels=3)
    model = build_classifier(data)

    with pytest.raises(ShapeError):
        train(model, mismatched)

    with pytest.raises(ShapeError):
        evaluate(model, mismatched)

    with pytest.raises(ShapeError):
        predict(model, mismatched.images[0])


def test_plot_training_history_smoke():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    model = build_classifier(data)
    history = train(model, data, epochs=2)

    fig = plot_training_history(history)

    assert fig is not None
