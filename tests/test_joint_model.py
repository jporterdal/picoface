"""The supervised VAE (one model that classifies and generates) and the
model-agnostic interface it plugs into (picoface-phase3c)."""

import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from _splits import ACCURACY_N_PER_CLASS, SHAPE_CLASSES, train_and_held_out

from picoface import classifier, generator
from picoface._internals.generator_internals import LOG_VAR_FLOOR, LOG_VAR_LR_MULTIPLIER
from picoface._internals.model_api import _preprocess_images
from picoface._internals.stub_data import make_stub_dataset
from picoface.classifier import (
    BaseShapeError,
    CapabilityError,
    build_classifier,
    evaluate,
    predict,
)
from picoface.datasets import Dataset
from picoface.generator import GeneratorError, build_autoencoder, build_vae, generate, train
from picoface.viz import plot_training_history

VAE_SERIES = (
    "loss",
    "reconstruction_loss",
    "kl_loss",
    "classification_loss",
    "accuracy",
    "kl_weight",
    "reconstruction_log_var",
    "classification_log_var",
)


def _shuffled_labels(data: Dataset, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    return Dataset(
        images=data.images, labels=rng.permutation(data.labels), class_names=data.class_names
    )


def test_one_trained_vae_classifies_held_out_images_and_generates():
    torch.manual_seed(0)
    data, held_out = train_and_held_out(n_per_class=ACCURACY_N_PER_CLASS)
    model = build_vae(data)

    train(model, data)

    assert evaluate(model, held_out) > 0.8  # 2 classes; observed 1.0 in 10/10 seeds
    assert predict(model, held_out.images[0]) in data.class_names
    images = generate(model, n=5)
    assert images.shape == (5, 16, 16, 3)
    assert images.dtype == np.uint8


def test_vae_classifies_spatial_classes_above_chance_on_held_out_data():
    # Loose by design (Decision 8): observed 0.82-0.98 over 10 seeds.
    torch.manual_seed(0)
    data, held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS, kind="shapes", class_names=SHAPE_CLASSES
    )
    model = build_vae(data)

    train(model, data)

    assert evaluate(model, held_out) > 1 / len(SHAPE_CLASSES) + 0.3


def test_labels_actually_supervise_the_vae():
    # Regression tripwire against a silently unsupervised training path.
    # Observed 0.92 with correct labels and 0.33 (chance) with shuffled ones.
    data, held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS, kind="shapes", class_names=SHAPE_CLASSES
    )

    torch.manual_seed(0)
    supervised = build_vae(data)
    train(supervised, data)
    torch.manual_seed(0)
    shuffled = build_vae(data)
    train(shuffled, _shuffled_labels(data))

    assert evaluate(supervised, held_out) > evaluate(shuffled, held_out) + 0.2


def test_predict_returns_a_class_name():
    data = make_stub_dataset(n_per_class=4, class_names=["circle", "square", "star"])
    model = build_vae(data)
    train(model, data, epochs=2)

    assert predict(model, data.images[0]) in data.class_names


def test_vae_records_num_classes_and_class_names():
    data = make_stub_dataset(n_per_class=4, class_names=["x", "y", "z"])
    model = build_vae(data)

    assert model.num_classes == 3
    assert model.class_names == ["x", "y", "z"]
    assert tuple(model.classify(_preprocess_images(data.images[:5], model)).shape) == (5, 3)


def test_vae_class_count_mismatch_raises_shape_error():
    two_class = make_stub_dataset(n_per_class=4)
    three_class = make_stub_dataset(n_per_class=4, class_names=["a", "b", "c"])
    model = build_vae(two_class)

    with pytest.raises(generator.ShapeError):
        evaluate(model, three_class)
    with pytest.raises(generator.ShapeError):
        train(model, three_class, epochs=1)


def test_image_shape_mismatch_raises_shape_error():
    data = make_stub_dataset(n_per_class=4, height=16, width=16)
    other = make_stub_dataset(n_per_class=4, height=24, width=12, channels=3)
    vae = build_vae(data)

    with pytest.raises(generator.ShapeError):
        predict(vae, other.images[0])
    with pytest.raises(generator.ShapeError):
        evaluate(vae, other)
    with pytest.raises(generator.ShapeError):
        train(build_autoencoder(data), other, epochs=1)


def test_one_handler_catches_shape_errors_from_either_arm():
    data = make_stub_dataset(n_per_class=4, height=16, width=16)
    other = make_stub_dataset(n_per_class=4, height=24, width=12, channels=3)
    cnn = build_classifier(data)
    vae = build_vae(data)

    for model in (cnn, vae):
        with pytest.raises(BaseShapeError):
            predict(model, other.images[0])
    assert issubclass(classifier.ShapeError, BaseShapeError)
    assert issubclass(generator.ShapeError, BaseShapeError)
    assert not issubclass(generator.ShapeError, classifier.ShapeError)


def test_history_series_per_model_kind():
    data = make_stub_dataset(n_per_class=8)

    vae_history = train(build_vae(data), data, epochs=3)
    ae_history = train(build_autoencoder(data), data, epochs=3)
    cnn_history = train(build_classifier(data), data, epochs=3)

    for name in VAE_SERIES:
        assert len(getattr(vae_history, name)) == 3, name
    assert all(0.0 <= a <= 1.0 for a in vae_history.accuracy)

    assert len(ae_history.loss) == 3
    assert len(ae_history.reconstruction_loss) == 3
    for name in VAE_SERIES[2:]:
        assert getattr(ae_history, name) == [], name

    assert len(cnn_history.loss) == 3
    assert cnn_history.reconstruction_loss == []
    assert cnn_history.kl_loss == []


@pytest.mark.parametrize("epochs", [3, 20])
def test_kl_weight_anneals_to_one_within_any_run_length(epochs):
    data = make_stub_dataset(n_per_class=8)

    history = train(build_vae(data), data, epochs=epochs)

    assert history.kl_weight[0] < history.kl_weight[-1]
    assert history.kl_weight[-1] == 1.0
    assert history.kl_weight == sorted(history.kl_weight)


@pytest.mark.parametrize("model_kind", ["cnn", "ae"])
def test_generate_rejects_models_that_cannot_generate(model_kind):
    data = make_stub_dataset(n_per_class=4)
    model = build_classifier(data) if model_kind == "cnn" else build_autoencoder(data)

    with pytest.raises(GeneratorError):
        generate(model, 3)
    with pytest.raises(CapabilityError):  # one shared handler covers both
        generate(model, 3)


@pytest.mark.parametrize("verb", [evaluate, predict])
def test_autoencoder_cannot_classify(verb):
    data = make_stub_dataset(n_per_class=4)
    model = build_autoencoder(data)
    target = data if verb is evaluate else data.images[0]

    with pytest.raises(GeneratorError, match="classify"):
        verb(model, target)
    with pytest.raises(CapabilityError):  # one shared handler covers both
        verb(model, target)


def test_train_rejects_non_models_clearly():
    data = make_stub_dataset(n_per_class=4)

    with pytest.raises(CapabilityError):
        train(object(), data)


def test_evaluate_rejects_a_model_without_classification():
    from picoface._internals.model_api import _Model

    class NoClassify(_Model):
        built_by = "a test model"

    with pytest.raises(CapabilityError):
        evaluate(NoClassify(), make_stub_dataset(n_per_class=4))


def test_verbs_are_the_same_functions_from_both_arms():
    assert classifier.train is generator.train
    assert classifier.evaluate is generator.evaluate
    assert classifier.predict is generator.predict
    assert generator.generate is not None


def test_evaluate_is_deterministic_and_does_not_change_parameters():
    data = make_stub_dataset(n_per_class=8)
    model = build_vae(data)
    train(model, data, epochs=2)
    before = [p.detach().clone() for p in model.parameters()]

    first = evaluate(model, data)
    second = evaluate(model, data)

    assert first == second
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_classify_is_differentiable_wrt_input_pixels():
    data = make_stub_dataset(n_per_class=4)
    model = build_vae(data)
    pixels = _preprocess_images(data.images[:3], model).requires_grad_(True)

    model.classify(pixels).sum().backward()

    assert pixels.grad is not None and pixels.grad.abs().sum() > 0


@pytest.mark.parametrize("build", [build_classifier, build_autoencoder, build_vae])
def test_optimizer_groups_cover_every_parameter_once(build):
    model = build(make_stub_dataset(n_per_class=4))

    groups = model.optimizer_param_groups(1e-3)

    ids = [id(p) for group in groups for p in group["params"]]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert all("weight_decay" not in group for group in groups)


def test_vae_log_variances_train_at_a_multiple_of_the_learning_rate():
    model = build_vae(make_stub_dataset(n_per_class=4))
    log_var_ids = {id(model.reconstruction_log_var), id(model.classification_log_var)}

    for group in model.optimizer_param_groups(2e-3):
        group_ids = {id(p) for p in group["params"]}
        if group_ids == log_var_ids:
            assert group["lr"] == pytest.approx(2e-3 * LOG_VAR_LR_MULTIPLIER)
        else:
            assert not group_ids & log_var_ids
            assert group["lr"] == 2e-3


def test_stored_log_variances_are_held_at_or_above_the_floor():
    data = make_stub_dataset(n_per_class=8)
    model = build_vae(data)
    with torch.no_grad():
        model.reconstruction_log_var.fill_(LOG_VAR_FLOOR - 3)
        model.classification_log_var.fill_(LOG_VAR_FLOOR - 3)

    train(model, data, epochs=2)

    assert model.reconstruction_log_var.item() >= LOG_VAR_FLOOR
    assert model.classification_log_var.item() >= LOG_VAR_FLOOR


@pytest.mark.parametrize("fn", [build_autoencoder, build_vae])
def test_builders_take_only_data(fn):
    assert list(inspect.signature(fn).parameters) == ["data"]


def test_train_signature_has_no_classification_weight():
    assert list(inspect.signature(train).parameters) == [
        "model",
        "data",
        "epochs",
        "batch_size",
        "learning_rate",
    ]


def test_generate_takes_no_class_argument():
    assert list(inspect.signature(generate).parameters) == ["model", "n"]


@pytest.mark.parametrize(
    "height,width,channels,class_names",
    [
        (16, 16, 3, ["a", "b"]),
        (24, 12, 1, ["x", "y", "z"]),
    ],
)
def test_vae_is_shape_agnostic(height, width, channels, class_names):
    data = make_stub_dataset(
        n_per_class=6,
        height=height,
        width=width,
        channels=channels,
        class_names=class_names,
    )
    model = build_vae(data)

    history = train(model, data, epochs=2)

    assert len(history.accuracy) == 2
    assert predict(model, data.images[0]) in class_names
    assert 0.0 <= evaluate(model, data) <= 1.0


def test_plot_training_history_shows_accuracy_for_the_vae_only():
    data = make_stub_dataset(n_per_class=8)

    vae = plot_training_history(train(build_vae(data), data, epochs=2))
    cnn = plot_training_history(train(build_classifier(data), data, epochs=2))

    assert len(vae.axes) == 2
    assert len(cnn.axes) == 1
