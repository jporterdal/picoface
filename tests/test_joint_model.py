import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from picoface import classifier, generator
from picoface._internals.model_api import _preprocess_images
from picoface._internals.stub_data import make_stub_dataset
from picoface.classifier import (
    BaseShapeError,
    CapabilityError,
    build_classifier,
    evaluate,
    predict,
)
from picoface.generator import GeneratorError, build_autoencoder, build_vae, generate, train
from picoface.viz import plot_training_history, show_latent_space

JOINT_BUILDERS = [build_autoencoder, build_vae]

# 16 stub images at the default batch size is one optimizer step per epoch, so
# the default 10 epochs is too few for the stub dataset to show classification
# (see picoface-phase3b-joint-model design.md); 50 epochs reached 100% in 10/10
# seeds. Tests asserting training progress also fix a torch seed and compare
# only the first few epochs (tasks.md 1.9).
STUB_EPOCHS = 50


@pytest.mark.parametrize("build", JOINT_BUILDERS)
@pytest.mark.parametrize("class_names", [["a", "b"], ["a", "b", "c"]])
def test_joint_model_classifies_above_chance_after_one_train_call(build, class_names):
    torch.manual_seed(0)
    data = make_stub_dataset(n_per_class=8, class_names=class_names)
    model = build(data)

    train(model, data, epochs=STUB_EPOCHS)

    assert evaluate(model, data) > 1 / len(class_names)


@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_predict_returns_a_class_name_on_joint_models(build):
    data = make_stub_dataset(n_per_class=4, class_names=["circle", "square", "star"])
    model = build(data)
    train(model, data, epochs=2)

    assert predict(model, data.images[0]) in data.class_names


@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_joint_model_records_num_classes_and_class_names(build):
    data = make_stub_dataset(n_per_class=4, class_names=["x", "y", "z"])
    model = build(data)

    assert model.num_classes == 3
    assert model.class_names == ["x", "y", "z"]
    assert tuple(model.classify(_preprocess_images(data.images[:5], model)).shape) == (5, 3)


@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_joint_model_class_count_mismatch_raises_shape_error(build):
    two_class = make_stub_dataset(n_per_class=4)
    three_class = make_stub_dataset(n_per_class=4, class_names=["a", "b", "c"])
    model = build(two_class)

    with pytest.raises(generator.ShapeError):
        evaluate(model, three_class)
    with pytest.raises(generator.ShapeError):
        train(model, three_class, epochs=1)


@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_joint_model_image_shape_mismatch_raises_shape_error(build):
    data = make_stub_dataset(n_per_class=4, height=16, width=16)
    other = make_stub_dataset(n_per_class=4, height=24, width=12, channels=3)
    model = build(data)

    with pytest.raises(generator.ShapeError):
        predict(model, other.images[0])
    with pytest.raises(generator.ShapeError):
        evaluate(model, other)


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


def test_vae_history_has_all_series_and_cnn_history_has_only_loss():
    data = make_stub_dataset(n_per_class=8)

    vae_history = train(build_vae(data), data, epochs=3)
    cnn_history = train(build_classifier(data), data, epochs=3)

    for name in ("loss", "reconstruction_loss", "kl_loss", "classification_loss", "accuracy"):
        assert len(getattr(vae_history, name)) == 3, name
    assert all(0.0 <= a <= 1.0 for a in vae_history.accuracy)

    assert len(cnn_history.loss) == 3
    assert cnn_history.reconstruction_loss == []
    assert cnn_history.kl_loss == []


def test_autoencoder_history_has_no_kl_series():
    data = make_stub_dataset(n_per_class=8)

    history = train(build_autoencoder(data), data, epochs=3)

    assert history.kl_loss == []
    for name in ("loss", "reconstruction_loss", "classification_loss", "accuracy"):
        assert len(getattr(history, name)) == 3, name


def test_joint_vae_reconstruction_loss_decreases_with_classification_branch():
    torch.manual_seed(0)
    data = make_stub_dataset(n_per_class=8)
    model = build_vae(data)

    history = train(model, data, epochs=5)

    assert history.reconstruction_loss[-1] < history.reconstruction_loss[0]


@pytest.mark.parametrize("model_kind", ["cnn", "ae"])
def test_generate_rejects_models_that_cannot_generate(model_kind):
    data = make_stub_dataset(n_per_class=4)
    model = build_classifier(data) if model_kind == "cnn" else build_autoencoder(data)

    with pytest.raises(GeneratorError):
        generate(model, 3)
    with pytest.raises(CapabilityError):  # one shared handler covers both
        generate(model, 3)


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


def test_train_is_the_same_function_from_both_arms():
    assert classifier.train is generator.train
    assert classifier.evaluate is not None and classifier.predict is not None
    assert generator.generate is not None


@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_evaluate_is_deterministic_and_does_not_change_parameters(build):
    data = make_stub_dataset(n_per_class=8)
    model = build(data)
    train(model, data, epochs=2)
    before = [p.detach().clone() for p in model.parameters()]

    first = evaluate(model, data)
    second = evaluate(model, data)

    assert first == second
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))


@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_classify_is_differentiable_wrt_input_pixels(build):
    data = make_stub_dataset(n_per_class=4)
    model = build(data)
    pixels = _preprocess_images(data.images[:3], model).requires_grad_(True)

    model.classify(pixels).sum().backward()

    assert pixels.grad is not None and pixels.grad.abs().sum() > 0


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
@pytest.mark.parametrize("build", JOINT_BUILDERS)
def test_joint_models_are_shape_agnostic(build, height, width, channels, class_names):
    data = make_stub_dataset(
        n_per_class=6,
        height=height,
        width=width,
        channels=channels,
        class_names=class_names,
    )
    model = build(data)

    history = train(model, data, epochs=2)

    assert len(history.accuracy) == 2
    assert predict(model, data.images[0]) in class_names
    assert 0.0 <= evaluate(model, data) <= 1.0


def test_plot_training_history_shows_accuracy_for_joint_models_only():
    data = make_stub_dataset(n_per_class=8)

    joint = plot_training_history(train(build_vae(data), data, epochs=2))
    cnn = plot_training_history(train(build_classifier(data), data, epochs=2))

    assert len(joint.axes) == 2
    assert len(cnn.axes) == 1


def test_show_latent_space_still_plots_one_point_per_image_on_a_joint_model():
    data = make_stub_dataset(n_per_class=8)
    model = build_vae(data)
    train(model, data, epochs=2)

    fig = show_latent_space(model, data)

    assert sum(len(c.get_offsets()) for c in fig.axes[0].collections) == len(data.images)
    assert np.asarray(fig.axes[0].collections[0].get_offsets()).shape[1] == 2


def test_joint_vae_training_wall_clock_under_generous_ceiling():
    data = make_stub_dataset(n_per_class=8)

    history = train(build_vae(data), data)

    assert history.wall_clock_seconds < 300.0
