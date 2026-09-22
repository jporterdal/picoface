"""The capstone linkage functions and the `latent_access` capability behind
them (picoface-phase4)."""

import inspect

import numpy as np
import pytest
import torch
from _splits import ACCURACY_N_PER_CLASS, SHAPE_CLASSES, train_and_held_out

from picoface import linkage
from picoface._internals.linkage_internals import (
    _agreement,
    _ascend,
    _class_latent_clusters,
    _sample_near_clusters,
)
from picoface._internals.model_api import _preprocess_images, _require_capability
from picoface._internals.stub_data import make_stub_dataset
from picoface.classifier import CapabilityError, build_classifier
from picoface.datasets import Dataset
from picoface.generator import GeneratorError, build_autoencoder, build_vae, generate, train
from picoface.linkage import activation_maximize, classify_generated


@pytest.fixture(scope="module")
def trained():
    """A CNN and a VAE trained separately on the same spatially separable data."""
    torch.manual_seed(0)
    data, _held_out = train_and_held_out(
        n_per_class=ACCURACY_N_PER_CLASS, kind="shapes", class_names=SHAPE_CLASSES
    )
    vae = build_vae(data)
    train(vae, data)
    cnn = build_classifier(data)
    train(cnn, data)
    return data, cnn, vae


def _reordered(data: Dataset) -> Dataset:
    """The same images and classes, with the class list enumerated in reverse."""
    k = len(data.class_names)
    return Dataset(
        images=data.images,
        labels=(k - 1) - np.asarray(data.labels),
        class_names=list(reversed(data.class_names)),
    )


# --- model-interface: latent_access -------------------------------------------


def test_vae_latent_access_round_trips_to_the_image_shape():
    data = make_stub_dataset(n_per_class=4)
    vae = build_vae(data)
    pixels = _preprocess_images(data.images[:5], vae)

    mu = vae.encode_mu(pixels)
    decoded = vae.decode(mu)

    assert "latent_access" in vae.capabilities
    assert tuple(mu.shape) == (5, vae.latent_dim)
    assert tuple(decoded.shape) == (5, 3, 16, 16)
    assert vae.decode(torch.randn(3, vae.latent_dim)).shape == vae.sample(3).shape


def test_encoding_does_not_change_the_model():
    data = make_stub_dataset(n_per_class=4)
    vae = build_vae(data)
    before = [p.detach().clone() for p in vae.parameters()]

    _class_latent_clusters(vae, data)

    assert all(torch.equal(b, p) for b, p in zip(before, vae.parameters()))


@pytest.mark.parametrize("build", [build_classifier, build_autoencoder])
def test_models_without_a_latent_space_lack_latent_access(build):
    model = build(make_stub_dataset(n_per_class=4))

    assert "latent_access" not in model.capabilities
    with pytest.raises(CapabilityError, match="latent space"):
        _require_capability(model, "latent_access", "classify_generated")


def test_generate_still_samples_from_the_prior():
    data = make_stub_dataset(n_per_class=4)

    images = generate(build_vae(data), n=4)

    assert images.shape == (4, 16, 16, 3)
    assert images.dtype == np.uint8


# --- classify_generated() -----------------------------------------------------


def test_class_clusters_are_latent_vectors_that_separate_the_classes(trained):
    data, _cnn, vae = trained

    clusters = _class_latent_clusters(vae, data)

    assert list(clusters) == data.class_names
    means = torch.stack([mean for mean, _std in clusters.values()])
    spreads = torch.stack([std for _mean, std in clusters.values()])
    assert tuple(means.shape) == (len(SHAPE_CLASSES), vae.latent_dim)
    # Loose by design: the closest pair of class means sat 1.3-3.2x the
    # typical within-class spread over 5 seeds.
    distances = torch.cdist(means, means)
    closest = distances[~torch.eye(len(SHAPE_CLASSES), dtype=torch.bool)].min()
    assert closest > 0.5 * spreads.norm(dim=1).mean()


def test_sampling_near_clusters_gives_n_uint8_images_per_class(trained):
    data, _cnn, vae = trained

    images, intended = _sample_near_clusters(vae, _class_latent_clusters(vae, data), n=4)

    assert images.shape == (4 * len(SHAPE_CLASSES), 16, 16, 3)
    assert images.dtype == np.uint8
    assert intended == [name for name in data.class_names for _ in range(4)]


def test_agreement_matches_a_hand_computed_report():
    per_class, overall = _agreement(
        intended=["a", "a", "b", "b"], predicted=["a", "b", "b", "b"], class_names=["a", "b"]
    )

    assert per_class == {"a": 0.5, "b": 1.0}
    assert overall == 0.75


def test_classify_generated_reports_every_class_from_in_memory_models(trained):
    data, cnn, vae = trained

    report = classify_generated(cnn, vae, data, n=5)

    assert list(report.per_class) == data.class_names
    assert all(0.0 <= rate <= 1.0 for rate in report.per_class.values())
    assert 0.0 <= report.overall <= 1.0
    assert report.images.shape == (5 * len(SHAPE_CLASSES), 16, 16, 3)
    assert report.images.dtype == np.uint8
    assert len(report.intended) == len(report.predicted) == len(report.images)
    assert set(report.predicted) <= set(data.class_names)


def test_mismatched_class_names_are_rejected():
    data = make_stub_dataset(n_per_class=4)
    other = make_stub_dataset(n_per_class=4, class_names=["x", "y"])

    with pytest.raises(linkage.ShapeError, match="classes"):
        classify_generated(build_classifier(other), build_vae(data), data)
    with pytest.raises(linkage.ShapeError, match="classes"):
        classify_generated(build_classifier(other), build_vae(other), data)


def test_mismatched_image_shapes_are_rejected():
    data = make_stub_dataset(n_per_class=4, height=16, width=16)
    other = make_stub_dataset(n_per_class=4, height=24, width=12)

    with pytest.raises(linkage.ShapeError, match="shape"):
        classify_generated(build_classifier(other), build_vae(data), data)


def test_class_order_does_not_have_to_match(trained):
    data, _cnn, vae = trained
    reordered = _reordered(data)
    cnn = build_classifier(reordered)
    train(cnn, reordered, epochs=2)

    report = classify_generated(cnn, vae, data, n=2)

    assert list(report.per_class) == data.class_names


@pytest.mark.parametrize("build", [build_classifier, build_autoencoder])
def test_a_generator_without_a_latent_space_is_rejected(build):
    data = make_stub_dataset(n_per_class=4)

    with pytest.raises(GeneratorError, match="latent space"):
        classify_generated(build_classifier(data), build(data), data)


def test_a_classifier_that_cannot_classify_is_rejected():
    data = make_stub_dataset(n_per_class=4)

    with pytest.raises(CapabilityError, match="classify"):
        classify_generated(build_autoencoder(data), build_vae(data), data)


def test_n_must_be_positive():
    data = make_stub_dataset(n_per_class=4)

    with pytest.raises(ValueError):
        classify_generated(build_classifier(data), build_vae(data), data, n=0)


# --- activation_maximize() ----------------------------------------------------


def test_ascent_raises_the_target_class_score(trained):
    _data, cnn, _vae = trained
    torch.manual_seed(0)
    start = torch.rand(1, 3, 16, 16)
    target = 1

    after = _ascend(cnn, target, start)

    assert cnn.classify(after)[0, target] > cnn.classify(start)[0, target]
    assert after.min() >= 0.0 and after.max() <= 1.0


@pytest.mark.parametrize("model_kind", ["cnn", "vae"])
def test_activation_maximize_returns_one_image_for_any_classifying_model(trained, model_kind):
    _data, cnn, vae = trained
    model = cnn if model_kind == "cnn" else vae

    image = activation_maximize(model, target_class="cross")

    assert image.shape == (16, 16, 3)
    assert image.dtype == np.uint8


def test_activation_maximize_leaves_the_model_unchanged(trained):
    _data, cnn, _vae = trained
    before = [p.detach().clone() for p in cnn.parameters()]
    grads_before = [None if p.grad is None else p.grad.clone() for p in cnn.parameters()]

    activation_maximize(cnn, target_class="ring")

    assert all(torch.equal(b, p) for b, p in zip(before, cnn.parameters()))
    for g, p in zip(grads_before, cnn.parameters()):
        assert (g is None and p.grad is None) or torch.equal(g, p.grad)


def test_activation_maximize_rejects_a_model_that_cannot_classify():
    with pytest.raises(CapabilityError, match="classify"):
        activation_maximize(build_autoencoder(make_stub_dataset(n_per_class=4)), "class_a")


def test_activation_maximize_rejects_an_unknown_class(trained):
    _data, cnn, _vae = trained

    with pytest.raises(ValueError, match="triangle"):
        activation_maximize(cnn, target_class="triangle")


def test_linkage_signatures():
    assert list(inspect.signature(classify_generated).parameters) == [
        "classifier_model",
        "generator_model",
        "data",
        "n",
    ]
    assert list(inspect.signature(activation_maximize).parameters) == ["model", "target_class"]
