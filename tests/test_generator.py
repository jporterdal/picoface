import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from picoface._internals.stub_data import make_stub_dataset
from picoface.datasets import Dataset
from picoface.generator import (
    GeneratorError,
    ShapeError,
    build_autoencoder,
    build_vae,
    generate,
    train,
)


def test_build_train_autoencoder_end_to_end():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)

    model = build_autoencoder(data)
    history = train(model, data, epochs=2)

    assert len(history.loss) == 2


def test_build_train_generate_vae_end_to_end():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)

    model = build_vae(data)
    train(model, data, epochs=2)
    images = generate(model, n=5)

    assert images.shape == (5, 16, 16, 3)
    assert images.dtype == np.uint8


def test_train_accepts_both_ae_and_vae_models_unchanged():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    ae_model = build_autoencoder(data)
    vae_model = build_vae(data)

    ae_history = train(ae_model, data, epochs=2)
    vae_history = train(vae_model, data, epochs=2)

    assert len(ae_history.loss) == 2
    assert len(vae_history.loss) == 2


def test_generate_on_autoencoder_raises_generator_error():
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    model = build_autoencoder(data)
    train(model, data, epochs=1)

    with pytest.raises(GeneratorError):
        generate(model, n=3)


def test_vae_training_wall_clock_under_generous_ceiling():
    # The full combined objective (reconstruction + annealed KL + weighted
    # classification), default stub and default train() parameters.
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    model = build_vae(data)

    history = train(model, data)

    assert history.wall_clock_seconds < 300.0


def test_autoencoder_ignores_labels():
    torch.manual_seed(0)
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    relabeled = Dataset(
        images=data.images,
        labels=np.arange(len(data.labels)) % 3,
        class_names=["x", "y", "z"],
    )

    torch.manual_seed(1)
    history = train(build_autoencoder(data), data, epochs=3)
    torch.manual_seed(1)
    relabeled_history = train(build_autoencoder(data), relabeled, epochs=3)

    assert history.loss == relabeled_history.loss


def test_autoencoder_does_not_check_class_count():
    two_class = make_stub_dataset(n_per_class=4)
    three_class = make_stub_dataset(n_per_class=4, class_names=["a", "b", "c"])
    model = build_autoencoder(two_class)

    assert model.num_classes is None
    history = train(model, three_class, epochs=1)

    assert len(history.reconstruction_loss) == 1


@pytest.mark.parametrize(
    "height,width,channels,class_names",
    [
        (16, 16, 3, ["a", "b"]),
        (24, 12, 1, ["x", "y", "z"]),
    ],
)
def test_shape_agnostic(height, width, channels, class_names):
    data = make_stub_dataset(
        n_per_class=6,
        height=height,
        width=width,
        channels=channels,
        class_names=class_names,
    )

    ae_model = build_autoencoder(data)
    ae_history = train(ae_model, data, epochs=2)

    vae_model = build_vae(data)
    vae_history = train(vae_model, data, epochs=2)
    images = generate(vae_model, n=3)

    assert len(ae_history.loss) == 2
    assert len(vae_history.loss) == 2
    assert images.shape == (3, height, width, channels)


def test_decode_shape_mismatch_raises_shape_error():
    # Height/width of 6 downsamples to an odd intermediate size (3) that the
    # transpose-conv upsample path can't exactly round-trip back to 6.
    data = make_stub_dataset(n_per_class=4, height=6, width=6, channels=3)

    with pytest.raises(ShapeError):
        build_autoencoder(data)

    with pytest.raises(ShapeError):
        build_vae(data)


def test_vae_reconstruction_loss_decreases():
    # Seeded: with 16 images and batch_size=16 this is one optimizer step per
    # epoch. Compares the first epoch with the best later one rather than the
    # last: reconstruction may rise as the KL weight anneals up (picoface-phase3c
    # tasks.md, 7.15). First-vs-best held in 10/10 seeds, first-vs-last in 9/10.
    torch.manual_seed(0)
    data = make_stub_dataset(n_per_class=8, height=16, width=16, channels=3)
    model = build_vae(data)

    history = train(model, data, epochs=5)

    assert min(history.reconstruction_loss[1:]) < history.reconstruction_loss[0]
