"""model-interface: wrong kind of input is named in the error."""

from pathlib import Path

import numpy as np
import pytest

from picoface.classifier import (
    BaseShapeError,
    ShapeError,
    build_classifier,
    evaluate,
    predict,
    train,
)
from picoface.generator import ShapeError as GeneratorShapeError
from picoface.generator import build_autoencoder, build_vae
from picoface.linkage import classify_generated
from picoface._internals.stub_data import make_stub_dataset


class _FakePicture:
    """Stands in for a mediaComp Picture: anything with getImage()."""

    def getImage(self):
        return None


@pytest.fixture(scope="module")
def data():
    return make_stub_dataset(n_per_class=4, height=16, width=16, channels=1)


@pytest.fixture(scope="module")
def classifier(data):
    return build_classifier(data)


@pytest.fixture(scope="module")
def vae(data):
    return build_vae(data)


# --- non-dataset passed as data ------------------------------------------------


def test_picture_passed_to_evaluate(classifier):
    with pytest.raises(TypeError) as info:
        evaluate(classifier, _FakePicture())

    message = str(info.value)
    assert "evaluate()" in message
    assert "a picture" in message
    assert "load_dataset()" in message
    assert "predict(" in message


@pytest.mark.parametrize("path", ["train.npz", Path("train.npz")])
def test_file_path_passed_to_build_classifier(path):
    with pytest.raises(TypeError) as info:
        build_classifier(path)

    message = str(info.value)
    assert "build_classifier()" in message
    assert "load_dataset('train.npz')" in message


def test_image_array_passed_to_train(classifier, data):
    with pytest.raises(TypeError) as info:
        train(classifier, data.images)

    message = str(info.value)
    assert "train()" in message
    assert "image array of shape (8, 16, 16, 1)" in message
    assert "predict(" in message


@pytest.mark.parametrize(
    "call",
    [
        lambda bad, cnn, vae: build_autoencoder(bad),
        lambda bad, cnn, vae: build_vae(bad),
        lambda bad, cnn, vae: classify_generated(cnn, vae, bad),
    ],
    ids=["build_autoencoder", "build_vae", "classify_generated"],
)
def test_every_data_taking_function_checks_its_dataset(call, classifier, vae):
    with pytest.raises(TypeError, match="needs a dataset"):
        call({"images": None}, classifier, vae)


# --- predict() image-array checks --------------------------------------------------


def test_two_dimensional_image(classifier, data):
    image = data.images[0][..., 0]

    with pytest.raises(ShapeError) as info:
        predict(classifier, image)

    message = str(info.value)
    assert "(16, 16)" in message
    assert "np.newaxis" in message


def test_float_image(classifier, data):
    image = data.images[0].astype(np.float32) / 255

    with pytest.raises(ShapeError) as info:
        predict(classifier, image)

    assert "0 to 255" in str(info.value)


def test_batch_passed_to_predict(classifier, data):
    with pytest.raises(ShapeError) as info:
        predict(classifier, data.images)

    message = str(info.value)
    assert "one image" in message
    assert "evaluate(" in message
    assert "data.images[i]" in message


def test_image_format_errors_share_the_base_shape_error(classifier, vae, data):
    image = data.images[0][..., 0]

    for model, error_cls in ((classifier, ShapeError), (vae, GeneratorShapeError)):
        with pytest.raises(BaseShapeError) as info:
            predict(model, image)
        assert isinstance(info.value, error_cls)
