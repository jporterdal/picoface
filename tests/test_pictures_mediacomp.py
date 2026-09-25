"""mediacomp-bridge against the real mediaComp from PyPI, when it is installed.

Optional: skipped wherever mediaComp can't be imported. Importing it also
imports tkinter, pygame, and sounddevice, and sounddevice raises `OSError`
(not `ImportError`) when the PortAudio library is missing, so any exception
means "skip". `show()` and `pictureTool()` need wxPython and a display, so
they are never called here; the manual Windows check covers them.
"""

import pytest

try:
    import mediaComp as mc
except Exception as error:  # noqa: BLE001
    pytest.skip(f"mediaComp is not importable: {error!r}", allow_module_level=True)

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from picoface._internals.stub_data import make_stub_dataset  # noqa: E402
from picoface.classifier import build_classifier, predict, train  # noqa: E402
from picoface.generator import build_vae, generate  # noqa: E402
from picoface.pictures import (  # noqa: E402
    PictureError,
    crop_and_center,
    picture_to_array,
    save_images,
    scale_down,
)


def make_greyscale(picture):
    """A student's grayscale loop, written with mediaComp's own functions."""
    for pixel in mc.getPixels(picture):
        gray = (mc.getRed(pixel) + mc.getGreen(pixel) + mc.getBlue(pixel)) // 3
        mc.setColor(pixel, mc.makeColor(gray, gray, gray))
    return picture


def _drawing(color=None):
    picture = mc.makeEmptyPicture(300, 200)
    mc.addOvalFilled(picture, 30, 20, 100, 100, color or mc.makeColor(20, 20, 20))
    return picture


def _assert_is_usable_picture(picture, width, height):
    assert type(picture) is type(mc.makeEmptyPicture(1, 1))
    assert (mc.getWidth(picture), mc.getHeight(picture)) == (width, height)
    for x, y in [(0, 0), (width - 1, height - 1), (width // 2, height // 2)]:
        mc.getRed(mc.getPixelAt(picture, x, y))


def test_a_grayscale_drawing_converts_and_a_colour_one_is_rejected():
    picture = _drawing()
    array = picture_to_array(picture)
    assert array.shape == (200, 300, 1)
    assert array[70, 80, 0] == 20 and array[0, 0, 0] == 255

    with pytest.raises(PictureError, match="grayscale first"):
        picture_to_array(_drawing(mc.makeColor(0, 0, 140)))


def test_the_helpers_return_real_pictures_that_mediacomp_can_use():
    original = _drawing()
    before = picture_to_array(original)

    centered = crop_and_center(original)
    side = mc.getWidth(centered)
    _assert_is_usable_picture(centered, side, side)
    small = scale_down(centered)
    _assert_is_usable_picture(small, 28, 28)
    assert picture_to_array(small).shape == (28, 28, 1)
    np.testing.assert_array_equal(picture_to_array(original), before)


def test_saved_images_open_with_makepicture_and_round_trip(tmp_path):
    images = np.random.default_rng(0).integers(0, 256, (3, 28, 28, 1), dtype=np.uint8)

    paths = save_images(images, tmp_path)

    for image, path in zip(images, paths):
        picture = mc.makePicture(path)
        _assert_is_usable_picture(picture, 224, 224)
        np.testing.assert_array_equal(picture_to_array(scale_down(picture)), image)


def test_a_drawing_is_classified_with_grayscale_first_or_last():
    torch.manual_seed(0)
    classes = ["square", "cross", "ring", "hbar", "vbar", "diamond", "circle"]
    data = make_stub_dataset(
        n_per_class=64, height=28, width=28, channels=1, class_names=classes, kind="shapes"
    )
    model = build_classifier(data)
    train(model, data)
    blue = mc.makeColor(0, 0, 140)

    gray_first = scale_down(crop_and_center(make_greyscale(_drawing(blue))))
    gray_last = make_greyscale(scale_down(crop_and_center(_drawing(blue))))

    _assert_is_usable_picture(gray_first, 28, 28)
    assert predict(model, gray_first) == predict(model, gray_last) == "circle"
    assert predict(model, gray_first) == predict(model, picture_to_array(gray_first))


def test_the_readme_walkthrough_runs(tmp_path, monkeypatch):
    """Every code line of README's "Using picoface with mediaComp", except `show()`."""
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    classes = ["square", "cross", "ring", "hbar", "vbar", "diamond", "circle"]
    data = make_stub_dataset(
        n_per_class=64, height=28, width=28, channels=1, class_names=classes, kind="shapes"
    )
    classifier_model = build_classifier(data)
    train(classifier_model, data)
    vae_model = build_vae(data)
    train(vae_model, data, epochs=1)
    # What the README's Quickstart has already defined.
    namespace = {
        "classifier_model": classifier_model,
        "vae_model": vae_model,
        "predict": predict,
        "generate": generate,
    }

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    section = readme.split("## Using picoface with mediaComp")[1].split("\n## ")[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 2
    code = "\n".join(blocks).replace("show(picture)", "")
    code = code.replace("print(predict(", "prediction = (predict(")
    exec(code, namespace)  # noqa: S102

    assert namespace["prediction"] == "circle"
    _assert_is_usable_picture(namespace["picture"], 224, 224)
