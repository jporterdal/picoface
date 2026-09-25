"""mediacomp-bridge: picoface.pictures, and predict() with pictures."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from fake_picture import FakePicture
from PIL import Image, ImageDraw

from picoface._internals.stub_data import make_stub_dataset
from picoface.classifier import ShapeError, build_classifier, predict, train
from picoface.pictures import (
    PictureError,
    crop_and_center,
    picture_to_array,
    save_images,
    scale_down,
)

WHITE = (255, 255, 255)
DARK = (20, 20, 20)


def _pixels(pic) -> np.ndarray:
    return np.asarray(pic.getImage())


def _make_greyscale(pic) -> FakePicture:
    """A student's grayscale loop: each pixel's channels set to their (truncated) mean."""
    out = FakePicture.blank(pic.getWidth(), pic.getHeight())
    for y in range(pic.getHeight()):
        for x in range(pic.getWidth()):
            red, green, blue = pic.getBasicPixel(x, y)
            gray = (red + green + blue) // 3
            out.setBasicPixel(x, y, (gray, gray, gray))
    return out


def _colourful_picture(side: int = 200) -> FakePicture:
    """Blocks of random colour, with a dark-blue circle on top."""
    blocks = np.random.default_rng(0).integers(0, 256, (10, 10, 3), dtype=np.uint8)
    image = Image.fromarray(blocks).resize((side, side), Image.Resampling.NEAREST)
    pic = FakePicture(image)
    pic.addOvalFilled((0, 0, 140), side // 4, side // 4, side // 2, side // 2)
    return pic


# picture_to_array


def test_a_grayscale_picture_converts_with_x_y_as_column_row():
    pic = FakePicture.blank(5, 3, (200, 200, 200))
    pic.setBasicPixel(4, 1, (17, 17, 17))

    array = picture_to_array(pic)

    assert array.shape == (3, 5, 1) and array.dtype == np.uint8
    assert array[1, 4, 0] == 17
    assert (array == 200).sum() == 14


def test_a_color_picture_is_rejected_with_guidance():
    pic = FakePicture.blank(5, 3, (200, 200, 200))
    pic.setBasicPixel(0, 2, (200, 0, 0))

    with pytest.raises(PictureError, match="grayscale first"):
        picture_to_array(pic)


def test_picture_to_array_names_what_it_was_given_instead():
    with pytest.raises(TypeError, match="ndarray"):
        picture_to_array(np.zeros((4, 4, 1), np.uint8))


# crop_and_center


def test_an_off_center_drawing_is_centered_and_framed_like_the_forge():
    pic = FakePicture.blank(200, 100)
    pic.addOvalFilled(DARK, 10, 10, 50, 50)

    out = crop_and_center(pic)

    pixels = _pixels(out)
    side = pixels.shape[0]
    assert pixels.shape == (side, side, 3)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    assert (border == 255).all()
    rows, cols = np.nonzero(pixels[..., 0] < 128)
    assert abs((rows.min() + rows.max() + 1) / 2 - side / 2) <= 1
    assert abs((cols.min() + cols.max() + 1) / 2 - side / 2) <= 1
    assert (rows.max() - rows.min() + 1) / side == pytest.approx(0.65, abs=0.05)


def test_a_blank_image_is_rejected():
    with pytest.raises(PictureError, match="no figure found"):
        crop_and_center(FakePicture.blank(50, 40))


def test_a_picture_in_gives_a_new_rgb_picture_and_leaves_the_original():
    pic = FakePicture.blank(80, 60)
    pic.addOvalFilled(DARK, 5, 5, 20, 20)
    before = _pixels(pic).copy()

    out = crop_and_center(pic)

    assert type(out) is FakePicture and out is not pic
    assert out.getImage().mode == "RGB"
    np.testing.assert_array_equal(_pixels(pic), before)


@pytest.mark.parametrize("shape", [(60, 80), (60, 80, 1)])
def test_a_grayscale_array_in_gives_an_hxwx1_array_and_leaves_the_original(shape):
    image = np.full(shape, 230, np.uint8)
    image[10:30, 20:40] = 30
    before = image.copy()

    out = crop_and_center(image)

    assert isinstance(out, np.ndarray) and out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[0] == out.shape[1] and out.shape[2] == 1
    np.testing.assert_array_equal(image, before)


def test_colour_is_accepted_and_the_canvas_takes_the_background_colour():
    image = np.zeros((60, 80, 3), np.uint8)
    image[:] = (200, 220, 240)
    image[20:30, 50:60] = (0, 0, 120)

    out = crop_and_center(image)

    assert out.shape[2] == 3
    assert out[0, 0].tolist() == [200, 220, 240]
    assert out[-1, -1].tolist() == [200, 220, 240]
    assert isinstance(crop_and_center(_picture_with_blue_circle()), FakePicture)


def _picture_with_blue_circle() -> FakePicture:
    pic = FakePicture.blank(300, 200)
    pic.addOvalFilled((0, 0, 140), 180, 30, 90, 90)
    return pic


def _triangle(radius: float, angle_degrees: float) -> FakePicture:
    pic = FakePicture.blank(300, 300)
    points = [
        (
            150 + radius * math.cos(math.radians(angle_degrees + 120 * k)),
            150 + radius * math.sin(math.radians(angle_degrees + 120 * k)),
        )
        for k in range(3)
    ]
    ImageDraw.Draw(pic.getImage()).polygon(points, fill=DARK)
    return pic


@pytest.mark.parametrize("angle", [-90, 0, 17])
def test_a_triangle_and_a_circle_of_the_same_radius_come_out_at_the_same_scale(angle):
    circle = FakePicture.blank(300, 300)
    circle.addOvalFilled(DARK, 110, 110, 80, 80)

    circle_side = _pixels(crop_and_center(circle)).shape[0]
    triangle_side = _pixels(crop_and_center(_triangle(40, angle))).shape[0]

    assert triangle_side == pytest.approx(circle_side, rel=0.03)


# scale_down


def test_a_large_square_drawing_is_shrunk_and_a_thin_outline_survives():
    pic = FakePicture.blank(200, 200)
    ImageDraw.Draw(pic.getImage()).ellipse([30, 30, 170, 170], outline=(0, 0, 0), width=2)

    out = scale_down(pic)

    pixels = _pixels(out)[..., 0].astype(int)
    assert pixels.shape == (28, 28)
    # Sample the ring where it crosses the middle row and column.
    ring = [pixels[14, :8].min(), pixels[14, 20:].min(), pixels[:8, 14].min(), pixels[20:, 14].min()]
    assert max(ring) < 255 - 40


def test_a_non_square_image_is_rejected_with_guidance():
    with pytest.raises(PictureError, match=r"square.*crop_and_center\(\)"):
        scale_down(FakePicture.blank(200, 100))


def test_an_image_smaller_than_size_is_rejected():
    with pytest.raises(PictureError, match="only shrink"):
        scale_down(np.zeros((20, 20, 1), np.uint8))


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((56, 56), (28, 28, 1)), ((56, 56, 1), (28, 28, 1)), ((56, 56, 3), (28, 28, 3))],
)
def test_arrays_keep_their_channel_count_and_are_left_unchanged(shape, expected):
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    before = image.copy()

    out = scale_down(image)

    assert out.shape == expected and out.dtype == np.uint8
    np.testing.assert_array_equal(image, before)


def test_a_picture_in_gives_a_new_rgb_picture():
    pic = _colourful_picture(56)
    before = _pixels(pic).copy()

    out = scale_down(pic)

    assert type(out) is FakePicture and out.getImage().mode == "RGB"
    assert out.getWidth() == out.getHeight() == 28
    np.testing.assert_array_equal(_pixels(pic), before)


def test_a_gray_picture_stays_exactly_gray():
    gray = np.random.default_rng(0).integers(0, 256, (200, 200), dtype=np.uint8)
    pic = FakePicture(Image.fromarray(gray).convert("RGB"))

    assert picture_to_array(scale_down(pic)).shape == (28, 28, 1)


def test_grayscale_conversion_can_come_before_or_after_scaling():
    pic = _colourful_picture()

    gray_first = scale_down(_make_greyscale(pic))
    gray_last = _make_greyscale(scale_down(pic))

    a, b = picture_to_array(gray_first), picture_to_array(gray_last)
    assert a.shape == b.shape == (28, 28, 1)
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 2


# save_images


def test_save_images_writes_rgb_pngs_in_order_into_a_new_folder(tmp_path):
    images = np.random.default_rng(0).integers(0, 256, (3, 28, 28, 1), dtype=np.uint8)
    folder = tmp_path / "new" / "folder"

    paths = save_images(images, folder)

    assert [Path(p).name for p in paths] == ["image_000.png", "image_001.png", "image_002.png"]
    for image, path in zip(images, paths):
        saved = Image.open(path)
        assert saved.mode == "RGB" and saved.size == (224, 224)
        pixels = np.asarray(saved)
        np.testing.assert_array_equal(pixels[::8, ::8, 0], image[..., 0])
        blocks = pixels.reshape(28, 8, 28, 8, 3)
        assert (blocks == blocks[:, :1, :, :1]).all()


def test_save_images_accepts_a_single_image(tmp_path):
    image = np.zeros((28, 28, 3), np.uint8)

    assert save_images(image, tmp_path, scale=2) == [str(tmp_path / "image_000.png")]
    assert Image.open(tmp_path / "image_000.png").size == (56, 56)


def test_a_relative_folder_gives_absolute_paths(tmp_path, monkeypatch):
    # mediaComp's makePicture() puts its media folder in front of a relative path.
    monkeypatch.chdir(tmp_path)

    paths = save_images(np.zeros((28, 28, 1), np.uint8), "generated")

    assert paths == [str(tmp_path.resolve() / "generated" / "image_000.png")]
    assert Path(paths[0]).is_absolute()


def test_saving_and_re_preparing_round_trips(tmp_path):
    images = np.random.default_rng(1).integers(0, 256, (2, 28, 28, 1), dtype=np.uint8)

    for image, path in zip(images, save_images(images, tmp_path)):
        pic = FakePicture(Image.open(path))
        np.testing.assert_array_equal(picture_to_array(scale_down(pic)), image)


# predict() with pictures


@pytest.fixture(scope="module")
def model():
    data = make_stub_dataset(n_per_class=4, height=16, width=16, channels=1)
    return build_classifier(data), data


def test_predicting_from_a_prepared_picture_matches_the_array(model):
    classifier, data = model
    for image in data.images[:4]:
        pic = FakePicture(Image.fromarray(image[..., 0]).convert("RGB"))

        assert predict(classifier, pic) == predict(classifier, picture_to_array(pic))


def test_a_picture_of_the_wrong_size_is_rejected_not_resized(model):
    classifier, _ = model

    with pytest.raises(ShapeError, match=r"\(200, 200, 1\)"):
        predict(classifier, FakePicture.blank(200, 200))


def test_a_color_picture_is_rejected_by_predict_with_the_same_error(model):
    classifier, _ = model
    pic = FakePicture.blank(16, 16, (10, 20, 30))

    with pytest.raises(PictureError, match="grayscale first"):
        predict(classifier, pic)


# End to end


def test_a_drawn_circle_is_classified_through_the_whole_pipeline():
    # The stub's seventh shape is a circle; trains in about a second.
    torch.manual_seed(0)
    classes = ["square", "cross", "ring", "hbar", "vbar", "diamond", "circle"]
    data = make_stub_dataset(
        n_per_class=64, height=28, width=28, channels=1, class_names=classes, kind="shapes"
    )
    classifier = build_classifier(data)
    train(classifier, data)

    dark = FakePicture.blank(300, 200)
    dark.addOvalFilled(DARK, 30, 20, 100, 100)
    blue = FakePicture.blank(300, 200)
    blue.addOvalFilled((0, 0, 140), 30, 20, 100, 100)

    assert predict(classifier, scale_down(crop_and_center(dark))) == "circle"
    gray_first = predict(classifier, scale_down(crop_and_center(_make_greyscale(blue))))
    gray_last = predict(classifier, _make_greyscale(scale_down(crop_and_center(blue))))
    assert gray_first == gray_last == "circle"
