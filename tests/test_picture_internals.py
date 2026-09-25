"""mediacomp-bridge: the shared picture ↔ image-array conversion."""

import numpy as np
import pytest
from fake_picture import FakePicture
from PIL import Image

from picoface._internals.errors import PictureError
from picoface._internals.picture_internals import (
    array_to_picture,
    is_picture,
    like_input,
    picture_to_gray_array,
    picture_to_rgb_array,
    to_working_array,
)


class _NotAPicture:
    """Has getImage(), but it returns no Pillow image."""

    def getImage(self):
        return None


def test_detection_needs_a_getimage_that_returns_a_pillow_image():
    assert is_picture(FakePicture.blank(4, 3))
    assert not is_picture(_NotAPicture())
    assert not is_picture(np.zeros((4, 4, 1), np.uint8))
    assert not is_picture("drawing.png")


def test_a_color_picture_is_rejected_with_guidance_naming_the_pixel():
    pic = FakePicture.blank(5, 4, (90, 90, 90))
    pic.setBasicPixel(3, 1, (90, 91, 90))

    with pytest.raises(PictureError, match=r"not grayscale.*\(3, 1\).*grayscale first"):
        picture_to_gray_array(pic)


def test_grayscale_pictures_in_any_mode_convert():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)

    for mode in ("L", "RGB", "RGBA"):
        pic = FakePicture(Image.fromarray(gray).convert(mode))
        np.testing.assert_array_equal(picture_to_gray_array(pic), gray[..., np.newaxis])


def test_the_rgb_reader_does_not_check_for_gray():
    pic = FakePicture.blank(2, 2, (10, 20, 30))

    assert picture_to_rgb_array(pic).tolist() == [[[10, 20, 30]] * 2] * 2


@pytest.mark.parametrize("channels", [1, 3])
def test_array_to_picture_to_array_round_trips_exactly(channels):
    array = np.random.default_rng(0).integers(0, 256, size=(5, 7, channels), dtype=np.uint8)

    pic = array_to_picture(array, FakePicture)

    assert type(pic) is FakePicture and pic.getImage().mode == "RGB"
    back = picture_to_gray_array(pic) if channels == 1 else picture_to_rgb_array(pic)
    np.testing.assert_array_equal(back, array)


def test_working_arrays_gain_a_channel_axis_and_are_checked():
    assert to_working_array(np.zeros((4, 6), np.uint8)).shape == (4, 6, 1)
    assert to_working_array(np.zeros((4, 6, 3), np.uint8)).shape == (4, 6, 3)
    assert to_working_array(FakePicture.blank(6, 4)).shape == (4, 6, 3)

    with pytest.raises(PictureError, match="uint8"):
        to_working_array(np.zeros((4, 6), np.float32))
    with pytest.raises(PictureError, match="channels"):
        to_working_array(np.zeros((4, 6, 2), np.uint8))


def test_results_come_back_in_the_kind_given():
    array = np.full((3, 3, 1), 7, np.uint8)

    assert like_input(array, np.zeros((9, 9), np.uint8)) is array
    assert isinstance(like_input(array, FakePicture.blank(9, 9)), FakePicture)
