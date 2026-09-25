"""Use mediaComp pictures with picoface, and picoface's images in mediaComp.

A *picture* is a mediaComp `Picture`, or anything that behaves like one (its
`getImage()` returns a Pillow image). An *image array* is picoface's own
format: a uint8 numpy array shaped (height, width, channels). picoface never
imports mediaComp; install it separately with `pip install mediaComp`.

To classify a drawing: draw it, convert it to grayscale with your own code,
then `crop_and_center()`, `scale_down()`, and `predict()`. The grayscale step
can come anywhere before `predict()`, because `crop_and_center()` and
`scale_down()` work on colour and grayscale alike. To look at picoface's
images in mediaComp, `save_images()` them and open a path with `makePicture()`.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image

from picoface._internals.errors import PictureError
from picoface._internals.image_checks import check_image_array
from picoface._internals.picture_internals import (
    FIGURE_THRESHOLD,
    RADIUS_FRACTION,
    enclosing_circle,
    is_picture,
    like_input,
    picture_to_gray_array,
    to_working_array,
)

__all__ = [
    "PictureError",
    "picture_to_array",
    "crop_and_center",
    "scale_down",
    "save_images",
]


def picture_to_array(picture) -> np.ndarray:
    """Return a grayscale picture as a uint8 image array shaped (height, width, 1).

    Row `y`, column `x` of the result is the picture's pixel at `(x, y)`.
    Raises `PictureError` if any pixel's red, green, and blue values differ:
    convert the picture to grayscale first. This function never converts it.
    """
    if not is_picture(picture):
        raise TypeError(
            f"picture_to_array() needs a picture (such as one from mediaComp's "
            f"makePicture()), but was given a {type(picture).__name__}."
        )
    return picture_to_gray_array(picture)


def crop_and_center(image):
    """Return a new square image with the figure centered, framed like the training images.

    `image` is a picture or an image array, in colour or grayscale. The
    background colour is taken from the image's outermost rows and columns,
    and the figure is every pixel whose brightness clearly differs from it.
    The new canvas is filled with the background colour and sized so the
    figure fills about the same share of it as a figure does in a default
    Dataset Forge image; the figure itself is never shrunk. A picture gives
    back a new picture of the same type, an image array gives back an image
    array with the same number of channels. `image` is not changed.

    Raises `PictureError` if no figure is found.
    """
    array = to_working_array(image)
    height, width, channels = array.shape

    border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])
    background = np.median(border, axis=0)
    brightness = array.mean(axis=2)
    rows, cols = np.nonzero(np.abs(brightness - background.mean()) > FIGURE_THRESHOLD)
    if len(rows) == 0:
        raise PictureError(
            "no figure found: every pixel is close to the background colour. "
            "Draw one figure that is clearly darker or lighter than its background."
        )

    # Frame the smallest circle around the figure, as the Forge sizes figures.
    centre_y, centre_x, radius = enclosing_circle(rows, cols)
    side = max(
        math.ceil(2 * radius / RADIUS_FRACTION),
        rows.max() - rows.min() + 1,
        cols.max() - cols.min() + 1,
    )

    canvas = np.empty((side, side, channels), np.uint8)
    canvas[:] = np.rint(background).astype(np.uint8)
    top = round(side / 2 - centre_y)
    left = round(side / 2 - centre_x)
    src_top, src_left = max(0, -top), max(0, -left)
    src_bottom, src_right = min(height, side - top), min(width, side - left)
    canvas[src_top + top : src_bottom + top, src_left + left : src_right + left] = array[
        src_top:src_bottom, src_left:src_right
    ]
    return like_input(canvas, image)


def scale_down(image, size: int = 28):
    """Return a new image shrunk to `size`×`size` pixels, with antialiasing.

    `image` is a square picture or image array, in colour or grayscale. Each
    new pixel is the average of the pixels it covers, in each colour channel,
    the same smoothing the training images were made with, so thin lines fade
    to gray rather than vanish. A picture gives back a new picture of the same
    type, an image array gives back an image array with the same number of
    channels. `image` is not changed.

    Raises `PictureError` if `image` is not square (use `crop_and_center()`
    first) or is smaller than `size`.
    """
    array = to_working_array(image)
    height, width, channels = array.shape
    if height != width:
        raise PictureError(
            f"scale_down() needs a square image, but this one is {width} wide and "
            f"{height} tall. Use crop_and_center() first to make it square."
        )
    if height < size:
        raise PictureError(
            f"scale_down() can only shrink an image, but this one is {height}×{width}, "
            f"smaller than size={size}."
        )

    scaled = np.stack(
        [
            np.asarray(Image.fromarray(array[..., k]).resize((size, size), Image.Resampling.BOX))
            for k in range(channels)
        ],
        axis=-1,
    )
    return like_input(scaled, image)


def save_images(images: np.ndarray, folder, scale: int = 8) -> list[str]:
    """Save image arrays as PNG files that mediaComp's `makePicture()` can open.

    `images` is one image array shaped (height, width, channels) or several
    shaped (count, height, width, channels), such as the result of
    `generate()` or `activation_maximize()`, or a `classify_generated()`
    report's `images`. Each is written to `folder` (created if needed) as an
    RGB PNG named `image_000.png`, `image_001.png`, ..., enlarged `scale`
    times so each pixel becomes a solid `scale`×`scale` block. Files with the
    same names are overwritten.

    Returns the files' absolute paths, as strings, in image order.
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = images[np.newaxis]
    check_image_array(images, name="images", batch=True, error_cls=PictureError)
    if not isinstance(scale, int) or scale < 1:
        raise PictureError(f"scale must be a whole number of at least 1, got {scale!r}.")

    # Absolute, because makePicture() puts its media folder in front of a relative path.
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    digits = max(3, len(str(len(images) - 1)))
    paths = []
    for i, image in enumerate(images):
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        enlarged = image.repeat(scale, axis=0).repeat(scale, axis=1)
        path = folder / f"image_{i:0{digits}d}.png"
        Image.fromarray(enlarged).save(path)
        paths.append(str(path))
    return paths
