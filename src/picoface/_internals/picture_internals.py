"""Moving images between pictures and image arrays, shared by `pictures` and `predict()`.

Not part of the public API. A *picture* is any object that behaves like a
mediaComp `Picture`: its `getImage()` returns a Pillow image, and its type can
be called with a Pillow image to make a new one. mediaComp itself is never
imported. An *image array* is picoface's uint8 H×W×C format.

`picoface.pictures`' helpers work on H×W×C arrays (C = 1 or 3) and convert only
at the edges: a picture is read as RGB with no grayscale check, so colour and
grayscale pictures both work; only `picture_to_gray_array()` requires gray.
"""

import math

import numpy as np
from PIL import Image

from picoface._internals.errors import PictureError
from picoface._internals.image_checks import check_image_array

# A figure's circumscribed radius as a fraction of half the image's side: the
# Dataset Forge's default `radius_fraction`. A Forge test keeps the two equal.
RADIUS_FRACTION = 0.65

# A pixel belongs to the figure when its brightness differs from the
# background's by more than this: half the Forge's default `min_contrast`.
FIGURE_THRESHOLD = 48


def is_picture(obj) -> bool:
    """Whether `obj` has a callable `getImage()` that returns a Pillow image."""
    get_image = getattr(obj, "getImage", None)
    return callable(get_image) and isinstance(get_image(), Image.Image)


def picture_to_rgb_array(picture) -> np.ndarray:
    """A picture as an H×W×3 uint8 RGB array; any alpha channel is ignored."""
    return np.array(picture.getImage().convert("RGB"), dtype=np.uint8)


def picture_to_gray_array(picture) -> np.ndarray:
    """A grayscale picture as an H×W×1 uint8 image array.

    Raises `PictureError`, naming the first non-gray pixel and the fix, if any
    pixel's red, green, and blue values differ. Never converts the picture.
    """
    rgb = picture_to_rgb_array(picture)
    not_gray = (rgb[..., 0] != rgb[..., 1]) | (rgb[..., 1] != rgb[..., 2])
    if not_gray.any():
        rows, cols = np.nonzero(not_gray)
        y, x = int(rows[0]), int(cols[0])
        red, green, blue = (int(v) for v in rgb[y, x])
        raise PictureError(
            f"this picture is not grayscale: the pixel at ({x}, {y}) has red {red}, "
            f"green {green}, blue {blue}. picoface needs a grayscale picture, where "
            f"every pixel's red, green, and blue values are equal. Convert the "
            f"picture to grayscale first, for example with a loop that sets each "
            f"pixel's red, green, and blue to their average."
        )
    return rgb[..., :1].copy()


def to_working_array(image, *, name: str = "image") -> np.ndarray:
    """A picture or image array as an H×W×C uint8 array (C = 1 or 3).

    A picture is read as RGB. An H×W array gets a channel axis. Anything that
    is still not an image array raises `PictureError`.
    """
    if is_picture(image):
        return picture_to_rgb_array(image)
    if isinstance(image, np.ndarray) and image.ndim == 2:
        image = image[..., np.newaxis]
    check_image_array(image, name=name, batch=False, error_cls=PictureError)
    return image


def array_to_picture(array: np.ndarray, picture_type: type):
    """A new picture of `picture_type`, in RGB mode, holding an H×W×C array."""
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return picture_type(Image.fromarray(np.ascontiguousarray(array)))


def like_input(array: np.ndarray, original):
    """`array` in the kind `original` was: a new picture of its type, or an array."""
    if is_picture(original):
        return array_to_picture(array, type(original))
    return array


def enclosing_circle(rows: np.ndarray, cols: np.ndarray) -> tuple[float, float, float]:
    """The smallest circle holding every given pixel: (centre_y, centre_x, radius).

    Coordinates are continuous (pixel k spans [k, k+1]). Only each row's
    outermost pixels can touch the circle, so only those are searched, with
    Welzl's algorithm; the radius then grows by half a pixel to cover the
    pixels themselves, not just their centres.
    """
    ys, xs = [], []
    for row in np.unique(rows):
        in_row = cols[rows == row]
        ys += [row, row]
        xs += [in_row.min(), in_row.max()]
    order = np.random.default_rng(0).permutation(len(ys))
    points = [(ys[i] + 0.5, xs[i] + 0.5) for i in order]

    circle = (*points[0], 0.0)
    for i, p in enumerate(points):
        if _outside(p, circle):
            circle = (*p, 0.0)
            for j, q in enumerate(points[:i]):
                if _outside(q, circle):
                    circle = _circle_on(p, q)
                    for s in points[:j]:
                        if _outside(s, circle):
                            circle = _circle_through(p, q, s)
    centre_y, centre_x, radius = circle
    return centre_y, centre_x, radius + 0.5


def _outside(point, circle) -> bool:
    y, x, r = circle
    return math.hypot(point[0] - y, point[1] - x) > r + 1e-9


def _circle_on(p, q) -> tuple[float, float, float]:
    """The circle with `p` and `q` at opposite ends of a diameter."""
    return (p[0] + q[0]) / 2, (p[1] + q[1]) / 2, math.hypot(p[0] - q[0], p[1] - q[1]) / 2


def _circle_through(p, q, s) -> tuple[float, float, float]:
    """The circle through three points; for collinear points, the one on the farthest pair."""
    (ay, ax), (by, bx), (cy, cx) = p, q, s
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return max((_circle_on(p, q), _circle_on(p, s), _circle_on(q, s)), key=lambda c: c[2])
    a2, b2, c2 = ax**2 + ay**2, bx**2 + by**2, cx**2 + cy**2
    x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return y, x, math.hypot(ay - y, ax - x)
