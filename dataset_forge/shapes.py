"""The class registry: one draw function per class, keyed by class name.

A draw function paints only geometry — figure pixels at 255 on a 0 canvas —
and knows nothing about shades, noise, or how its placement was chosen, so a
new class is one function here plus one entry in a config's `class_names`.
Every figure fits inside the circle of `placement.radius` around its centre.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from PIL import ImageDraw

INK = 255
CUT = 0

# A regular pentagram's inner-to-outer vertex radius ratio.
_STAR_INNER_RATIO = 0.382

# The smiley layout, relative to the face's radius, before rotation (y down).
_EYE_OFFSETS = ((-0.35, -0.25), (0.35, -0.25))
_EYE_RADIUS = 0.14
_MOUTH_RADIUS = 0.5
_MOUTH_SPAN_DEGREES = (30.0, 150.0)  # clockwise from 3 o'clock: the lower arc
_ARC_SEGMENTS = 24


@dataclass(frozen=True)
class Placement:
    """Where and how to draw one figure, in canvas pixels.

    `angle` is in radians, clockwise on screen. `stroke` is the width of
    outlines and face features.
    """

    cx: float
    cy: float
    radius: float
    angle: float
    stroke: float

    def point(self, u: float, v: float) -> tuple[float, float]:
        """The canvas position of local offset (u, v), rotated by `angle`."""
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        return self.cx + u * cos - v * sin, self.cy + u * sin + v * cos


# Canvas coordinates are continuous (pixel k spans [k, k+1]); Pillow addresses
# pixels by index, so every coordinate is shifted by half a pixel.


def _polygon(draw: ImageDraw.ImageDraw, points, fill: int) -> None:
    draw.polygon([(x - 0.5, y - 0.5) for x, y in points], fill=fill)


def _disc(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: int) -> None:
    draw.ellipse((x - r, y - r, x + r - 1, y + r - 1), fill=fill)


def _vertices(p: Placement, radii: list[float], start: float) -> list[tuple[float, float]]:
    """Vertices at the given radii, evenly spaced in angle from `start`."""
    step = 2 * math.pi / len(radii)
    return [
        p.point(r * math.cos(start + k * step), r * math.sin(start + k * step))
        for k, r in enumerate(radii)
    ]


def _regular_polygon(p: Placement, sides: int, start: float):
    return _vertices(p, [p.radius] * sides, start)


def _star_points(p: Placement, points: int, inner_ratio: float, start: float):
    return _vertices(p, [p.radius, p.radius * inner_ratio] * points, start)


def _square(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _polygon(draw, _regular_polygon(p, 4, math.pi / 4), INK)


def _triangle(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _polygon(draw, _regular_polygon(p, 3, -math.pi / 2), INK)


def _star(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _polygon(draw, _star_points(p, 5, _STAR_INNER_RATIO, -math.pi / 2), INK)


def _circle(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _disc(draw, p.cx, p.cy, p.radius, INK)


def _ring(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _disc(draw, p.cx, p.cy, p.radius, INK)
    _disc(draw, p.cx, p.cy, p.radius - p.stroke, CUT)


def _face_features(draw: ImageDraw.ImageDraw, p: Placement, fill: int) -> None:
    """Two eyes and a smiling mouth, in `fill`, rotated with the face."""
    for u, v in _EYE_OFFSETS:
        _disc(draw, *p.point(u * p.radius, v * p.radius), _EYE_RADIUS * p.radius, fill)

    start, end = (math.radians(d) for d in _MOUTH_SPAN_DEGREES)
    angles = [start + (end - start) * k / _ARC_SEGMENTS for k in range(_ARC_SEGMENTS + 1)]
    outer = _MOUTH_RADIUS * p.radius + p.stroke / 2
    inner = _MOUTH_RADIUS * p.radius - p.stroke / 2
    band = [p.point(outer * math.cos(a), outer * math.sin(a)) for a in angles] + [
        p.point(inner * math.cos(a), inner * math.sin(a)) for a in reversed(angles)
    ]
    _polygon(draw, band, fill)


def _smiley(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _ring(draw, p)
    _face_features(draw, p, INK)


def _negative_smiley(draw: ImageDraw.ImageDraw, p: Placement) -> None:
    _circle(draw, p)
    _face_features(draw, p, CUT)


SHAPES: dict[str, Callable[[ImageDraw.ImageDraw, Placement], None]] = {
    "square": _square,
    "ring": _ring,
    "circle": _circle,
    "triangle": _triangle,
    "star": _star,
    "smiley": _smiley,
    "negative_smiley": _negative_smiley,
}


def check_class_names(class_names) -> None:
    """Raise `ValueError` naming any class that has no draw function."""
    unknown = [name for name in class_names if name not in SHAPES]
    if unknown:
        raise ValueError(
            f"no draw function for class(es) {unknown!r}; available classes: {list(SHAPES)!r}."
        )


def draw_figure(draw: ImageDraw.ImageDraw, class_name: str, placement: Placement) -> None:
    check_class_names([class_name])
    SHAPES[class_name](draw, placement)
