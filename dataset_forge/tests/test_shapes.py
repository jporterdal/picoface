import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import math  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dataset_forge.shapes import SHAPES, Placement, check_class_names, draw_figure  # noqa: E402

SIZE = 96
COURSE_CLASSES = ["square", "ring", "circle", "triangle", "star", "smiley", "negative_smiley"]


def _mask(class_name: str, angle: float) -> np.ndarray:
    canvas = Image.new("L", (SIZE, SIZE), 0)
    placement = Placement(cx=SIZE / 2, cy=SIZE / 2, radius=30, angle=angle, stroke=6)
    draw_figure(ImageDraw.Draw(canvas), class_name, placement)
    return np.asarray(canvas)


def test_registry_has_the_course_classes():
    assert list(SHAPES) == COURSE_CLASSES


def test_unknown_class_names_are_named_along_with_the_available_ones():
    with pytest.raises(ValueError, match="hexagon") as error:
        check_class_names(["circle", "hexagon"])

    assert "negative_smiley" in str(error.value)


@pytest.mark.parametrize("class_name", COURSE_CLASSES)
@pytest.mark.parametrize("angle", [0.0, 0.7, math.pi / 2, 4.0])
def test_every_class_draws_ink_inside_its_circumscribed_circle(class_name, angle):
    mask = _mask(class_name, angle)

    rows, cols = np.nonzero(mask)
    distance = np.hypot(rows + 0.5 - SIZE / 2, cols + 0.5 - SIZE / 2)
    assert len(rows) > 0
    # A touched pixel's centre can sit half a diagonal past the edge, plus
    # the rasterizer's one-pixel inclusive outline.
    assert distance.max() <= 30 + 2


def test_smileys_look_different_when_rotated_but_circles_do_not():
    for class_name in ("smiley", "negative_smiley"):
        assert not np.array_equal(_mask(class_name, 0.0), _mask(class_name, math.pi))
    assert np.array_equal(_mask("circle", 0.0), _mask("circle", math.pi))
