import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import math  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from dataset_forge.render import render_coverage, render_image, sample_params, shade  # noqa: E402
from dataset_forge.shapes import _EYE_OFFSETS, SHAPES, Placement  # noqa: E402
from dataset_forge.tests._configs import default_config  # noqa: E402

CONFIG = default_config()


def _draws(class_name: str, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [sample_params(rng, CONFIG, class_name) for _ in range(n)]


@pytest.mark.parametrize("class_name", list(SHAPES))
def test_figures_are_never_clipped(class_name):
    for params in _draws(class_name):
        coverage = render_coverage(params, CONFIG)
        border = np.concatenate([coverage[0], coverage[-1], coverage[:, 0], coverage[:, -1]])
        assert not border.any()


@pytest.mark.parametrize("class_name", list(SHAPES))
@pytest.mark.parametrize("corner", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_the_largest_figure_pushed_into_a_corner_is_not_clipped(class_name, corner):
    room = CONFIG.max_radius + CONFIG.clearance
    cx = room if corner[0] == 0 else CONFIG.width - room
    cy = room if corner[1] == 0 else CONFIG.height - room
    for angle in np.linspace(0, 2 * math.pi, 13):
        params = replace(
            _draws(class_name, n=1)[0],
            placement=Placement(cx=cx, cy=cy, radius=CONFIG.max_radius, angle=angle, stroke=2.0),
        )
        coverage = render_coverage(params, CONFIG)
        border = np.concatenate([coverage[0], coverage[-1], coverage[:, 0], coverage[:, -1]])
        assert not border.any()


def test_foreground_is_always_darker_by_at_least_the_minimum_contrast():
    for params in _draws("circle", n=2000):
        low, high = CONFIG.background_range
        assert low <= params.background <= high
        assert params.background - params.foreground >= CONFIG.min_contrast
        assert params.foreground >= 0


def test_variation_covers_rotation_size_position_and_shades():
    draws = _draws("square", n=500)

    def spread(values):
        return max(values) - min(values)

    assert spread([p.placement.angle for p in draws]) > 1.9 * math.pi
    radius_range = 2 * CONFIG.radius_jitter * CONFIG.nominal_radius
    assert spread([p.placement.radius for p in draws]) > 0.75 * radius_range
    assert spread([p.placement.cx for p in draws]) > 2
    assert spread([p.placement.cy for p in draws]) > 2
    assert spread([p.background for p in draws]) > 100


def test_images_are_28x28_grayscale_uint8_and_never_repeat():
    rng = np.random.default_rng(0)

    images = [render_image(rng, CONFIG, "star") for _ in range(20)]

    assert all(image.shape == (28, 28, 1) and image.dtype == np.uint8 for image in images)
    assert len({image.tobytes() for image in images}) == len(images)


def test_full_coverage_without_noise_is_exactly_the_foreground_shade():
    params = _draws("circle", n=1)[0]
    rng = np.random.default_rng(0)

    image = shade(np.ones((28, 28)), params, noise_sigma=0.0, rng=rng)

    assert np.all(image == round(params.foreground))


def test_the_negative_smiley_shares_the_polarity():
    # A large canvas, so the eyes cover whole pixels.
    config = default_config(height=112, width=112)
    params = sample_params(np.random.default_rng(0), config, "negative_smiley")
    rng = np.random.default_rng(0)

    image = shade(render_coverage(params, config), params, 0.0, rng)[..., 0]

    def at(x, y):
        return image[int(y), int(x)]

    p = params.placement
    background, foreground = round(params.background), round(params.foreground)
    assert foreground < background
    assert at(p.cx, p.cy) == foreground
    eye_u, eye_v = _EYE_OFFSETS[0]
    assert at(*p.point(eye_u * p.radius, eye_v * p.radius)) == background
    assert image[0, 0] == image[0, -1] == image[-1, 0] == image[-1, -1] == background


@pytest.mark.parametrize(
    ("plain", "featured"), [("circle", "negative_smiley"), ("ring", "smiley")]
)
def test_face_classes_differ_from_their_plain_counterparts_only_inside_the_figure(
    plain, featured
):
    rng = np.random.default_rng(0)
    for params in _draws(plain, n=20):
        a = shade(render_coverage(params, CONFIG), params, 0.0, rng)[..., 0]
        b = shade(render_coverage(replace(params, class_name=featured), CONFIG), params, 0.0, rng)
        rows, cols = np.nonzero(a != b[..., 0])

        p = params.placement
        assert len(rows) > 0
        assert np.hypot(rows + 0.5 - p.cy, cols + 0.5 - p.cx).max() <= p.radius
