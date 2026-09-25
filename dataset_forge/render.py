"""Rendering one image: sample its variation, draw its figure, shade it.

Three separate stages, so draw functions never see shades or noise:
1. `sample_params()` draws the rotation, size, position, stroke, and shades.
2. `render_coverage()` draws the figure's mask on a supersampled canvas and
   averages it down, so each output pixel holds the fraction of its area the
   figure covers — this is the anti-aliasing.
3. `shade()` maps coverage to gray between the two shades and adds noise.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from dataset_forge.config import ForgeConfig
from dataset_forge.shapes import Placement, draw_figure


@dataclass(frozen=True)
class FigureParams:
    """One image's sampled variation. `placement` is in output pixels."""

    class_name: str
    placement: Placement
    background: float
    foreground: float


def sample_params(rng: np.random.Generator, config: ForgeConfig, class_name: str) -> FigureParams:
    """Draw one image's variation within the ranges `config` allows.

    The centre is kept far enough from every edge that the figure's whole
    circumscribed circle, plus `config.clearance`, lies inside the image.
    """
    radius = config.nominal_radius
    radius *= rng.uniform(1 - config.radius_jitter, 1 + config.radius_jitter)
    stroke = config.stroke_fraction * radius
    stroke *= rng.uniform(1 - config.stroke_jitter, 1 + config.stroke_jitter)
    stroke = max(stroke, config.min_stroke)
    room = radius + config.clearance
    cx = rng.uniform(room, config.width - room)
    cy = rng.uniform(room, config.height - room)
    angle = rng.uniform(0, 2 * math.pi)

    low, high = config.background_range
    background = rng.uniform(low, high)
    foreground = rng.uniform(0, background - config.min_contrast)

    return FigureParams(
        class_name=class_name,
        placement=Placement(cx=cx, cy=cy, radius=radius, angle=angle, stroke=stroke),
        background=background,
        foreground=foreground,
    )


def render_coverage(params: FigureParams, config: ForgeConfig) -> np.ndarray:
    """An (H, W) float array in [0, 1]: how much of each pixel the figure covers."""
    s = config.supersample
    p = params.placement
    canvas = Image.new("L", (config.width * s, config.height * s), 0)
    scaled = Placement(
        cx=p.cx * s, cy=p.cy * s, radius=p.radius * s, angle=p.angle, stroke=p.stroke * s
    )
    draw_figure(ImageDraw.Draw(canvas), params.class_name, scaled)

    mask = np.asarray(canvas, dtype=np.float64) / 255
    return mask.reshape(config.height, s, config.width, s).mean(axis=(1, 3))


def shade(
    coverage: np.ndarray, params: FigureParams, noise_sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """An (H, W, 1) uint8 image: coverage mapped between the two shades, plus noise."""
    image = params.background + (params.foreground - params.background) * coverage
    image = image + rng.normal(0, noise_sigma, size=coverage.shape)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)[..., np.newaxis]


def render_image(rng: np.random.Generator, config: ForgeConfig, class_name: str) -> np.ndarray:
    """One freshly varied (H, W, 1) uint8 image of `class_name`."""
    params = sample_params(rng, config, class_name)
    return shade(render_coverage(params, config), params, config.noise_sigma, rng)
