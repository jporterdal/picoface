"""The settings one export is rendered from, loaded from and saved as JSON."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Figures keep at least this many output pixels clear of every image edge,
# so no anti-aliased ink ever lands in the outermost rows and columns.
EDGE_MARGIN = 1.0


@dataclass(frozen=True)
class ForgeConfig:
    """Everything that determines an export, except the seed.

    Sizes are relative to a figure's circumscribed radius (the smallest circle
    around it), which is what "the same size" means across classes. Shades are
    gray levels 0-255; the foreground is always darker than the background.
    """

    name: str
    class_names: tuple[str, ...]
    height: int = 28
    width: int = 28
    channels: int = 1
    train_per_class: int = 1000
    test_per_class: int = 200
    # Nominal circumscribed radius, as a fraction of half the image's shorter side.
    radius_fraction: float = 0.65
    # Each figure's radius varies by up to this fraction either way.
    radius_jitter: float = 0.15
    # Outline and face-feature stroke width, as a fraction of the radius.
    stroke_fraction: float = 0.18
    # Each figure's stroke varies by up to this fraction either way.
    stroke_jitter: float = 0.15
    # Strokes never get thinner than this many output pixels.
    min_stroke: float = 1.0
    # Background shade range, inclusive.
    background_range: tuple[int, int] = (96, 255)
    # The foreground is at least this much darker than the background.
    min_contrast: int = 96
    # Standard deviation of per-pixel Gaussian noise, in gray levels.
    noise_sigma: float = 6.0
    # Figures are drawn at this many times the output resolution per side,
    # then averaged down, which anti-aliases their edges.
    supersample: int = 4

    def __post_init__(self):
        if not self.class_names:
            raise ValueError("class_names must list at least one class.")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(f"class_names has duplicates: {list(self.class_names)!r}.")
        for field_name in ("height", "width", "train_per_class", "test_per_class", "supersample"):
            value = getattr(self, field_name)
            if value < 1:
                raise ValueError(f"{field_name} must be at least 1, got {value}.")
        if self.channels != 1:
            raise ValueError(f"only grayscale (channels=1) is supported, got {self.channels}.")
        low, high = self.background_range
        if not 0 <= low <= high <= 255:
            raise ValueError(
                "background_range must satisfy 0 <= low <= high <= 255, "
                f"got {self.background_range}."
            )
        if low - self.min_contrast < 0:
            raise ValueError(
                f"a background of {low} leaves no foreground shade at least {self.min_contrast} "
                "darker; raise background_range's lower end or lower min_contrast."
            )
        if not 0 <= self.radius_jitter < 1 or not 0 <= self.stroke_jitter < 1:
            raise ValueError("radius_jitter and stroke_jitter must be in [0, 1).")
        half_side = min(self.height, self.width) / 2
        if self.max_radius + self.clearance > half_side:
            raise ValueError(
                f"the largest figure (radius {self.max_radius:.2f} px) does not fit in a "
                f"{self.height}x{self.width} image; lower radius_fraction or radius_jitter."
            )

    @property
    def nominal_radius(self) -> float:
        """The unjittered circumscribed radius, in output pixels."""
        return self.radius_fraction * min(self.height, self.width) / 2

    @property
    def max_radius(self) -> float:
        return self.nominal_radius * (1 + self.radius_jitter)

    @property
    def clearance(self) -> float:
        """How far, in output pixels, a figure's circumscribed circle stays from every edge.

        `EDGE_MARGIN`, plus one supersampled pixel: the rasterizer fills
        outlines inclusively, so ink can land up to one canvas pixel past a
        figure's true edge.
        """
        return EDGE_MARGIN + 1 / self.supersample

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ForgeConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)!r}.")
        values = dict(values)
        for key in ("class_names", "background_range"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "ForgeConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
