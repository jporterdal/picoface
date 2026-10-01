"""The rotation distributions: how a figure's angle is drawn within its range.

Each distribution turns one uniform draw `u` in [0, 1) into an angle in
radians within ±`limit` of upright. Using exactly one draw, whatever the
distribution, keeps every later draw from the same random stream in place.
"""

from collections.abc import Callable
from statistics import NormalDist

# The normal distribution's standard deviation is half the range, so the
# range's edges sit this many standard deviations from upright.
_NORMAL_CUTOFF = 2.0
_STANDARD = NormalDist()
_LOW, _HIGH = _STANDARD.cdf(-_NORMAL_CUTOFF), _STANDARD.cdf(_NORMAL_CUTOFF)


def _uniform(u: float, limit: float) -> float:
    return (2 * u - 1) * limit


def _normal(u: float, limit: float) -> float:
    """A normal draw centred on upright, redrawn whenever it lands past ±`limit`.

    Inverting the CDF of the normal distribution cut off at the range's edges
    gives exactly the angles redrawing would, from a single draw.
    """
    if limit == 0:
        return 0.0
    return _STANDARD.inv_cdf(_LOW + u * (_HIGH - _LOW)) * limit / _NORMAL_CUTOFF


ROTATION_DISTRIBUTIONS: dict[str, Callable[[float, float], float]] = {
    "uniform": _uniform,
    "normal": _normal,
}
