import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import math  # noqa: E402

from dataset_forge.rotation import ROTATION_DISTRIBUTIONS  # noqa: E402

LIMIT = math.radians(30)
# The largest float below 1: the top of a uniform draw.
TOP = 1 - 2**-53
GRID = [(k + 0.5) / 10_000 for k in range(10_000)]


@pytest.mark.parametrize("name", list(ROTATION_DISTRIBUTIONS))
def test_angles_stay_within_the_range_at_both_ends_of_the_draw(name):
    angle = ROTATION_DISTRIBUTIONS[name]

    assert angle(0.0, LIMIT) == pytest.approx(-LIMIT)
    assert -LIMIT <= angle(0.0, LIMIT)
    assert angle(TOP, LIMIT) == pytest.approx(LIMIT)
    assert angle(TOP, LIMIT) <= LIMIT
    assert angle(0.5, LIMIT) == pytest.approx(0.0)


@pytest.mark.parametrize("name", list(ROTATION_DISTRIBUTIONS))
def test_a_range_of_zero_is_always_upright(name):
    assert all(ROTATION_DISTRIBUTIONS[name](u, 0.0) == 0 for u in (0.0, 0.3, TOP))


def test_uniform_spreads_angles_evenly():
    angles = [ROTATION_DISTRIBUTIONS["uniform"](u, LIMIT) for u in GRID]

    steps = [b - a for a, b in zip(angles, angles[1:])]
    assert max(steps) == pytest.approx(min(steps))


def test_normal_puts_about_72_percent_within_half_the_range():
    # 68.3% of a normal lies within one standard deviation, of the 95.4% kept within two.
    angles = [ROTATION_DISTRIBUTIONS["normal"](u, LIMIT) for u in GRID]

    within = sum(abs(a) <= LIMIT / 2 for a in angles) / len(angles)
    assert within == pytest.approx(0.6827 / 0.9545, abs=0.002)
