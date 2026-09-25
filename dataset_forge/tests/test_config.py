import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import json  # noqa: E402

from dataset_forge.config import ForgeConfig  # noqa: E402
from dataset_forge.export import DEFAULT_CONFIG  # noqa: E402
from dataset_forge.tests._configs import tiny_config  # noqa: E402


def test_config_round_trips_through_json(tmp_path):
    config = tiny_config(background_range=(100, 200), noise_sigma=3.5)
    path = tmp_path / "config.json"

    config.save(path)

    assert ForgeConfig.load(path) == config


def test_default_config_is_the_seven_class_28px_grayscale_course_set():
    config = ForgeConfig.load(DEFAULT_CONFIG)

    assert config.class_names == (
        "square",
        "ring",
        "circle",
        "triangle",
        "star",
        "smiley",
        "negative_smiley",
    )
    assert (config.height, config.width, config.channels) == (28, 28, 1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(class_names=()), "at least one class"),
        (dict(class_names=("ring", "ring")), "duplicates"),
        (dict(train_per_class=0), "train_per_class"),
        (dict(test_per_class=-1), "test_per_class"),
        (dict(height=0), "height"),
        (dict(channels=3), "grayscale"),
        (dict(background_range=(100, 50)), "background_range"),
        (dict(background_range=(50, 200), min_contrast=96), "background_range.*min_contrast"),
        (dict(radius_jitter=1.0), "jitter"),
        (dict(radius_fraction=1.0), "does not fit"),
    ],
)
def test_invalid_configs_are_rejected_with_a_clear_message(overrides, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**overrides)


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**tiny_config().to_dict(), "colour": "red"}))

    with pytest.raises(ValueError, match="colour"):
        ForgeConfig.load(path)


def test_picofaces_framing_constant_matches_the_default_radius_fraction():
    # picoface can't import the Forge, so `crop_and_center()` keeps its own copy.
    from picoface._internals.picture_internals import RADIUS_FRACTION

    assert RADIUS_FRACTION == tiny_config().radius_fraction
    assert RADIUS_FRACTION == ForgeConfig.load(DEFAULT_CONFIG).radius_fraction
