import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

from dataset_forge.export import export, read_manifest  # noqa: E402
from dataset_forge.render import render_coverage, sample_params, shade  # noqa: E402
from dataset_forge.tests._configs import default_config, tiny_config  # noqa: E402
from dataset_forge.tests._exports import load_split, offset_class, rewrite_split  # noqa: E402
from dataset_forge.validate import ink_fraction, validate_and_record, validate_export  # noqa: E402


@pytest.fixture(scope="module")
def course_export(tmp_path_factory):
    """All seven course classes at a reduced count: enough for stable baselines."""
    config = default_config(train_per_class=120, test_per_class=60)
    return export(config, seed=0, out_dir=tmp_path_factory.mktemp("course"))


def test_the_course_classes_pass_with_ink_fraction_well_above_the_background_gate(course_export):
    report = validate_export(course_export)

    assert report.passed, str(report)
    assert report.chance == pytest.approx(1 / 7)
    assert report.background_accuracy < report.chance + 0.1
    # Not gated: reported only, and here clearly informative.
    assert report.ink_fraction_accuracy > report.chance + 0.1


def test_results_are_recorded_in_the_manifest(tmp_path):
    out_dir = export(tiny_config(), seed=0, out_dir=tmp_path)

    report = validate_and_record(out_dir)

    recorded = read_manifest(out_dir)["validation"]
    assert recorded["passed"] == report.passed
    assert set(recorded["checks"]) == {"loads", "balanced", "disjoint", "background"}
    assert recorded["background_accuracy"] == report.background_accuracy
    assert recorded["ink_fraction_accuracy"] == report.ink_fraction_accuracy


def test_a_class_dependent_background_fails_the_background_check(tmp_path):
    # A narrow background range, then each class lightened by a different amount.
    out_dir = export(tiny_config(background_range=(130, 150)), seed=0, out_dir=tmp_path)
    offset_class(out_dir, label=1, amount=40)
    offset_class(out_dir, label=2, amount=80)

    report = validate_export(out_dir)

    assert not report.passed
    assert report.failed_checks == ["background"]


def test_brightness_that_follows_figure_area_is_reported_not_gated(tmp_path):
    # Fixed shades and no noise: a filled circle is simply darker than a ring.
    config = tiny_config(
        class_names=("circle", "ring"),
        train_per_class=40,
        test_per_class=20,
        background_range=(250, 250),
        min_contrast=210,
        noise_sigma=0.0,
    )
    out_dir = export(config, seed=0, out_dir=tmp_path)

    report = validate_export(out_dir)

    assert report.passed, str(report)
    assert report.mean_brightness_accuracy > report.chance + 0.1


def test_an_image_in_both_splits_fails_the_disjointness_check(tmp_path):
    out_dir = export(tiny_config(), seed=0, out_dir=tmp_path)
    train_images, train_labels = load_split(out_dir, "train")
    test_images, _ = load_split(out_dir, "test")
    train_images[0] = test_images[0]
    rewrite_split(out_dir, "train", train_images, train_labels)

    assert "disjoint" in validate_export(out_dir).failed_checks


def test_a_class_missing_from_a_split_fails_the_balance_check(tmp_path):
    out_dir = export(tiny_config(), seed=0, out_dir=tmp_path)
    images, labels = load_split(out_dir, "test")
    keep = labels != 2
    rewrite_split(out_dir, "test", images[keep], labels[keep])

    report = validate_export(out_dir)

    assert "balanced" in report.failed_checks
    assert "test has per-class counts [4, 4, 0]" in str(report)


def test_a_bundle_with_the_wrong_image_shape_fails_the_load_check(tmp_path):
    out_dir = export(tiny_config(), seed=0, out_dir=tmp_path)
    images, labels = load_split(out_dir, "train")
    rewrite_split(out_dir, "train", images[:, :20, :20], labels)

    assert validate_export(out_dir).failed_checks == ["loads"]


def test_ink_fraction_tracks_how_much_of_the_image_the_figure_covers():
    image = np.full((1, 10, 10, 1), 220, dtype=np.uint8)
    image[0, :3] = 30  # 30% of pixels

    assert ink_fraction(image)[0] == pytest.approx(0.3)


def test_ink_fraction_counts_the_dark_figure_of_a_rendered_image():
    config = default_config()
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_params(rng, config, "circle")
        circle = shade(render_coverage(params, config), params, config.noise_sigma, rng)
        ring_params = replace(params, class_name="ring")
        ring = shade(render_coverage(ring_params, config), ring_params, config.noise_sigma, rng)

        circle_ink, ring_ink = ink_fraction(np.stack([circle, ring]))

        assert 0.1 < circle_ink < 0.45
        assert ring_ink < circle_ink
