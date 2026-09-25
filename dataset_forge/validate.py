"""Checking an export before it reaches students.

    python -m dataset_forge.validate DIR

Reads only the export folder, through picoface's own `load_dataset()`, so it
checks exactly what students will receive. Gated checks (any failure fails
the export): both bundles load with the manifest's shape and classes, each is
class-balanced, the two are disjoint, and mean brightness alone does not
tell the classes apart. Reported but not gated: how well ink fraction alone
does — real shapes differ in area, and shape-aware models should beat it.
"""

import argparse
import hashlib
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset_forge.export import SPLITS, read_manifest, write_manifest
from picoface.datasets import Dataset, DatasetWarning, load_dataset

# A single-statistic classifier may beat chance by less than this; the same
# margin the stub dataset's brightness test uses (tests/test_stub_data.py).
BRIGHTNESS_MARGIN = 0.1


@dataclass
class ValidationReport:
    """Each gated check's outcome and message, plus the two baselines."""

    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)
    chance: float | None = None
    mean_brightness_accuracy: float | None = None
    ink_fraction_accuracy: float | None = None

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for ok, _ in self.checks.values())

    @property
    def failed_checks(self) -> list[str]:
        return [name for name, (ok, _) in self.checks.items() if not ok]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": {
                name: {"passed": ok, "detail": msg} for name, (ok, msg) in self.checks.items()
            },
            "chance": self.chance,
            "mean_brightness_accuracy": self.mean_brightness_accuracy,
            "ink_fraction_accuracy": self.ink_fraction_accuracy,
        }

    def __str__(self) -> str:
        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        for name, (ok, msg) in self.checks.items():
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
        if self.chance is not None:
            lines.append(f"  chance:                       {self.chance:.3f}")
        if self.mean_brightness_accuracy is not None:
            lines.append(f"  mean-brightness-only (gated): {self.mean_brightness_accuracy:.3f}")
        if self.ink_fraction_accuracy is not None:
            lines.append(f"  ink-fraction-only (baseline): {self.ink_fraction_accuracy:.3f}")
        return "\n".join(lines)


def mean_brightness(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).mean(axis=1)


def ink_fraction(images: np.ndarray) -> np.ndarray:
    """Estimated fraction of each image's pixels that belong to the figure.

    Pixels below the midpoint between the image's 5th and 95th percentile
    values — roughly its foreground and background shades — count as ink,
    since figures are darker than their background.
    """
    flat = images.reshape(len(images), -1).astype(np.float64)
    low, high = np.percentile(flat, [5, 95], axis=1)
    return (flat < ((low + high) / 2)[:, None]).mean(axis=1)


def nearest_class_mean_accuracy(
    train_feature: np.ndarray, train: Dataset, test_feature: np.ndarray, test: Dataset
) -> float:
    """Accuracy on `test` of classifying a scalar feature by its nearest class mean in `train`."""
    classes = range(len(train.class_names))
    centroids = np.array([train_feature[train.labels == c].mean() for c in classes])
    predicted = np.abs(test_feature[:, None] - centroids).argmin(axis=1)
    return float(np.mean(predicted == test.labels))


def _check_loads(out_dir: Path, config: dict) -> tuple[dict[str, Dataset], str]:
    expected_shape = (config["height"], config["width"], config["channels"])
    expected_classes = list(config["class_names"])
    # The balance check reports a class with no images itself, with counts.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DatasetWarning)
        datasets = {split: load_dataset(out_dir / f"{split}.npz") for split in SPLITS}
    for split, data in datasets.items():
        shape = tuple(data.images.shape[1:])
        if shape != expected_shape:
            raise ValueError(f"{split} images are {shape}, expected {expected_shape}")
        if data.images.dtype != np.uint8:
            raise ValueError(f"{split} images are {data.images.dtype}, expected uint8")
        if data.class_names != expected_classes:
            raise ValueError(f"{split} classes are {data.class_names}, expected {expected_classes}")
    detail = f"both splits load as {expected_shape} uint8 with {len(expected_classes)} classes"
    return datasets, detail


def _check_balance(datasets: dict[str, Dataset], config: dict) -> tuple[bool, str]:
    k = len(config["class_names"])
    problems = []
    for split, data in datasets.items():
        expected = config[f"{split}_per_class"]
        counts = np.bincount(data.labels, minlength=k)
        if len(counts) != k or np.any(counts != expected):
            problems.append(
                f"{split} has per-class counts {counts.tolist()}, expected {expected} each"
            )
    if problems:
        return False, "; ".join(problems)
    return True, ", ".join(f"{split}: {config[f'{split}_per_class']} per class" for split in SPLITS)


def _check_disjoint(datasets: dict[str, Dataset]) -> tuple[bool, str]:
    def hashes(images: np.ndarray) -> set[bytes]:
        return {hashlib.sha256(image.tobytes()).digest() for image in images}

    shared = hashes(datasets["train"].images) & hashes(datasets["test"].images)
    if shared:
        return False, f"{len(shared)} image(s) appear in both train and test"
    return True, "no image appears in both train and test"


def validate_export(out_dir: str | Path) -> ValidationReport:
    """Run every check on the export in `out_dir`."""
    out_dir = Path(out_dir)
    report = ValidationReport()
    config = read_manifest(out_dir)["config"]

    try:
        datasets, detail = _check_loads(out_dir, config)
    except (OSError, KeyError, ValueError) as error:
        report.checks["loads"] = (False, str(error))
        return report
    report.checks["loads"] = (True, detail)
    report.checks["balanced"] = _check_balance(datasets, config)
    report.checks["disjoint"] = _check_disjoint(datasets)

    train, test = datasets["train"], datasets["test"]
    report.chance = 1 / len(train.class_names)
    report.mean_brightness_accuracy = nearest_class_mean_accuracy(
        mean_brightness(train.images), train, mean_brightness(test.images), test
    )
    report.ink_fraction_accuracy = nearest_class_mean_accuracy(
        ink_fraction(train.images), train, ink_fraction(test.images), test
    )
    limit = report.chance + BRIGHTNESS_MARGIN
    report.checks["mean_brightness"] = (
        report.mean_brightness_accuracy < limit,
        f"mean-brightness-only accuracy {report.mean_brightness_accuracy:.3f} "
        f"must be below chance + {BRIGHTNESS_MARGIN} = {limit:.3f}",
    )
    return report


def validate_and_record(out_dir: str | Path) -> ValidationReport:
    """Validate the export in `out_dir` and store the result in its manifest."""
    report = validate_export(out_dir)
    manifest = read_manifest(out_dir)
    manifest["validation"] = report.to_dict()
    write_manifest(out_dir, manifest)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Dataset Forge export.")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    report = validate_and_record(args.out_dir)
    print(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
