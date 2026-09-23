"""Rendering an export's two splits and writing them in the data-contract format.

An export folder holds `train.npz`, `test.npz`, the `classes.json` both share
(picoface's `load_dataset()` reads it from the `.npz` file's folder), and
`manifest.json`, which records everything needed to render it again.
"""

import json
import platform
import subprocess
import zlib
from pathlib import Path

import numpy as np
import PIL

from dataset_forge.config import ForgeConfig
from dataset_forge.render import render_image
from dataset_forge.shapes import check_class_names

PACKAGE_DIR = Path(__file__).parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "default.json"
# Gitignored: exports are regenerated from a config and seed, never committed.
OUTPUT_DIR = PACKAGE_DIR / "output"

SPLITS = ("train", "test")
MANIFEST = "manifest.json"


def default_out_dir(config: ForgeConfig, seed: int) -> Path:
    return OUTPUT_DIR / f"{config.name}-seed{seed}"


def _class_rng(seed: int, split: str, class_name: str) -> np.random.Generator:
    """An independent random stream for one class in one split.

    Keyed by split and class name, so the test split doesn't change when the
    training count does, and existing classes don't change when one is added.
    """
    key = (SPLITS.index(split), zlib.crc32(class_name.encode()))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def render_split(config: ForgeConfig, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    """All of one split's images and labels, class by class in config order."""
    per_class = config.train_per_class if split == "train" else config.test_per_class
    n = per_class * len(config.class_names)
    images = np.empty((n, config.height, config.width, config.channels), dtype=np.uint8)
    labels = np.empty(n, dtype=np.int64)

    for label, class_name in enumerate(config.class_names):
        rng = _class_rng(seed, split, class_name)
        start = label * per_class
        for i in range(start, start + per_class):
            images[i] = render_image(rng, config, class_name)
        labels[start : start + per_class] = label

    return images, labels


def _git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PACKAGE_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--", str(PACKAGE_DIR))),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def build_manifest(config: ForgeConfig, seed: int) -> dict:
    return {
        "config": config.to_dict(),
        "seed": seed,
        "git": _git_state(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
    }


def read_manifest(out_dir: str | Path) -> dict:
    with open(Path(out_dir) / MANIFEST) as f:
        return json.load(f)


def write_manifest(out_dir: str | Path, manifest: dict) -> None:
    with open(Path(out_dir) / MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def export(config: ForgeConfig, seed: int = 0, out_dir: str | Path | None = None) -> Path:
    """Render both splits and write them to `out_dir`; return the folder.

    Everything is rendered before anything is written, so a failure (such as
    an unknown class name) leaves no partial export behind.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    check_class_names(config.class_names)
    out_dir = Path(out_dir) if out_dir is not None else default_out_dir(config, seed)

    splits = {split: render_split(config, seed, split) for split in SPLITS}

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, (images, labels) in splits.items():
        np.savez_compressed(out_dir / f"{split}.npz", images=images, labels=labels)
    with open(out_dir / "classes.json", "w") as f:
        json.dump({str(i): name for i, name in enumerate(config.class_names)}, f, indent=2)
        f.write("\n")
    write_manifest(out_dir, build_manifest(config, seed))
    return out_dir
