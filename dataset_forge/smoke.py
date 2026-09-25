"""Feasibility smoke check: does picoface, at its defaults, cope with an export?

    python -m dataset_forge.smoke DIR [--seed N] [--n N] [--epochs N] [--figures DIR]

Trains a CNN (`build_classifier`) and the student's VAE (`build_vae`) on the
export's training split with picoface's default settings, then reports:
- each model's wall-clock training time, held-out accuracy, and per-class
  confusion on the testing split;
- `classify_generated()` agreement: the CNN judging images the VAE generated
  for each class;
- reconstruction agreement: the CNN judging the VAE's reconstructions of real
  test images, which separates "the decoder can't draw this class" from "the
  capstone samples the wrong part of the latent space";
- edge darkening of `activation_maximize()` images for each class and model:
  the mean of the outermost 2-pixel ring minus the mean of the 2-pixel ring
  inside it. Its blur pads with zeros (black), which on light-background data
  could darken the edges; strongly negative values mean it does.

`--figures DIR` also writes `generated.png`, `reconstructed.png`, and
`activation_maximize.png` there (one row per class, real images first where
there are any), for looking at what the numbers are about.
`--epochs` overrides `train()`'s default, to separate the effect of more data
from that of more optimizer steps. A measurement for Phase 6, not a test and
not tuning: nothing here passes or fails.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from picoface._internals.model_api import _preprocess_images, _to_uint8_images
from picoface.classifier import build_classifier, evaluate, predict, train
from picoface.datasets import Dataset, load_dataset
from picoface.generator import build_vae
from picoface.linkage import activation_maximize, classify_generated

_FIGURE_SCALE = 6
# activation_maximize() images per class and model, each from its own random start.
_ASCENT_STARTS = 4
_EDGE_RING = 2


def confusion_matrix(model, data: Dataset) -> np.ndarray:
    """Counts of (true class, predicted class) over `data`, in `data`'s class order."""
    k = len(data.class_names)
    index = {name: i for i, name in enumerate(data.class_names)}
    matrix = np.zeros((k, k), dtype=np.int64)
    for image, label in zip(data.images, data.labels):
        matrix[label, index[predict(model, image)]] += 1
    return matrix


def reconstruct(vae, images: np.ndarray) -> np.ndarray:
    """The VAE's reconstruction of each image, decoded from its latent mean.

    Uses the `latent_access` capability behind `classify_generated()`, which
    has no public verb: the Forge is instructor tooling, so it may reach it.
    """
    with torch.no_grad():
        return _to_uint8_images(vae.decode(vae.encode_mu(_preprocess_images(images, vae))))


def reconstruction_report(cnn, vae, data: Dataset, n: int) -> dict:
    """Per class, the first `n` images, their reconstructions, and CNN agreement on them."""
    report = {}
    for label, name in enumerate(data.class_names):
        real = data.images[data.labels == label][:n]
        recon = reconstruct(vae, real)
        agreement = float(np.mean([predict(cnn, image) == name for image in recon]))
        report[name] = {"real": real, "reconstructed": recon, "agreement": agreement}
    return report


def edge_darkening(images: np.ndarray) -> float:
    """Mean of the images' outermost `_EDGE_RING`-pixel ring minus that of the ring inside it."""
    height, width = images.shape[1:3]
    rows, cols = np.mgrid[0:height, 0:width]
    depth = np.minimum.reduce([rows, cols, height - 1 - rows, width - 1 - cols])
    outer = images[:, depth < _EDGE_RING].astype(np.float64)
    inner = images[:, (depth >= _EDGE_RING) & (depth < 2 * _EDGE_RING)].astype(np.float64)
    return float(outer.mean() - inner.mean())


def activation_maximize_report(models: dict, class_names: list[str]) -> dict:
    """Per class and model, `_ASCENT_STARTS` activation_maximize() images and their edge darkening."""
    report = {}
    for name in class_names:
        report[name] = {}
        for model_name, r in models.items():
            images = np.stack(
                [activation_maximize(r["model"], name) for _ in range(_ASCENT_STARTS)]
            )
            report[name][model_name] = {
                "images": images,
                "edge_darkening": edge_darkening(images),
            }
    return report


def smoke(out_dir: str | Path, seed: int = 0, n: int = 20, epochs: int | None = None) -> dict:
    """Train both models on `out_dir`'s export and measure them."""
    out_dir = Path(out_dir)
    train_data = load_dataset(out_dir / "train.npz")
    test_data = load_dataset(out_dir / "test.npz")
    train_kwargs = {} if epochs is None else {"epochs": epochs}
    torch.manual_seed(seed)

    results = {"class_names": train_data.class_names, "train_data": train_data, "models": {}}
    for name, build in (("cnn", build_classifier), ("vae", build_vae)):
        model = build(train_data)
        start = time.perf_counter()
        train(model, train_data, **train_kwargs)
        results["models"][name] = {
            "model": model,
            "train_seconds": time.perf_counter() - start,
            "test_accuracy": evaluate(model, test_data),
            "confusion": confusion_matrix(model, test_data),
        }

    cnn, vae = results["models"]["cnn"]["model"], results["models"]["vae"]["model"]
    results["classify_generated"] = classify_generated(cnn, vae, train_data, n=n)
    results["reconstruction"] = reconstruction_report(cnn, vae, test_data, n=n)
    results["activation_maximize"] = activation_maximize_report(
        results["models"], train_data.class_names
    )
    return results


def format_results(results: dict) -> str:
    """The results as Markdown tables, ready to paste into a diagnostics record."""
    names = results["class_names"]
    lines = ["| Model | Train time (s) | Held-out accuracy |", "|---|---|---|"]
    for model_name, r in results["models"].items():
        lines.append(f"| {model_name} | {r['train_seconds']:.1f} | {r['test_accuracy']:.3f} |")

    for model_name, r in results["models"].items():
        lines += ["", f"{model_name} confusion (rows: true class, columns: predicted):", ""]
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for name, row in zip(names, r["confusion"]):
            lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")

    generated, recon = results["classify_generated"], results["reconstruction"]
    lines += [
        "",
        "CNN agreement on the VAE's images (generated for the class; reconstructed from "
        "real test images):",
        "",
        "| Class | classify_generated() | Reconstruction |",
        "|---|---|---|",
    ]
    for name in names:
        lines.append(
            f"| {name} | {generated.per_class[name]:.2f} | {recon[name]['agreement']:.2f} |"
        )
    overall_recon = np.mean([r["agreement"] for r in recon.values()])
    lines.append(f"| overall | {generated.overall:.2f} | {overall_recon:.2f} |")

    ascent = results["activation_maximize"]
    model_names = list(results["models"])
    lines += [
        "",
        f"activation_maximize() edge darkening ({_ASCENT_STARTS} random starts; outermost "
        f"{_EDGE_RING}-pixel ring minus the next ring in; negative means darker edges):",
        "",
        "| Class | " + " | ".join(model_names) + " |",
        "|---" * (len(model_names) + 1) + "|",
    ]
    for name in names:
        values = " | ".join(f"{ascent[name][m]['edge_darkening']:+.1f}" for m in model_names)
        lines.append(f"| {name} | {values} |")
    return "\n".join(lines)


def _grid(rows: list[list[np.ndarray]]) -> Image.Image:
    """One row per list; `None` entries leave a gap. Images are (H, W, 1) uint8."""
    height, width = next(im for row in rows for im in row if im is not None).shape[:2]
    cell_h, cell_w = height * _FIGURE_SCALE, width * _FIGURE_SCALE
    sheet = Image.new("L", (max(map(len, rows)) * cell_w, len(rows) * cell_h), 255)
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            if image is not None:
                tile = Image.fromarray(image[..., 0], mode="L").resize(
                    (cell_w, cell_h), Image.Resampling.NEAREST
                )
                sheet.paste(tile, (c * cell_w, r * cell_h))
    return sheet


def write_figures(results: dict, figures_dir: str | Path) -> list[Path]:
    """`generated.png`: per class, 4 real training images, a gap, 12 generated ones.
    `reconstructed.png`: per class, 6 real test images, a gap, their reconstructions.
    `activation_maximize.png`: per class, the CNN's activation_maximize() images,
    a gap, the VAE's.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data, report = results["train_data"], results["classify_generated"]
    generated_rows, reconstructed_rows = [], []
    for label, name in enumerate(results["class_names"]):
        real = list(data.images[data.labels == label][:4])
        made = [im for im, intended in zip(report.images, report.intended) if intended == name]
        generated_rows.append(real + [None] + made[:12])
        recon = results["reconstruction"][name]
        reconstructed_rows.append(
            list(recon["real"][:6]) + [None] + list(recon["reconstructed"][:6])
        )

    ascent_rows = []
    for name in results["class_names"]:
        row = []
        for model_name in results["models"]:
            row += list(results["activation_maximize"][name][model_name]["images"]) + [None]
        ascent_rows.append(row[:-1])

    paths = [
        figures_dir / "generated.png",
        figures_dir / "reconstructed.png",
        figures_dir / "activation_maximize.png",
    ]
    _grid(generated_rows).save(paths[0])
    _grid(reconstructed_rows).save(paths[1])
    _grid(ascent_rows).save(paths[2])
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Smoke-check picoface's defaults on an export.")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="torch seed for model training")
    parser.add_argument("--n", type=int, default=20, help="images per class for agreement")
    parser.add_argument("--epochs", type=int, default=None, help="override train()'s default")
    parser.add_argument("--figures", type=Path, default=None, help="folder to write figures to")
    args = parser.parse_args(argv)

    results = smoke(args.out_dir, seed=args.seed, n=args.n, epochs=args.epochs)
    print(format_results(results))
    if args.figures is not None:
        for path in write_figures(results, args.figures):
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
