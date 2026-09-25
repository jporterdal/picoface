# Dataset Forge

The instructor-only tool that renders `picoface`'s course dataset: one
grayscale shape per image, with randomized rotation, size, position, shades,
and noise, exported as class-balanced train and test splits in picoface's
data-contract format (`.npz` + `classes.json`, readable with
`picoface.datasets.load_dataset()`).

It lives outside the installable `picoface` package, so its dependencies
never reach a student's `pip install picoface`. The Forge imports picoface
(to validate exports and for the smoke check); picoface never imports the
Forge.

## Install

From the repo root, install picoface first (see the top-level README), then
the Forge's pinned requirements:

```bash
pip install -e .
pip install -r dataset_forge/requirements.txt
```

Pillow and numpy are pinned exactly: an export is only byte-for-byte
reproducible with the same versions.

## Make a dataset

Run from the repo root:

```bash
python -m dataset_forge                      # configs/default.json, seed 0
python -m dataset_forge --seed 1             # another draw of the same design
python -m dataset_forge --config my.json --out some/folder
```

The default output folder is `dataset_forge/output/<config name>-seed<N>/`:

| File | What it is |
|---|---|
| `train.npz`, `test.npz` | The two splits: `images` (uint8, N×H×W×1) and `labels` |
| `classes.json` | Label index → class name, shared by both splits |
| `manifest.json` | The full config, seed, git commit, tool versions, and validation results |

Students load a split with `load_dataset("…/train.npz")`.

**Exports are never committed.** `dataset_forge/output/` is gitignored. To
recreate an export exactly, reuse the config and seed in its manifest, with
the pinned requirements installed.

## Validation

Every export is checked right after it is written, and the command exits
non-zero if a check fails. The folder is kept so you can inspect it, and the
failure is recorded in its manifest. To re-check an export later:

```bash
python -m dataset_forge.validate dataset_forge/output/default-seed0
```

Gated checks:
- both splits load with `load_dataset()` with the configured shape and classes;
- each split has exactly the configured number of images per class;
- no image appears in both splits;
- a classifier using only each image's mean brightness scores below chance + 0.1.

Also reported, but not gated: a classifier using only each image's ink
fraction (how much of it the figure covers). Real shapes differ in area, so
this scores well above chance. A model that has learned shapes should beat it
by a wide margin.

## Look at the shapes

```bash
python -m dataset_forge.contact_sheet        # writes dataset_forge/output/contact_sheet.png
```

One row per class, twelve fresh renders each, scaled up so pixels stay
visible. Check it after adding a class or changing size, stroke, or
resolution settings.

## Smoke-check picoface on an export

```bash
python -m dataset_forge.smoke dataset_forge/output/default-seed0
python -m dataset_forge.smoke dataset_forge/output/default-seed0 \
    --figures dataset_forge/output/figures/seed0    # also write the figures
```

Trains a CNN and a VAE with picoface's default settings, and prints:
- each model's training time and held-out accuracy;
- each model's per-class confusion;
- `classify_generated()` agreement: how often the CNN agrees with the class
  the VAE generated an image for;
- reconstruction agreement: how often the CNN recognizes the VAE's
  reconstructions of real test images. If this is low too, the decoder can't
  draw the class at all, whatever the capstone samples.

`--figures DIR` writes `generated.png` and `reconstructed.png` (one row per
class, real images first). `--epochs N` overrides `train()`'s default. Run it
on an otherwise idle machine: training times are only meaningful then.

It is a measurement, not a test: nothing passes or fails.

## Add a class

1. In `shapes.py`, write a draw function that paints the figure at 255 on the
   0 canvas it is given. It must stay inside the circle of
   `placement.radius` around `(placement.cx, placement.cy)`, rotated by
   `placement.angle` (`placement.point(u, v)` does the rotation). Use `CUT` (0)
   to cut holes. Shades, noise, placement, and anti-aliasing are handled for
   you.
2. Register it in `SHAPES` under its class name.
3. Add the name to a config's `class_names`. Order sets the label numbers.

Adding a class doesn't change the images of the existing classes for the
same seed.

## Settings

`configs/default.json` holds every setting; `config.py` documents each one.
The main ones:

- `class_names`: the classes, in label order.
- `height`, `width`: image size (28×28 by default).
- `train_per_class`, `test_per_class`: images per class in each split.
- `radius_fraction`, `radius_jitter`: figure size, and how much it varies.
- `stroke_fraction`: outline and face-feature width, relative to the figure.
- `background_range`, `min_contrast`: random shades. The foreground is
  always darker than the background, by at least `min_contrast`, so figures
  look like dark drawings on light paper, as mediaComp draws by default.
  Every class shares this polarity, `negative_smiley` included.
- `noise_sigma`: pixel noise.

## Tests

```bash
pytest dataset_forge/tests
```

A plain `pytest` from the repo root also runs them. Without the Forge's
requirements installed, they are skipped rather than failing.
