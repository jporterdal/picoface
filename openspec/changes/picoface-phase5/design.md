## Context

`dataset_forge/` holds only a placeholder README and an empty `requirements.txt` from Phase 0. `dataset_forge/output/` is already gitignored. See proposal.md for the motivation and scope, and `specs/dataset-forge/spec.md` for the behavior contract.

Three existing pieces shape the approach:
- `load_dataset(path)` (`src/picoface/datasets.py`) reads `classes.json` from the `.npz` file's own folder. A `train.npz` and a `test.npz` in one folder therefore share one `classes.json`, with no `data-contract` change.
- The stub's `kind="shapes"` mode (`src/picoface/_internals/stub_data.py`) established the lesson this phase inherits: classes separable by a global statistic teach nothing about shapes. Its test (`tests/test_stub_data.py`) gates a nearest-class-mean classifier on mean brightness at below chance + 0.1. The stub achieved that by giving every figure exactly the same pixel count. Real, varied, anti-aliased shapes can't do that without distorting the taxonomy, so this phase uses a different mechanism (Decision 4).
- Pillow is already in the development environment (12.3.0, pulled in by matplotlib), but `picoface` does not declare it.

## Goals / Non-Goals

**Goals:**
- One command produces a validated, reproducible export from a checked-in default config.
- Geometry, shading, and noise are separate stages, so adding a class touches only geometry.
- The default export passes the mean-brightness gate by a comfortable margin, not by luck of a seed.

**Non-Goals:**
- Choosing how students receive the dataset (bundled, downloaded, or shipped with the notebooks). That is Phase 7's distribution decision. This phase only guarantees the dataset can be regenerated exactly.
- Tuning models on the export, or changing `picoface` defaults. The smoke check (Decision 7) measures; Phase 6 acts on what it finds.
- Augmentation at training time. All variation is baked into the rendered images; `train()` is unchanged.
- Occlusion, clutter, multiple figures per image, or backgrounds with texture or gradients. Each image holds one figure on a flat, noisy background.
- Color. The config carries a channel count so color can be added later, but only `channels = 1` is implemented and validated in this phase.

## Decisions

### 1. Package layout and entry point

`dataset_forge/` becomes an importable Python package run from the repo root:

```
dataset_forge/
  __init__.py
  __main__.py       python -m dataset_forge [--config PATH] [--seed N] [--out DIR]
  config.py         ForgeConfig (frozen dataclass), load/save as JSON
  shapes.py         the class registry: name -> draw function
  render.py         variation sampling, supersampled mask, shading, noise
  export.py         split generation, .npz/classes.json/manifest writing
  validate.py       post-export checks and baseline report
  smoke.py          one-off feasibility check (Decision 7)
  configs/default.json
  tests/
  requirements.txt  pinned Pillow and numpy
  README.md
```

The default output folder is `dataset_forge/output/<config name>-seed<N>/` (already gitignored), holding `train.npz`, `test.npz`, `classes.json`, and `manifest.json`. `.npz` files are written with `np.savez_compressed`; `load_dataset()` reads compressed and uncompressed files alike.

**Alternatives considered:**
- **A standalone script with no package structure** — rejected: tests need to import the renderer and validator separately, and a flat script grows into an unstructured file quickly.
- **Install the Forge as its own distribution (a second `pyproject.toml`)** — rejected for now: adds packaging surface for a tool only the instructor runs from a checkout. Running with `python -m` from the repo root is enough, and nothing blocks packaging it later.

### 2. Geometry is a coverage mask; shading happens afterward

Each class's draw function paints only a geometric mask: figure pixels at 255 on a 0 canvas, on a supersampled canvas (4× the target resolution per side), using Pillow's `ImageDraw` in mode `"L"`. A draw function receives the figure's centre, circumscribed radius, rotation angle, and outline stroke width, all in supersampled pixel units, and draws within that circumscribed circle. Cut-outs (the `negative_smiley`'s features) are drawn with fill 0 on top of the filled disc.

The mask is downsampled to the target resolution with a box filter, so each output pixel's value is exactly the fraction of its area the figure covers. This is the anti-aliasing. Shading then maps that coverage to gray: `image = background + (foreground − background) × coverage`, plus noise, clipped and cast to `uint8`.

Keeping draw functions shade-free is what makes the taxonomy extensible (spec: Extensible class taxonomy). A new class never deals with shading, noise, variation ranges, or anti-aliasing. It also gives validation an exact ink fraction for free during testing, although the gate itself only ever reads exported pixels (Decision 5).

Rotation is applied to vertex coordinates (polygons, star, eye positions, mouth-arc centre), not by rotating a raster, so edges stay sharp before downsampling. Circles and rings are rotation-invariant and ignore the angle. The smiley's mouth is drawn with `ImageDraw.arc` with its start and end angles offset by the rotation, around a mouth centre that is itself rotated about the face centre.

**Alternatives considered:**
- **Draw directly in gray shades per image** — rejected: every draw function would have to handle shading and cut-outs in background color, which is exactly what a new class should not have to know.
- **Extend the stub's pure-numpy, score-ranked rasterizer** — rejected: exact for equal area, but awkward for stars, arcs, and faces, and it doesn't anti-alias. Pillow was agreed during exploration.
- **Rasterize upright and rotate the image** — rejected: rotating a raster resamples it and softens the edges twice.

### 3. The initial taxonomy's geometry

Every figure is sized by its circumscribed radius `R` (the radius of the smallest circle around it, centred on its centre). This is what "same size" means across classes, and it is what the in-frame guarantee is computed from.

| Class | Geometry (relative to `R`) |
|---|---|
| `square` | filled square, vertices at `R` |
| `triangle` | filled equilateral triangle, vertices at `R` |
| `star` | filled five-pointed star, outer vertices at `R`, inner vertices at `0.382 R` (a regular pentagram's ratio) |
| `circle` | filled disc of radius `R` |
| `ring` | circle outline, outer radius `R`, stroke `s` |
| `smiley` | `ring` plus two filled eyes and a mouth arc of stroke `s` |
| `negative_smiley` | `circle` with two eyes and a mouth arc of stroke `s` cut out |

The two smileys share one face-feature layout. Eyes are discs of radius `0.14 R` at `(±0.35 R, −0.25 R)` from the centre. The mouth is an arc of radius `0.5 R` centred at `(0, 0)`, spanning 30°–150° below the centre. This layout is provisional: task 2.4 checks legibility at 24×24 by eye and adjusts it before the default config is final. The outline stroke `s` is a configured fraction of `R` (default `0.18 R`, with slight jitter), with a floor of about one output pixel so outlines never vanish at the smallest sizes.

Any rotation of a square is a square, not a diamond, because there is no diamond class. That is what "defined without regard to rotation" buys. The two smileys are the only classes whose images depend on rotation, which is intended: the model has to recognize a face at any angle.

### 4. Randomized shading, not area matching, defeats mean brightness

Each image draws its background shade `b ~ U(b_min, b_max)` and its foreground shade `f ~ U(b + c_min, 255)`, with defaults `b ∈ [0, 159]` and `c_min = 96`. The image's mean brightness is `b + (f − b) × ink_fraction`, and the per-image shade spread swamps the ink-fraction difference between classes.

A numeric sketch during exploration, using ideal figure areas, seven classes, ±10–20% radius jitter, and these shade ranges, put the best possible mean-brightness-only accuracy at 0.22–0.23 against 0.14 chance. The gate (chance + 0.1 = 0.243) uses a nearest-class-mean classifier, which does no better than that. The sketch leaves little margin, so task 3.4 measures on real exports and widens the shade ranges, or narrows the size jitter, if the default config's margin is thin.

Polarity is fixed, with the foreground always lighter. Otherwise "painted on the background" (`smiley`) and "cut out of the figure" (`negative_smiley`) could not be told apart without knowing which shade is the background.

**Alternatives considered:**
- **Area-matched sizing** (choose each figure's size so all classes render the same ink amount) — proposed during exploration, then set aside: it forces rings and smileys to be much larger than filled circles, which makes size a class cue instead. With randomized shading it is not needed for mean brightness. Kept as a fallback if the gate proves fragile.
- **Randomize polarity too** — rejected: breaks the smiley and negative-smiley distinction (above).

### 5. Validation reads only the exported files

`validate.py` runs against the export folder after writing, by loading it with `picoface.load_dataset()`. It never sees renderer internals, so it validates exactly what students will receive. It is also a standalone command (`python -m dataset_forge.validate DIR`) for re-checking an export later. The export command fails with a non-zero exit status if any gated check fails. The export folder is kept for inspection, and the failure is recorded in the manifest.

- **Mean-brightness gate:** a nearest-class-mean classifier on per-image mean brightness, fit on train, scored on test, must be below `1/k + 0.1`. This is the stub test's classifier and threshold, so the bar is the same one the plumbing was proven against.
- **Ink-fraction baseline (reported, not gated):** each image's ink fraction is estimated from its pixels: the fraction above the midpoint between its 5th and 95th percentile values (roughly the background and foreground shades). A nearest-class-mean classifier on that is fit and scored the same way. Recorded in the manifest and the validation report.
- **Disjointness:** hash each image's bytes, and check no training hash appears in the testing set. With continuous random parameters a collision is essentially impossible, so this check is a guard against a seeding bug (e.g. both splits drawn from the same stream).

**Alternatives considered:**
- **Gate on ink fraction too** — rejected during exploration: real shapes differ in area, and removing that would take either size ranges too wide for 24×24 or area matching (Decision 4). A shape-aware model beating this baseline by a wide margin is the evidence that it learned shapes.

### 6. Reproducibility: one seed, independent streams, pinned renderer

One integer seed seeds a `numpy.random.SeedSequence`, which is split into independent child streams for the training split and the testing split. This makes the two splits independent even though they come from one seed, and the testing split doesn't change if the training count changes. All per-image parameters (rotation, radius, position, stroke, shades, noise) come from numpy. Pillow only rasterizes, which is deterministic for a given version.

`requirements.txt` pins exact Pillow and numpy versions. The manifest records the full resolved config, the seed, the Forge's git commit (if available), and the Python, numpy, and Pillow versions. Byte-identical output is only promised within the same pinned environment (spec: Reproducible exports).

### 7. Feasibility smoke check as a script, recorded in diagnostics.md

`python -m dataset_forge.smoke DIR` loads an export's two bundles and, with `picoface` at default settings:
- trains a CNN (`build_classifier`) and a VAE (`build_vae`) on the training bundle;
- reports each model's wall-clock training time and held-out accuracy on the testing bundle;
- reports the CNN's per-class confusion (the smiley/circle pair is the one to watch);
- reports `classify_generated()` agreement.

It is a script, not a test: it takes tens of seconds to minutes, and its output is a measurement, not a pass/fail. Its results for the default export, on this development machine and noted as such, go in `diagnostics.md` in this change. That follows Phase 3c's diagnostics record, and it feeds Phase 6 and the 24→28 decision. It is the only part of the Forge that imports `picoface`'s training code.

### 8. Test isolation

Forge tests live in `dataset_forge/tests/`. Its `conftest.py` calls `pytest.importorskip("PIL")`, so a plain `pytest` from the repo root collects and runs them when the Forge's requirements are installed, and skips them cleanly otherwise (spec: The project's test suite runs without the Forge's dependencies). Pillow is not added to `picoface`'s `dev` extras, which would blur the separation `packaging` requires. Tests use small configs (few images per class) so the suite stays fast. The seven-class default export is exercised once, at a reduced count, for the validation gate.

## Risks / Trade-offs

- **The mean-brightness gate's margin may be thin** (Decision 4's sketch put the best possible accuracy at 0.22–0.23 against a 0.243 gate) → Measured on real exports in task 3.4. Shade ranges and size jitter are config values and are adjusted until the default export clears the gate with margin. Area matching remains the fallback.
- **Smiley features may not be legible at 24×24** (eyes about 1–2 px, the mouth one stroke wide) → Task 2.4 inspects rendered samples by eye before the config is finalized. The smoke check's per-class confusion shows whether the CNN separates `smiley`/`ring` and `negative_smiley`/`circle`. 28×28 is the planned relief if not.
- **VAE blur will hit the face classes hardest** in `classify_generated()`: blurring away a smiley's features leaves a ring or a circle → Expected, and recorded by the smoke check as a Phase 6 input rather than fixed here. It does not affect the Forge's contract.
- **Ink fraction lets a model partly shortcut the task** → Accepted (Decision 5). The baseline is reported so Phase 6 can check that model accuracy sits well above it.
- **Reproducibility depends on the pinned Pillow version**; a different Pillow could rasterize edges differently → The manifest records versions. Exact reproduction requires the pinned environment, and the spec promises no more.
- **The dev environment gets Pillow transitively via matplotlib**, so a Forge import could appear to work in an environment where the Forge's requirements were never installed → Harmless for running the Forge, but the version may differ from the pin. The manifest records the Pillow version actually used, and the README says to install `dataset_forge/requirements.txt`.
- **1,000 training images per class at the default batch size is about 440 optimizer steps per epoch**, far more than the stub → Measured by the smoke check against the time budget. Phase 6 tunes `train()` defaults if needed. Image counts are config values.

## Migration Plan

Additive: nothing in `src/picoface/` or any existing spec changes, and no existing test is touched. To roll back, delete the new `dataset_forge/` contents.

## Open Questions

- Final image counts per class. The default starts at 1,000 train / 200 test, and the smoke check may argue for fewer (time budget) or more (accuracy). This is a config value and changes nothing else.
