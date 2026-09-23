## 1. Scaffolding and config

- [ ] 1.1 Create the `dataset_forge` package skeleton (`__init__.py`, `__main__.py`, `config.py`, `shapes.py`, `render.py`, `export.py`, `validate.py`, `smoke.py`, `configs/`, `tests/`) per design.md Decision 1. Pin exact Pillow and numpy versions in `dataset_forge/requirements.txt`. Verify: `python -c "import dataset_forge"` succeeds from the repo root, and `pip install -r dataset_forge/requirements.txt` resolves.
- [ ] 1.2 Add `dataset_forge/tests/conftest.py` with `pytest.importorskip("PIL")` (Decision 8). Verify: `pytest` from the repo root collects the Forge tests. Separately, running the Forge tests in an environment without Pillow (or with the import forced to fail via a monkeypatched `sys.modules["PIL"] = None` in a subprocess) reports them as skipped, not errored.
- [ ] 1.3 Implement `ForgeConfig` as a frozen dataclass: class list, height, width, channels, train/test counts per class, nominal radius fraction, radius jitter, stroke fraction and jitter, shade ranges, minimum contrast, noise sigma, and supersampling factor. Add JSON load/save, and reject non-positive counts, an empty class list, `channels != 1`, and shade ranges that cannot satisfy the minimum contrast. Verify: unit tests round-trip a config through JSON unchanged and cover each rejection with a clear message.

## 2. Rendering

- [ ] 2.1 Implement the class registry in `shapes.py` (name → draw function on a supersampled `"L"` mask; signature per Decision 2) with the seven classes from Decision 3, rotating vertex coordinates rather than rasters. Unknown names raise an error naming the class and the available ones. Verify: tests cover the registry lookup and unknown-name error, and check that each class draws non-empty ink inside its circumscribed circle at several rotations.
- [ ] 2.2 Implement per-image variation sampling in `render.py`: full-circle rotation, radius jitter around the nominal size, and position offsets constrained so the circumscribed circle (plus anti-aliasing margin) stays in frame. Also draw background and foreground shades with fixed polarity and minimum contrast, and stroke jitter with a floor of about one output pixel. All randomness comes from a passed-in numpy `Generator`. Verify: property-style tests over many draws check that figures are never clipped (no ink in the outermost row and column) and that `foreground − background ≥` the minimum contrast for every image.
- [ ] 2.3 Implement the mask → image pipeline: supersampled mask, box-filter downsample to coverage, shading `b + (f − b) × coverage`, Gaussian noise, clip, `uint8`, returning `H×W×1`. Verify: tests check shape and dtype, that a coverage-1 pixel with zero noise equals the foreground shade, that repeated renders of one class are not pixel-identical, and that `circle` vs `negative_smiley` and `ring` vs `smiley` at identical parameters differ only inside the circumscribed circle (spec scenario).
- [ ] 2.4 Render a contact sheet of every class at 24×24 (e.g. `dataset_forge/output/contact_sheet.png`, gitignored) and inspect it by eye. Adjust the face-feature layout and stroke defaults from Decision 3 until eyes and mouth are legible and every class reads as itself. Verify: the contact sheet is reviewed by the user, and any layout changes are reflected in design.md Decision 3.

## 3. Export and validation

- [ ] 3.1 Implement `export.py`: split the seed with `SeedSequence` into independent training and testing streams (Decision 6), render the configured counts per class in config order, and write `train.npz`, `test.npz` (compressed), `classes.json`, and `manifest.json` (resolved config, seed, git commit if available, Python/numpy/Pillow versions) to `dataset_forge/output/<config name>-seed<N>/` by default. Verify: tests with a small config check the folder's contents, that both `.npz` files load via `picoface.load_dataset()` with the configured shape and class names, and that labels follow config order, including a subset-of-classes config.
- [ ] 3.2 Add reproducibility tests. Verify: the same config and seed export identical images, labels, and class names; a different seed exports different images; changing only the training count leaves the testing bundle unchanged; the manifest round-trips to the config that produced it; and `git check-ignore dataset_forge/output/x` reports the default output location as ignored.
- [ ] 3.3 Implement `validate.py` (Decision 5), reading only the export folder. Checks:
  - the `load_dataset()` round trip, with shape and classes matching the manifest;
  - per-class balance in each split;
  - disjointness by image hash;
  - the mean-brightness gate (nearest class mean, fit on train, scored on test, `< 1/k + 0.1`).

  It also reports the ink-fraction baseline. It prints a report, writes the results into the manifest, and exits non-zero naming any failed check. Wire it into `python -m dataset_forge` after export, and expose it as `python -m dataset_forge.validate DIR`. Verify: tests check that a small valid export passes and reports both baselines; that a deliberately brightness-separable config (fixed shades with `circle` vs `ring`) fails, naming the mean-brightness check; that a hand-corrupted export (one duplicated image across splits, one class dropped from a split) fails, naming the right check; and that a high ink-fraction baseline alone does not fail.
- [ ] 3.4 Write `configs/default.json` with the seven classes, 24×24×1, 1,000 train / 200 test per class, and the Decision 3/4 defaults. Export it with seed 0 and run validation. Measure the mean-brightness gate's margin across at least 3 seeds. If any seed comes within 0.03 of the gate, widen the shade range or narrow the size jitter and re-measure. Verify: the default config passes validation on every seed tried, and the measured mean-brightness and ink-fraction accuracies (per seed) are recorded in design.md Decision 4.

## 4. Command line and docs

- [ ] 4.1 Implement `python -m dataset_forge [--config PATH] [--seed N] [--out DIR]` in `__main__.py`, defaulting to `configs/default.json` and seed 0, printing where the export went and the validation report. Unknown class names in the config fail before any rendering. Verify: a test runs the command in a subprocess with a small config into a temporary folder and checks the exit status and the output files. A second run with an unknown class exits non-zero with the class named and no files written.
- [ ] 4.2 Rewrite `dataset_forge/README.md`: what the Forge is for, installing its requirements (after `pip install -e .`), running an export and re-validating one, what the manifest is, adding a class (a registry entry plus a config line), and that exports are never committed. Verify: following the README from a fresh checkout reproduces the default export.

## 5. Feasibility smoke check

- [ ] 5.1 Implement `python -m dataset_forge.smoke DIR` (Decision 7). It trains `build_classifier` and `build_vae` with `picoface` defaults on the training bundle, and reports:
  - each model's wall-clock training time and held-out accuracy on the testing bundle;
  - the CNN's per-class confusion matrix;
  - `classify_generated()` per-class and overall agreement.

  Verify: runs to completion on a small export in a test marked slow, or skipped by default, and on the default export by hand.
- [ ] 5.2 Run the smoke check on the default export (seed 0, and 2 more seeds if time allows), and record the results in `openspec/changes/picoface-phase5/diagnostics.md` alongside the machine it ran on and the export's validation baselines. Include notes on smiley/circle and smiley/ring confusion, and a recommendation on staying at 24×24 or moving to 28×28. Verify: diagnostics.md exists with the tables filled in and is reviewed by the user.

## 6. Integration

- [ ] 6.1 Confirm the separation requirements: `picoface` imports nothing from `dataset_forge` (grep plus a test that imports every `picoface` module and asserts no `dataset_forge` module in `sys.modules`), and `pyproject.toml`'s dependencies are unchanged. Verify: the test passes and `git diff pyproject.toml` shows no dependency change.
- [ ] 6.2 Run the full test suite from the repo root with the Forge's requirements installed. Verify: `pytest` passes with no failures, and the existing `tests/` results are unchanged.
