## 1. Dataset Forge: flip polarity

- [ ] 1.1 Flip shade sampling in `render.py:sample_params()`: draw `background ~ U(low, high)`, then `foreground ~ U(0, background − min_contrast)` (design.md Decision 8). In `config.py`:
  - set the default `background_range` to (96, 255);
  - change the contrast check to require `low − min_contrast ≥ 0`, with a message naming both fields;
  - rewrite the docstring and comments to say the foreground is darker.

  Update `configs/default.json`. Verify: `test_render.py` property tests assert `background − foreground ≥ min_contrast` and `foreground ≥ 0` over many draws; the full-coverage, zero-noise test expects the (now darker) foreground shade; `test_config.py` covers the new rejection (e.g. `background_range=(50, 200)` with `min_contrast=96`).
- [ ] 1.2 Add a render test for the spec scenario "The negative smiley shares the polarity". Render a zero-noise `negative_smiley`. Its disc centre (outside the features) must be the foreground shade. Its corners and an eye centre must be the background shade, and lighter. Verify: the test passes.
- [ ] 1.3 Update `validate.py:ink_fraction()` to count pixels below the percentile midpoint as ink, and update its docstring. Update the fixtures in `test_validate.py` and `test_cli.py` that use `background_range=(40, 40)` / `min_contrast=210` to a valid dark-on-light equivalent (e.g. `(250, 250)` / `210`). Verify: the brightness-separable config still fails validation naming the mean-brightness check; an export's ink fraction for a `circle` is well below 0.5 and larger than for a `ring` at the same parameters.
- [ ] 1.4 Add a Forge test asserting that picoface's framing constant in `_internals/picture_internals.py` equals `ForgeConfig`'s default `radius_fraction` (design.md Decision 4). Implement it after 3.1 if needed. Verify: the test passes, and fails when either value is changed.
- [ ] 1.5 Update `dataset_forge/README.md` (shade and polarity description near line 129). Re-export the default config with seed 0 (`python -m dataset_forge --seed 0`, writing `dataset_forge/output/default-seed0/`), and run `python -m dataset_forge.contact_sheet` to look at it. The re-export overwrites a stale 1,000-per-class export from Phase 5. The old-polarity baseline for 2.2 is kept separately in `dataset_forge/output/default-seed0-old-polarity/`, a copy of `default-2x-seed0` made during exploration. That export's config equals today's `configs/default.json` (2,000 training images per class, background 0–159) apart from its name, and it is the export the design's polarity spike used. Verify: validation passes; the contact sheet shows every class, including `negative_smiley`, as dark on light; the user reviews the contact sheet.

## 2. Measurements and the `activation_maximize()` diagnostic

- [ ] 2.1 Extend `dataset_forge/smoke.py` (design.md Decision 10). Per class, for both the CNN and the VAE, render `activation_maximize()` from a few random starts. Write them as `activation_maximize.png` alongside the existing figures when `--figures` is given. Report each class's edge darkening (mean of the outermost 2-pixel ring minus mean of the next 2-pixel ring inward) in `format_results()`. Verify: running the smoke check with `--figures` on a small export writes the new figure, and prints an edge-darkening table covering every class and both models.
- [ ] 2.2 Sanity-check the flipped dataset with the smoke check, using seeds 0, 1 and 2, the seeds of the existing `default-seed*` exports (from their manifests). Steps:
  - Also export seeds 1 and 2 with the new polarity (`python -m dataset_forge --seed 1`, then `--seed 2`). This overwrites the stale 1,000-per-class Phase 5 exports in `default-seed1/` and `default-seed2/`.
  - Run `python -m dataset_forge.smoke dataset_forge/output/default-seed<N> --seed <N> --figures dataset_forge/output/figures/mediacomp-seed<N>` for N = 0, 1, 2, so each run's training seed equals its export seed.
  - Run the same for the old-polarity baseline: `dataset_forge/output/default-seed0-old-polarity` with `--seed 0`.

  In this change's `diagnostics.md`, record the results, per-class tables, and the edge-darkening comparison. For every run, also record the export directory, export seed, training seed, git commit, and machine. Verify:
  - `diagnostics.md` holds all four runs;
  - the flipped seed-0 run's CNN and VAE held-out accuracies are each within 0.01 of, or above, the old-polarity seed-0 run's;
  - the three flipped runs give the ranges used in 4.2.
- [ ] 2.3 **Gate: the instructor reviews the `activation_maximize` results.** Show the user the flipped and old-polarity `activation_maximize.png` figures and the edge-darkening table. Record their decision in `diagnostics.md`: either no action, or switch the ascent blur to replicate padding. If they choose the switch, make it in `linkage_internals.py`, re-run 2.1's figure, and have the user re-check it. Verify: the decision (and, if applicable, the before and after figures) is recorded, and the capstone-linkage tests still pass.
- [ ] 2.4 Flip the stub dataset's shade constants in `_internals/stub_data.py` (background 192, figure 64) and update its comment. Verify: the full test suite passes, including the stub's brightness-only-classifier scenario.

## 3. `picoface.pictures` and `predict()`

- [ ] 3.1 Add `pillow>=10` to `pyproject.toml` `dependencies`. Replace `test_picoface_does_not_declare_the_forges_dependencies` with a test that no entry in `pyproject.toml`'s dependencies carries an exact version pinned in `dataset_forge/requirements.txt` (design.md Decision 6). Verify: `pip install -e .` succeeds; the new packaging test passes, and fails if `Pillow==12.3.0` is added to `pyproject.toml`.
- [ ] 3.2 Create `_internals/picture_internals.py`:
  - picture detection (has a callable `getImage` that returns a Pillow image);
  - picture → H×W×1 uint8 array with the strict grayscale check and its guidance error (for `picture_to_array()` and `predict()`);
  - picture → H×W×3 uint8 RGB array with no grayscale check, via `convert("RGB")` (for the helpers);
  - array input → H×W×C (adding a channel axis to H×W), checked with `check_image_array(..., batch=False)` from `_internals/image_checks.py`;
  - array → picture of a given type in RGB mode;
  - the framing constant (0.65) and the figure threshold (48).

  Add `tests/fake_picture.py`, a minimal `Picture` stand-in (constructor from a Pillow image, `getImage()`, pixel access using RGB triples, like mediaComp). Verify: unit tests cover detection, the grayscale error, and that an array → picture → array round trip is exact.
- [ ] 3.3 Implement `picoface.pictures.picture_to_array()` per the `mediacomp-bridge` spec. Verify: tests for "A grayscale picture converts" (including the `(x, y)` ↔ `[row, col]` mapping, checked with one distinctly shaded pixel) and "A color picture is rejected with guidance".
- [ ] 3.4 Implement `crop_and_center()` per design.md Decision 4. Verify:
  - the spec scenarios pass: an off-centre dark circle on a white 200×100 fake picture comes back square, white-bordered, centred, with the diameter about 0.65 of the side (±0.05); a blank image raises the "no figure found" error;
  - a picture in gives a new picture of the same type in RGB mode, an array in gives an array, and the input is unchanged;
  - a colour picture and an H×W×3 array are accepted, and an H×W×3 array comes back H×W×3, filled with the background colour;
  - a triangle and a circle of the same circumscribed radius come out at the same scale.
- [ ] 3.5 Implement `scale_down(image, size=28)` with `Image.Resampling.BOX` (design.md Decision 5). Verify:
  - the spec scenarios pass: 200×200 → 28×28; a 2-pixel dark ring outline on a 200×200 canvas survives as a visibly darker ring (its minimum along the ring is well below the background); a 200×100 input raises an error naming `crop_and_center()`; an input smaller than `size` is rejected;
  - the input kind and channel count are preserved (H×W gives H×W×1), and the input is unchanged;
  - a gray RGB picture comes back with R == G == B at every pixel, so `picture_to_array()` accepts it;
  - the spec scenario "Grayscale conversion can come before or after scaling": with a channel-mean grayscale function in the test, both orders on a colour 200×200 picture give 28×28 gray pictures that differ by at most 2 at any pixel (integer truncation in the grayscale plus Pillow's rounding in each order).
- [ ] 3.6 Implement `save_images(images, folder, scale=8)` per design.md Decision 7. Verify:
  - it writes RGB PNGs named `image_000.png` etc. into a folder it creates if needed, and returns paths in order;
  - it accepts a single image and a batch;
  - reopening a saved file gives a 224×224 RGB image made of solid 8×8 blocks;
  - the spec's round trip (load as a fake picture → `scale_down` → `picture_to_array`) exactly reproduces the original.
- [ ] 3.7 Make `predict()` accept a picture: if the input is a picture, convert it with the shared converter at the placeholder comment ahead of `_check_single_image()` (see 3.11), so the image-array checks run on the result; never crop or resize it. Verify: tests for the three new `model-interface` scenarios (same result as passing the array; a 200×200 picture raises the model's shape error; a colour picture raises the grayscale error), and all existing `predict()` tests still pass.
- [ ] 3.8 End-to-end test on a trained model. Train a classifier briefly on a small stub dataset (no mediaComp). Draw a dark filled circle off-centre on a large white fake picture, then run it through `crop_and_center` → `scale_down` → `predict`. Repeat with a dark-blue circle, converting to grayscale (channel mean) first in one run and last, just before `predict`, in another. Verify: the test runs in the normal suite within its time budget, both grayscale orders predict the same class, and it predicts `circle` (or, if the stub classes don't include a circle, the pipeline produces an image of the model's shape that `predict` accepts; document which).
- [ ] 3.9 Add `tests/test_pictures_mediacomp.py`, an optional test against the real PyPI mediaComp (design.md Decision 1). At the top: `try: import mediaComp` / `except Exception: pytest.skip(..., allow_module_level=True)`. This catches `OSError` from sounddevice without PortAudio as well as `ImportError`. Using real `makeEmptyPicture`, `addOvalFilled`, and `makePicture`, without ever calling `show()`, run the 3.3–3.7 round trips and the `crop_and_center` → `scale_down` → `predict` pipeline, and check that `getRed(getPixelAt(...))` works on every returned picture. Do not add mediaComp to `dev` extras. The real-mediaComp environment for this machine already exists (design.md Decision 1): run the module with `~/.venvs/picoface-mediacomp/run python -m pytest tests/test_pictures_mediacomp.py`. Once `picoface.pictures` exists, the editable picoface install there picks it up. Verify: under that wrapper, the module runs (not skipped) and passes; in the project's own `venv`, the module is reported as skipped, not errored.
- [ ] 3.10 Manual check with the real mediaComp on Windows, covering what automated tests can't (`show()` needs a display). In Thonny on Windows, install `mediaComp` and picoface through Tools → Manage packages, then:
  - draw a circle with `makeEmptyPicture` and `addOvalFilled`, convert it to grayscale with a pixel loop, then `crop_and_center` → `scale_down` → `predict` using a model trained on the new default export;
  - `save_images(generate(vae, 8), ...)` then `makePicture(path)`, `show(...)`, and `pictureTool(...)`.

  Verify: both work without errors, the pictures display at a readable size, and the result is recorded in `diagnostics.md` (mediaComp version, Thonny version, Windows version).
- [ ] 3.11 Finish the picture-vs-array audit that `dataset-validation` handed over (its `audit.md`). `predict()` already has a placeholder comment where the picture conversion goes, ahead of its image-array checks; put 3.7's conversion there, so pictures get the same checks. Use the audit's vocabulary (*picture* for a Picture-like object, *image array* for a uint8 numpy array, *dataset* for a `Dataset`) in every `picoface.pictures` docstring and in `predict()`'s docstring. The audit row for `crop_and_center()` and `scale_down()` should record, as deliberate, that they also accept an H×W array with no channel axis (unlike `predict()`), and that they accept colour so that grayscale conversion can happen in either order. Point `generate()`, `activation_maximize()`, and `GeneratedImagesReport.images` to `save_images()` for viewing in mediaComp. Leave `_require_dataset()`'s picture test as a plain `hasattr(obj, "getImage")` rather than switching it to 3.2's stricter check: it only chooses the wording of an error message, and `tests/test_wrong_input.py`'s `_FakePicture` (whose `getImage()` returns `None`) relies on it. Verify: add `pictures`' rows to `dataset-validation`'s `audit.md` (or to this change's own notes if that change is archived), and update that file's opening "Case (task 4.1)" line to say the picture rows were added by `mediacomp-bridge` 3.11; confirm that each docstring matches.

## 4. Documentation

- [ ] 4.1 Add a "Using picoface with mediaComp" section to `README.md`:
  - the draw → grayscale (the student's own loop) → `crop_and_center` → `scale_down` → `predict` walkthrough, noting that the grayscale step can come anywhere before `predict`;
  - `save_images` → `makePicture` → `show`;
  - why each step exists (size, framing, dark on light), and that a wrong prediction on a drawing is something to investigate, not a bug;
  - an install note: mediaComp comes from PyPI (`pip install mediaComp`) and is installed separately, since picoface does not depend on it or install it. picoface works without it; only this section's walkthrough needs it;
  - a Thonny-on-Windows note: install torch, picoface, and mediaComp through Tools → Manage packages. The Windows PyPI torch wheel is already CPU-only, so no special index URL is needed.

  Verify: every code line in the section runs in 3.9's optional real-mediaComp test, except `show()` and `pictureTool()`, which are covered by 3.10's manual check; the user reviews the section.
- [ ] 4.2 Update README "What to expect" from 2.2's measurements, and describe the dataset as dark figures on light backgrounds. Verify: every number in the table matches `diagnostics.md`.
- [ ] 4.3 Update `openspec/ROADMAP.md`:
  - the mediaComp integration (the `mediacomp-bridge` capability, duck typing with no mediaComp dependency) under Capabilities and Key Design Decisions;
  - the polarity flip in the Dataset Forge description;
  - a "Potential future change" entry: replace zero padding in the convolution layers (replicate padding preferred), not needed per this change's spike (design.md Decision 9), plus the `activation_maximize` blur outcome from 2.3.

  Verify: the user reviews the ROADMAP diff.
- [ ] 4.4 Final check: run `pytest` (full suite, Forge included) and `openspec validate mediacomp-bridge --strict`. Verify: both pass.
