## Why

picoface prints nothing while it works. Its only output is the `DatasetWarning` for an empty class. `train()` is the problem. An exploration run on this repo's default Dataset Forge export (7,000 images, 28×28 grayscale, 7 classes) on an 8-thread machine gave these times:
- CNN: 67 s, 6.6 s per epoch;
- VAE: 179 s, 18 s per epoch.

For all of that time the student sees nothing. On an older student laptop it could last several times longer. A first-year student in Thonny can't tell a slow run from a frozen one, or know how long is left. The other public calls finish in under half a second, and they also run without saying what they did.

## What Changes

- **The student-facing functions that do work print to standard output by default**, and take a new keyword argument `verbose=True`. Passing `verbose=False` makes them silent. The functions covered are:
  - `load_dataset()`;
  - `build_classifier()`, `build_classifier_from_shape()`, `build_autoencoder()` and `build_vae()`;
  - `train()`, `evaluate()`, `predict()` and `generate()`;
  - `classify_generated()` and `activation_maximize()`;
  - `save_images()`, from `mediacomp-bridge`'s `picoface.pictures`.

  The other `picoface.pictures` helpers (`picture_to_array()`, `crop_and_center()` and `scale_down()`) stay silent and take no `verbose`. Each returns a picture or image array at once, and students look at the result with mediaComp's `show()`.

  Existing calls keep working unchanged. Return values, raised errors and random-number use are identical with output on or off.
- **`train()` reports progress:**
  - a header describing the job: model, image count, epochs, batches per epoch, batch size, learning rate, and number of weights;
  - a progress bar within each epoch that redraws in place, showing batches done;
  - a permanent line after each epoch, with every metric the model reports, plainly labelled (for example "training accuracy", not "accuracy"). It also shows the epoch's time, total elapsed time, and estimated time remaining;
  - a final summary.

  The guiding rule is to show more rather than less. Where a number could mislead, the output repeats and explains it rather than hiding it; one example is a VAE's total loss, which is negative.
- **The fast calls print a short summary of what they did:**

  | Function | Summary |
  |---|---|
  | `load_dataset()` | Image count, image size and colour, class count, and per-class image counts |
  | `build_*()` | Model kind, image shape, classes, and number of weights |
  | `evaluate()` | Overall and per-class accuracy |
  | `predict()` | The predicted class and its probability, plus the next most likely classes |
  | `generate()` | How many images, and the array shape |
  | `classify_generated()` | Per-class and overall agreement |
  | `activation_maximize()` | The class, the number of steps, and the class score before and after |
  | `save_images()` | One line: how many images were saved, and the folder they went to |
- **`TrainingHistory` gets a short `repr`**, so typing `train(...)` at the shell no longer echoes nine lists of numbers. The repr gives the epoch count, the time, and the final value of every recorded metric, and names the fields that hold the per-epoch lists. `print(history)` shows the full per-epoch table, so nothing becomes harder to see.
- **Output is plain ASCII**, flushed as it is printed, and never written to standard error, which Thonny shows in red, as if something had failed.
- **`dataset_forge/smoke.py`** passes `verbose=False` where it calls `predict()` once per image, and in the `activation_maximize()` diagnostic that `mediacomp-bridge` adds, which calls it several times per class for each model. Its other calls keep their output, which acts as a heartbeat during long smoke runs.

## Capabilities

### New Capabilities
- `progress-output`: what picoface prints while it works, and how to turn it off. Covers:
  - the shared `verbose` switch and the output channel and format rules;
  - `train()`'s progress, per-epoch and final reports;
  - the summaries from loading, building, evaluating, predicting, generating, the capstone functions and saving images;
  - the guarantee that output never changes results.

### Modified Capabilities
- `model-interface`: `train()`'s signature gains `verbose=True`. The shared training history gains a short, informative `repr` and a full per-epoch printed form.
- `shape-classifier`: the `train()` requirement's signature, which repeats the one in `model-interface`, gains `verbose=True`.
- `mediacomp-bridge`: the `save_images()` requirement's signature gains `verbose=True`. This delta applies to the main spec that `mediacomp-bridge` creates when it is archived.

## Impact

- **`src/picoface/_internals/`:**
  - a new `progress.py` holds the formatting and printing helpers: the progress bar, duration and number formatting, and metric labels;
  - `model_api.py` changes `train()`, `evaluate()`, `predict()` and `generate()`, and `TrainingHistory` gets a `__repr__` and `__str__`;
  - `linkage_internals.py`: `_ascend()` also reports the class score before and after;
  - the model classes each get a display name for the messages.
- **Public modules:** `datasets.py`, `classifier.py`, `generator.py` and `linkage.py` gain the `verbose` parameter and summary output. In `pictures.py`, only `save_images()` does.
- **`dataset_forge/smoke.py`:** passes `verbose=False` in its per-image `predict()` loops and its `activation_maximize()` diagnostic loop.
- **Tests:**
  - new tests for the output: it appears by default, `verbose=False` is silent, results are identical, it is ASCII-only, and it goes only to stdout;
  - the existing tests are unaffected, because pytest captures output.
- **Docs:** the README Quickstart shows sample output and mentions `verbose=False`.
- **No new dependencies.** tqdm and `logging` are deliberately not used; `design.md` explains why.
- **Out of scope:** making training faster. That is a separate future change. The README's "roughly 10 and 20 seconds" claim is stale (67 s and 179 s were measured), and that change should fix it.
- **Sequencing:**
  - `dataset-validation` is committed. It should be archived before this change is applied.
  - `mediacomp-bridge` is applied and archived before this change, which builds on it. This change adds `verbose` to its `save_images()`, silences its smoke-check diagnostic, and edits `predict()` after its picture conversion is in place. It also modifies a requirement from the `mediacomp-bridge` main spec, which exists only once that change is archived.
