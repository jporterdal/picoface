## 1. Dataset invariants and properties

- [ ] 1.1 Add `DatasetError(ValueError)` and `DatasetWarning(UserWarning)` to `src/picoface/datasets.py`, with docstrings (design.md Decision 4). Verify: both import from `picoface.datasets`, and `issubclass(DatasetError, ValueError)` and `issubclass(DatasetWarning, UserWarning)` hold in a test.
- [ ] 1.2 Create `src/picoface/_internals/image_checks.py` with one function that checks an image array's rank, dtype (uint8), and channel count (1 or 3), taking the error class and a context string for the message. It handles both a batch (N×H×W×C) and a single image (H×W×C). The rank error includes the `[..., np.newaxis]` hint; the float-dtype error includes the `(x * 255).round().astype(np.uint8)` hint (design.md Decisions 5 and 7). Verify: unit tests in a new `tests/test_image_checks.py` cover each rejection for both batch and single modes, asserting the error class and that the message contains the shape or dtype found.
- [ ] 1.3 Add `Dataset.__post_init__` to enforce the array invariants in the order from design.md Decision 6:
  - images, through the checker from 1.2, with N ≥ 1;
  - labels: 1-D, integer dtype, length N, values in 0…K−1 (report the offending value and its count);
  - class_names: at least 2, all `str`, no duplicates (name the duplicate and its indices).

  No conversion (Decision 2). Verify: new `tests/test_datasets.py` cases construct `Dataset(...)` directly for each violation and assert `DatasetError` plus the key facts in the message; the existing test helpers in `test_joint_model.py`, `test_linkage.py`, and `test_generator.py` still construct successfully.
- [ ] 1.4 Add `height`, `width`, and `num_classes` as `@property` on `Dataset` (Decision 3). Verify: a test for the spec scenario "Reading image size and class count" (a 28×28×1 bundle with 3 classes); `Dataset` equality and `repr` are unchanged for a well-formed dataset.
- [ ] 1.5 Fix `Dataset.__repr__` to show `labels.shape` rather than reusing N from images. Verify: `test_repr_shows_shape_not_pixels` still passes, and a new assertion checks that the repr's labels shape comes from `labels`.

## 2. load_dataset() file checks, conversion, and warning

- [ ] 2.1 Rewrite `load_dataset()` to check the `.npz` in order (Decision 6):
  - the file exists (`FileNotFoundError` naming the path);
  - the file is an `.npz` archive, not a single array;
  - `images` and `labels` are present (on failure, list the arrays that are present);
  - neither is an object array (catch numpy's `allow_pickle` `ValueError` and explain that all images must share one size).

  Verify: `tests/test_datasets.py` cases for a missing file, a `.npy` file, a missing `labels` key, and a ragged object array each raise the specified type, and the message names the file.
- [ ] 2.2 Add the `classes.json` checks:
  - the file exists (`FileNotFoundError` saying it must be in the same directory as the `.npz`, naming the directory);
  - the file is valid JSON;
  - the top level is an object;
  - the keys are exactly `"0"`…`"K-1"` (report missing and extra keys);
  - the values are non-empty strings.

  Verify: tests for each case, including `{"0": "a", "2": "b"}` and a JSON list, assert `DatasetError` or `FileNotFoundError` and that the message names `classes.json`.
- [ ] 2.3 Add lossless conversion before construction: integer images with min ≥ 0 and max ≤ 255 become uint8; integer labels become int64; out-of-range integer images raise `DatasetError` reporting the min and max (spec "Lossless conversion at load time"). Verify: an int64 image bundle within range loads as uint8 with equal values; an int16 bundle containing 300 is rejected; float images and float labels are rejected, not converted.
- [ ] 2.4 Wrap construction so a `DatasetError` from `Dataset.__post_init__` is re-raised with the `.npz` path added (`from err`) (Decision 1). Verify: loading an N×H×W bundle raises `DatasetError` whose message contains both the file name and the shape found, and whose `__cause__` is the original error.
- [ ] 2.5 After construction, issue `DatasetWarning` naming every class with zero labels (Decision 8). Verify: a test using `pytest.warns(DatasetWarning)` for a 3-class bundle with no `c` labels, checking that the dataset still loads with all three names; a stub-built dataset missing a class does not warn; the full test suite runs with no new warnings.
- [ ] 2.6 Update the module docstring in `datasets.py` and the `load_dataset()` docstring to describe the accepted format (uint8, C ∈ {1, 3}), the conversions, and the errors it raises. Verify: reading them, they match the data-contract spec delta.

## 3. Wrong-input checks on the model verbs

- [ ] 3.1 Add `_require_dataset(obj, verb)` to `_internals/model_api.py`. It raises `TypeError` and describes a picture (has `getImage`), an `np.ndarray` (with its shape), a `str`/`Path`, or any other type, as in design.md Decision 9. For a picture or array, the message also points to `predict()`. Call it first in `train`, `evaluate`, `build_classifier`, `build_autoencoder`, `build_vae`, and `classify_generated`. Verify: tests for the spec scenarios "A picture passed where a dataset is expected" (use a minimal fake class with `getImage()`) and "A file path passed where a dataset is expected", plus an ndarray case, each asserting `TypeError` and the key phrases.
- [ ] 3.2 In `predict()`, run the single-image check from 1.2 with the model's `shape_error_cls` before `_preprocess_images()`. Also reject 4-D input as a batch, pointing to `evaluate()` or `data.images[i]`. If `mediacomp-bridge` is already implemented, the check runs on the array produced by its picture conversion. If it isn't, leave the picture branch's position clear, with a comment (Decision 11). Broaden each arm's `ShapeError` docstring to "image shape, format, or class count". Verify: tests for the spec scenarios "A two-dimensional image array", "A float image array", and "A batch passed to predict" on a small classifier, asserting the arm's `ShapeError` and that a `BaseShapeError` handler catches it; the existing `test_classifier.py` shape-mismatch test still passes.
- [ ] 3.3 Optionally switch internal uses of `len(data.class_names)` and `data.images.shape[1:3]` to the new properties where it reads better (Decision 3). Verify: the full suite passes.

## 4. Picture-vs-array consistency audit (after mediacomp-bridge)

- [ ] 4.1 Check that `mediacomp-bridge` has been implemented (`src/picoface/pictures.py` exists and `predict()` accepts pictures). If it hasn't, do 4.2–4.3 for datasets and arrays only, and add a note to `openspec/changes/mediacomp-bridge/tasks.md` asking for the picture wording to be audited there. Verify: record which case applied at the top of the audit notes in 4.2.
- [ ] 4.2 Audit every name in `__all__` of `picoface.classifier`, `generator`, `linkage`, `viz`, `datasets`, and `pictures`, using the checklist in design.md Decision 10: parameter name, docstring vocabulary (*picture* / *image array* / *dataset*), what is returned, and the error for the wrong kind. Record the findings as a short table in this change's directory (`audit.md`), one row per function: current wording, problem (if any), fix. Verify: `audit.md` covers every exported function and class, and each row with a problem has a fix or a ROADMAP pointer.
- [ ] 4.3 Apply the wording fixes from `audit.md` to docstrings and parameter names, keeping signatures backward compatible (no renaming of a parameter a student might pass by keyword without keeping the old name working). Make `generate`, `activation_maximize`, and `GeneratedImagesReport.images` state that they return uint8 image arrays, and point to `save_images()` if the bridge exists. Verify: re-read each changed docstring against `audit.md`; the full test suite passes.
- [ ] 4.4 Apply the same vocabulary to the README's usage examples and API descriptions, including `data.height`, `data.width`, and `data.num_classes` where a hard-coded size or class count appears. Verify: `grep -n "28" README.md` shows no hard-coded image size in code examples where a property fits; the user reviews the README diff.
- [ ] 4.5 Add any audit findings that need more than wording or an error check to `openspec/ROADMAP.md`. Verify: each "ROADMAP" row in `audit.md` has a matching entry.

## 5. Integration

- [ ] 5.1 Load the default Dataset Forge export (or regenerate it) with the new `load_dataset()`. Verify: `tests/test_real_dataset.py` passes, with no `DatasetWarning` and no change to loaded values.
- [ ] 5.2 Re-run the malformed-bundle probe from the explore session (14 cases: missing key, N×H×W, float images, label length mismatch, labels (N,1), label out of range, float labels, ragged, missing `classes.json`, non-contiguous keys, JSON list, duplicate names, empty dataset, plus the empty-class warning). Verify: every case now raises `DatasetError`, `FileNotFoundError`, or `DatasetWarning` as the spec says, and the user reviews the printed messages for wording.
- [ ] 5.3 Run `openspec validate dataset-validation --strict` and the full test suite. Verify: both pass.
