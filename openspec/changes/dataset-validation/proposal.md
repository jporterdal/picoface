## Why

`load_dataset()` checks nothing it reads. A probe of 14 malformed bundles found two kinds of failure:
- **Loud but unclear.** `KeyError: '1'` for a `classes.json` with a gap in its indices. `ValueError: not enough values to unpack` from `repr()` when the images have no channel axis.
- **Silent.** Float images, label arrays of the wrong length, labels outside the class range, and duplicate class names all load without complaint. They then fail later inside training or linkage, far from the cause, or never fail and give wrong results.

Students need errors that name the problem where it happens. Once a `Dataset` is known to be well-formed, it can also expose `height`, `width`, and `num_classes` directly.

The mediaComp bridge (`mediacomp-bridge`, in progress) introduces a second kind of image: a mediaComp `Picture`. Students will now handle both pictures and image arrays, so each public function needs to say clearly which one it takes, and refuse the wrong one with a helpful message.

## What Changes

- **New `DatasetError(ValueError)`**, importable from `picoface.datasets`. Every message states what was expected, what was found, and what to do about it.
- **`load_dataset()` validates the bundle file by file**, and names the offending file in each error:
  - The `.npz` is readable and contains `images` and `labels`. Object/pickled arrays (e.g. images of different sizes) are rejected with an explanation.
  - `classes.json` is a JSON object whose keys are exactly `"0"` … `"K-1"`, with non-empty string values.
  - A missing file still raises `FileNotFoundError`, now with a message saying which file was expected, and where.
- **`load_dataset()` makes lossless conversions**: integer images whose values all fall within 0–255 become `uint8`, and integer labels become `int64`. It never guesses. Float images (0–1 or 0–255?), float labels, and out-of-range values are rejected with a suggested fix.
- **`Dataset` enforces its own invariants at construction**, so datasets built by the stub generator or by hand are checked too:
  - `images` is `uint8` N×H×W×C, with N ≥ 1 and C ∈ {1, 3}: grayscale, as the mediaComp bridge produces, or RGB.
  - `labels` is a 1-D integer array of length N, with every value in 0…K−1.
  - `class_names` has at least two entries, all distinct strings.
  - **BREAKING (only for malformed data):** grayscale N×H×W images, float images, and label arrays of the wrong length or shape, which loaded before, are now rejected. The error for N×H×W images says how to add the channel axis.
- **`load_dataset()` warns** (`DatasetWarning`, a `UserWarning`) when a class in `classes.json` has no images in the bundle.
- **New `Dataset` properties:** `height`, `width`, and `num_classes`, computed from `images` and `class_names`. `class_names` is unchanged.
- **Fix `Dataset.__repr__`**, which currently prints the labels count from `images.shape[0]` and so hides a length mismatch.
- **Picture-vs-array consistency audit of the public API.** Check every public function's parameter names, docstrings, and errors, so none looks as if it takes a mediaComp picture when it actually needs an array, or the reverse. The fixes that come out of it are part of this change:
  - Functions that take `data` raise a `TypeError` naming what they were given (e.g. a picture or a bare array) and pointing to `load_dataset()`, or to `predict()` for a single image.
  - `predict()` rejects image arrays of the wrong dtype or rank with a message that suggests the fix. That includes a 2-D H×W array, which today produces a confusing `image shape (28,)` message. It no longer silently accepts float arrays.
  - Docstrings use one vocabulary: *picture* means a mediaComp-style `Picture`; *image* or *image array* means a uint8 numpy array.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `data-contract`: the interchange format pins images to uint8 N×H×W×C with C ∈ {1, 3} and defines the exact `classes.json` shape. `Dataset` enforces its invariants and gains `height`, `width`, and `num_classes`. Malformed bundles raise `DatasetError` with explanatory messages. Lossless conversions at load time. A warning for classes with no examples. The size- and class-agnostic requirement is narrowed to allow only grayscale or RGB.
- `model-interface`: new requirement that public functions reject the wrong kind of input (a non-`Dataset` where a dataset is expected, or a malformed image array for `predict()`) with an error that names what was received and how to fix it.

## Impact

- `src/picoface/datasets.py`: `DatasetError`, `DatasetWarning`, validation in `load_dataset()` and `Dataset.__post_init__`, new properties, `__repr__` fix.
- `src/picoface/_internals/`: a shared image-array check used by `Dataset` and `predict()`; a `Dataset` type check in `model_api.py` (`train`, `evaluate`) and at the `build_*()` and `classify_generated()` entry points.
- Public docstrings in `classifier.py`, `generator.py`, `linkage.py`, `_internals/model_api.py`, and, once it exists, `pictures.py`. Also the README's usage examples, if the audit finds inconsistent wording.
- Tests: new `tests/test_datasets.py` cases for each malformed-bundle case and the properties; new wrong-input tests for the model verbs. Existing test fixtures that build `Dataset` directly are already valid and need no change.
- Sequencing: the dataset validation and properties are independent of `mediacomp-bridge`. The picture half of the audit should run after `mediacomp-bridge` is implemented, because that change makes `predict()` accept pictures (see design.md).
- No new dependencies. Well-formed bundles, including every Dataset Forge export, load exactly as before.
