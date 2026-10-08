## Context

See `proposal.md` for the motivation. The current state:

- `src/picoface/datasets.py` holds a frozen dataclass, `Dataset(images, labels, class_names)`, and `load_dataset()`. That function reads the `.npz`, reads `classes.json`, and builds the dataset without checking anything.
- `Dataset` values are created in three places: `load_dataset()`, the internal stub generator (`_internals/stub_data.py`), and three test helpers that relabel or reorder a stub (`test_joint_model.py`, `test_linkage.py`, `test_generator.py`). All three helpers already produce data that satisfies the rules in this change.
- Downstream code assumes the rules hold without checking them: `images.shape[1:]` is the model's input shape (`classifier.py`, `generator.py`, `check_data`), labels index `class_names` (`linkage_internals.py`), and `__repr__` unpacks four dimensions.
- Errors follow one pattern. Each arm has its own `ShapeError`, subclassing `BaseShapeError(ValueError)` in `_internals/errors.py`. `CapabilityError`/`GeneratorError` cover missing capabilities. Messages state what was expected and what was found (e.g. `linkage.py:64-74`).
- `predict()` goes through `_preprocess_images()`, which adds a batch axis to 3-D input and compares `shape[1:]` with the model's input shape. A 2-D array therefore reports a truncated shape such as `(28,)`. Float arrays pass the check and are divided by 255. A batch passes too, and then fails at `.item()`.
- `mediacomp-bridge` (in progress, not yet implemented) adds `picoface.pictures`, defines a picture as any object whose `getImage()` returns a Pillow image, and makes `predict()` accept pictures.

## Goals / Non-Goals

**Goals:**
- Every `Dataset` value in the system satisfies the interchange format, whether it came from a file, the stub, or a test helper, so downstream code and the new properties can rely on it.
- A student with a malformed bundle gets one error that names the file, the problem, and the fix.
- Every public function makes it clear whether it takes a picture, an image array, or a dataset.

**Non-Goals:**
- Reporting every problem in a bundle at once. Checks stop at the first failure (Decision 6).
- Checking image *content*, such as class balance beyond empty classes, blank images, or duplicate images. Dataset Forge's own `validate.py` covers quality checks for its exports.
- Changing `build_classifier_from_shape()`, which takes an explicit shape and no dataset.
- Accepting new on-disk formats (folders of PNGs, `.npy`, CSV).
- Changing model architectures or the interchange format's file layout.

## Decisions

### 1. Validation is split between `load_dataset()` and `Dataset.__post_init__`

```
  .npz + classes.json
          │
          ▼
  load_dataset()            file-level: npz readable, keys present, no object arrays,
          │                 classes.json shape; lossless dtype conversion; empty-class warning
          │   wraps DatasetError from below with the file path
          ▼
  Dataset.__post_init__     array-level invariants: images uint8 N×H×W×C, C ∈ {1,3}, N ≥ 1;
          ▲                 labels 1-D int, len N, in 0…K−1; class_names ≥ 2, distinct, str
          │
  stub / test helpers / student-built
```

`load_dataset()` knows file names and the raw JSON. Only `Dataset` sees every construction. `load_dataset()` catches a `DatasetError` raised by the constructor and re-raises it with the path added (`raise DatasetError(f"{path}: {err}") from err`), so file-level and array-level errors look the same to a student.

Alternatives considered:
- **Everything in `load_dataset()`.** Rejected: datasets built by the stub or by hand would go unchecked, so `height`/`width` could not assume four dimensions.
- **Everything in `Dataset`.** Rejected: the constructor can't see `.npz` keys or JSON structure, and can't name the file.

### 2. Conversions happen only in `load_dataset()`; `Dataset` is strict
`load_dataset()` converts integer images within 0–255 to uint8 and integer labels to int64 before it constructs the dataset. `__post_init__` never converts. It only checks, so the frozen dataclass needs no `object.__setattr__`. Labels may be any integer dtype when a dataset is built directly, since every consumer already calls `.long()` or `np.asarray()`. Images must be uint8 exactly.

Converting in the constructor was rejected. It would change arrays that our own code passed in, and hide mistakes in the stub or in tests.

Floats are never converted. We can't tell whether a float image is 0–1 or 0–255, and guessing wrongly gives a dataset that looks fine but trains on nonsense. The error suggests `(images * 255).round().astype(np.uint8)` for the 0–1 case.

### 3. `height`, `width`, `num_classes` are `@property`, not fields
They are computed from `images.shape[1]`, `images.shape[2]`, and `len(class_names)`. As properties they can never disagree with the arrays, they don't appear in `__init__`, `__eq__`, or the dataclass `repr`, and existing constructors need no change. Stored fields set in `__post_init__` would duplicate state and need `field(init=False, compare=False, repr=False)` to stay out of the way.

`channels` was offered and not requested. Code that needs it keeps using `images.shape[3]`.

Internal code may switch to the new properties where that reads better (e.g. `len(data.class_names)` becomes `data.num_classes` in `classifier.py`/`generator.py`), but it doesn't have to.

### 4. Error and warning types
| Situation | Type | Where defined |
|---|---|---|
| Bundle or dataset breaks the format | `DatasetError(ValueError)` | `picoface.datasets` |
| Class with no images, at load | `DatasetWarning(UserWarning)` | `picoface.datasets` |
| `.npz` or `classes.json` missing | `FileNotFoundError` (message improved) | built-in |
| Non-dataset passed as `data` | `TypeError` | built-in |
| `predict()` image array of wrong rank, dtype, or batch | the model's `ShapeError` | existing per-arm classes |

`DatasetError` subclasses `ValueError`, so existing `except ValueError` code keeps working. It lives in `datasets.py` because that module is already public and depends on nothing that would create an import cycle. `FileNotFoundError` stays for missing files: it is the most accurate type, and `open()` already raises it. Only the message changes.

Wrong kind of object is a type error, so `TypeError` is the idiomatic choice, and it is distinct from "right kind, wrong shape". For `predict()`, rank, dtype, and batch problems use the arm's `ShapeError`. That matches today's shape-mismatch error from the same call, and a student catching `BaseShapeError` still catches all of them. Its docstring broadens to "image shape, format, or class count". A separate `ImageFormatError` was rejected as one more type for students to learn.

### 5. One shared image-array check
`_internals/image_checks.py` (new) holds a single function that checks a uint8 image array's rank, dtype, and channel count, with a parameter for the error class and wording context. `Dataset.__post_init__` calls it for a batch (with `DatasetError`); `predict()` calls it for a single image (with the model's `ShapeError`) before `_preprocess_images()`. A single checker means the two rules can't drift apart, and the channel-axis hint (`images[..., np.newaxis]`) is written once.

`train()` and `evaluate()` need no new image checks: they take a `Dataset`, which is already valid, and `check_data()` compares its shape with the model's.

### 6. First failure wins, in a fixed order
Checks run from structure down to values:
1. file readable
2. arrays present
3. regular (non-object) arrays
4. images rank, then channels, then dtype, then range
5. labels rank, then length, then dtype, then range
6. `classes.json` structure, then duplicate names

Each check can assume the earlier ones passed, which keeps messages short and specific ("labels has 13,998 entries but images has 14,000" rather than a list of consequences). Collecting every problem was rejected: most malformed bundles have one root cause, and a list of five errors that all follow from a missing channel axis is harder for a student to act on.

### 7. Message format
Each message has three parts, following `linkage.py`: what was expected, what was found, and the fix. For example:

```
train.npz: images has shape (14000, 28, 28), but picoface needs images shaped
(count, height, width, channels) with 1 channel for grayscale or 3 for RGB.
For grayscale images, add the channel axis: images = images[..., np.newaxis]

classes.json: expected keys "0" to "2" (one per class), but found "0", "2";
key "1" is missing. Number the classes 0, 1, 2, ... with no gaps.

train.npz: labels contains 5 (in 12 labels), but classes.json defines 3
classes, so labels must be 0 to 2.
```

These are drafts. The user expects to edit the final wording by hand, so tests assert the facts that must appear (shapes, values, file name), not exact strings.

### 8. The empty-class warning fires only in `load_dataset()`
A missing class in a file on disk is probably a data-preparation mistake. A `Dataset` built by splitting, relabelling, or filtering may leave out a class on purpose (test helpers do this). Warning from `__post_init__` would add noise there, and the test suite would need filters. `np.bincount(labels, minlength=K)` gives the per-class counts used both for this warning and the range check.

### 9. Wrong-kind detection for `data` arguments
A `_require_dataset(obj, verb)` helper in `_internals/model_api.py` checks `isinstance(obj, Dataset)` and builds the message from what it was given:
- an object with `getImage()`: "a picture" (the same duck test as `mediacomp-bridge`; reuse its predicate if it exists by then);
- an `np.ndarray`: "an image array of shape …";
- a `str` or `Path`: "a file path; load it first with `load_dataset()`";
- otherwise: the type name.

It runs first in `train`, `evaluate`, `build_classifier`, `build_autoencoder`, `build_vae`, and `classify_generated`, before `data.images` is read. No test passes a stand-in for `Dataset`, so a strict `isinstance` check breaks nothing.

### 10. The picture-vs-array audit
The audit is a checklist task that produces fixes, not a separate document. For each public name in `__all__` of `classifier`, `generator`, `linkage`, `viz`, `datasets`, and (once it exists) `pictures`, check:
- the parameter name (`image` or `images` for arrays, `picture` for pictures, `data` for datasets);
- the docstring (states which kind it takes and returns, using the vocabulary *picture* / *image array* / *dataset*);
- the error raised for the wrong kind (Decisions 4 and 9).

Functions that return arrays (`generate`, `activation_maximize`, `GeneratedImagesReport.images`) say they return uint8 image arrays, and, once the bridge exists, point to `save_images()` for viewing in mediaComp. The README's usage examples get the same pass. Findings that need more than wording or an error check go to `openspec/ROADMAP.md` instead of this change.

### 11. Sequencing with `mediacomp-bridge`
Task groups 1–3 (dataset validation, properties, `data`-argument checks, `predict()` array checks) don't depend on the bridge and can land first. Task group 4 (the audit) should run after `mediacomp-bridge` is implemented, because that change adds `pictures.py` and makes `predict()` accept pictures. Both changes touch `predict()`: the bridge converts a picture to an array first, then this change's array check runs on the result. If this change is implemented first, `predict()`'s check must leave room for the bridge's picture branch ahead of it. The bridge's `stub_data.py` shade flip doesn't interact with this change.

## Risks / Trade-offs

- **[Bundles that loaded before now fail]** → Only malformed ones: N×H×W images, floats, wrong-length labels. Each message gives the fix. Every Dataset Forge export already conforms, and `tests/test_real_dataset.py` is the check.
- **[Validation cost on large datasets]** → The value-range checks are single passes over the array (`min`/`max`, `bincount`): milliseconds for 14,000×28×28, and they run once per construction.
- **[`isinstance` rejects future dataset-like types]** → There is one dataset type by spec ("Shared dataset return type"). A new type would be a spec change anyway.
- **[Restricting C to {1, 3} conflicts with a future RGBA or multispectral dataset]** → Not planned. The course is grayscale, and RGB covers mediaComp's pictures. Widening later means relaxing one check and one spec line.
- **[Draft message wording goes stale]** → Tests assert content (shapes, values, names), not exact text, so hand edits don't break them.
- **[Overlap with `mediacomp-bridge` in `predict()`]** → Decision 11 fixes the order. Whichever change lands second rebases its `predict()` edit onto the other.
