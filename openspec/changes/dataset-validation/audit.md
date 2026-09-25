# Picture-vs-array consistency audit

**Case (task 4.1):** when this audit was written, `mediacomp-bridge` was **not yet implemented**: `src/picoface/pictures.py` didn't exist, and `predict()` took only image arrays. The audit therefore covered datasets and image arrays only, and handed the wording for pictures to `mediacomp-bridge` as its task 3.11. The `pictures` rows and the picture wording for `predict()` at the end of the table were added by `mediacomp-bridge` 3.11.

**Vocabulary** (design.md Decision 10):
- *picture*: a mediaComp-style `Picture`;
- *image array*: a uint8 numpy array (H×W×C, or N×H×W×C for several);
- *dataset*: a `Dataset` from `load_dataset()`.

| Module | Name | Current wording | Problem | Fix |
|---|---|---|---|---|
| `datasets` | `load_dataset(path)` | Takes a path, returns a `Dataset`; documents the errors it raises | None | Docstring already updated in task 2.6 |
| `datasets` | `Dataset` | "images, integer labels, and class names" | Didn't say what form `images` takes | Docstring now says uint8 image array shaped (count, height, width, channels), 1 or 3 channels (task 1.3) |
| `classifier` | `build_classifier(data)` | "sized for `data`'s class count and image shape" | Doesn't say `data` must be a dataset; a path or image array failed with `AttributeError` | Docstring: "`data` is a dataset from `load_dataset()`". Raises `TypeError` for anything else (task 3.1) |
| `classifier` | `build_classifier_from_shape(num_classes, input_shape)` | "explicit class count and image shape" | Accepts any channel count, but datasets and `predict()` accept only 1 or 3, so a model built for 2 or 4 channels can never be trained | **ROADMAP**: out of scope here (design.md Non-Goals) |
| `classifier`, `generator` | `train(model, data)` | "Train `model` on `data`" | `data` has no annotation and no description; the wrong kind failed with `AttributeError` | Annotate `data: Dataset`; docstring names a dataset from `load_dataset()`; `TypeError` for anything else (task 3.1) |
| `classifier`, `generator` | `evaluate(model, data)` | "accuracy ... of `model` on `data`" | Same as `train` | Same as `train` |
| `classifier`, `generator` | `predict(model, image)` | "predicted class name for a single `image`" | Didn't say *image array*, dtype, or shape. A 2-D array reported a truncated shape; float arrays were silently accepted; a batch crashed inside torch | Docstring rewritten; array checks raise the model's `ShapeError` (task 3.2). Wording for pictures: `mediacomp-bridge` 3.11 |
| `classifier`, `generator` | `ShapeError` | "input shape or class count" / "input/output image shape" | Now also raised for image-array format (dtype, channels) | Docstring broadened to "shape or format, or a class count" (task 3.2) |
| `generator` | `build_autoencoder(data)`, `build_vae(data)` | "sized for `data`" | Same as `build_classifier` | Same as `build_classifier` |
| `generator` | `generate(model, n)` | "Sample `n` new images" | Doesn't say what comes back | Docstring: returns a uint8 image array shaped (n, height, width, channels) |
| `linkage` | `classify_generated(classifier_model, generator_model, data, n)` | "`data` is the labeled data the generator was trained on" | Doesn't say dataset | Docstring: "the dataset (from `load_dataset()`)"; `TypeError` for anything else (task 3.1) |
| `linkage` | `GeneratedImagesReport` | "`images[i]` was generated for class ..." | `images`' form not stated | Docstring: `images` is a uint8 image array, one image per row |
| `linkage` | `activation_maximize(model, target_class)` | "Returns one image in the model's image shape" | "image" is ambiguous now that pictures exist | Docstring: returns one uint8 image array shaped (height, width, channels) |
| `linkage` | `ShapeError` | "models or data used together disagree" | None | None |
| `viz` | `plot_training_history(history)` | Takes a `TrainingHistory` | None: no images | None |
| README | Quickstart, "Getting a dataset" | `data.images[0]`, `generate(...)` "new images"; format "uint8 N×H×W×C" | Channel rule and `classes.json` key rule not stated; no mention of errors or the new properties | Updated in task 4.4 |
| `classifier`, `generator` | `predict(model, image)` (pictures) | Took only image arrays | Pictures needed wording | Docstring: `image` is an image array or a grayscale picture of the model's size, converted as `picture_to_array()` converts it and never cropped or resized; `PictureError` for a picture that isn't grayscale (`mediacomp-bridge` 3.7, 3.11) |
| `pictures` | `picture_to_array(picture)` | New | None | Docstring: takes a *picture*, returns a uint8 *image array* shaped (height, width, 1); `PictureError` if not grayscale, `TypeError` for anything but a picture |
| `pictures` | `crop_and_center(image)`, `scale_down(image, size)` | New | Deliberately more lenient than `predict()`: they accept an H×W image array with no channel axis, and colour as well as grayscale, so a student's grayscale step can come before or after them | Docstrings: `image` is a *picture* or an *image array*, in colour or grayscale; a picture gives back a new picture of the same type, an image array gives back an image array with the same channel count |
| `pictures` | `save_images(images, folder, scale)` | New | None | Docstring: takes one *image array* or several, e.g. from `generate()`, `activation_maximize()`, or a report's `images`; returns file paths for mediaComp's `makePicture()` |
| `generator`, `linkage` | `generate()`, `activation_maximize()`, `GeneratedImagesReport.images` | Returned image arrays with no way to view them in mediaComp | None | Docstrings point to `pictures.save_images()` for viewing in mediaComp |
