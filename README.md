# picoface

A small, pip-installable, PyTorch-backed Python library for building and
training real image classifiers and generators from simple, well-named
function calls — designed for first-year non-CS students to train on an
ordinary laptop CPU in seconds to minutes.

**Status:** Phases 0–6 (data contract, classifier arm, generator arm, capstone
linkage, dataset generation, and end-to-end tuning) are implemented. Phase 7
(student docs and PyPI packaging-prep, this README included) is in progress.
Getting a real dataset to students and per-arm template notebooks are
explicitly out of this MVP — see [Getting a dataset](#getting-a-dataset) and
[Layout](#layout) below. See
[`openspec/ROADMAP.md`](openspec/ROADMAP.md) for the full project plan.

## Constraints

- **CPU-only, always.** No NVIDIA/CUDA or AMD/ROCm GPU is required or used —
  GPU-specific optimization is explicitly out of scope for this project.
- **Training completes in seconds to minutes** on an old laptop CPU, by
  design. The two models this library walks through (a small CNN classifier
  and a supervised VAE) train end-to-end at their default settings in roughly
  10 and 20 seconds respectively on a real dataset.

## Quickstart

```python
from picoface.datasets import load_dataset
from picoface.classifier import build_classifier, train, evaluate, predict
from picoface.generator import build_vae, generate
from picoface.linkage import classify_generated, activation_maximize

data = load_dataset("path/to/dataset.npz")  # see "Getting a dataset" below
print(data.height, data.width, data.num_classes)  # image size and class count

# Arm 1: an independent classifier, later used as the capstone's judge.
classifier_model = build_classifier(data)
train(classifier_model, data)
print(evaluate(classifier_model, data))           # accuracy on `data`
print(predict(classifier_model, data.images[0]))  # predicted class name

# Arm 2: the student's model -- one model that both classifies and generates.
vae_model = build_vae(data)
train(vae_model, data)          # same train(), works on either model
print(evaluate(vae_model, data))
new_images = generate(vae_model, 8)  # a uint8 image array of 8 new images

# Capstone: tie the two arms together.
report = classify_generated(classifier_model, vae_model, data)
print(report.overall)  # fraction of generated images the classifier agrees with

am_image = activation_maximize(classifier_model, data.class_names[0])  # one image array
```

Images in picoface are uint8 numpy arrays ("image arrays") shaped (height,
width, channels), or (count, height, width, channels) for several. `predict()`
takes one image array, such as `data.images[0]`; the functions that take
`data` need the dataset itself, from `load_dataset()`.

`train`, `evaluate`, and `predict` are the same functions regardless of which
model you built — `picoface.classifier` and `picoface.generator` both
re-export them.

Two related functions exist but aren't part of this walkthrough:
`build_autoencoder()` (an optional, reconstruction-only model — it doesn't
classify or generate, and isn't a required step on the way to `build_vae()`)
and `build_classifier_from_shape()` (builds a classifier from an explicit
class count and image shape, for when you don't have a `Dataset` object yet).

## Getting a dataset

`load_dataset(path)` reads a dataset bundle: an `.npz` file (`images`: uint8
N×H×W×C, with C = 1 for grayscale or 3 for RGB; `labels`: int N) plus a
companion `classes.json` in the same directory mapping label numbers `"0"`,
`"1"`, ... to class names. This is a fixed contract, not a guess — any
`.npz`/`classes.json` pair matching it will load. A bundle that doesn't match
raises a `DatasetError` naming the file, what was wrong, and how to fix it;
a class with no images loads with a `DatasetWarning`.

**picoface does not provide a real dataset itself.** Producing one is a
separate, instructor-side step (the `dataset_forge/` tool in this repo, not
part of the installable package) that happens before students receive
anything. Getting a real dataset into students' hands is explicitly outside
this library's scope.

## Using picoface with mediaComp

If you know [mediaComp](https://pypi.org/project/mediaComp/), you can draw a
picture, classify it with a model you trained, and open picoface's own images
as mediaComp pictures. `picoface.pictures` works with mediaComp's pictures, but
picoface doesn't depend on mediaComp or install it: everything else in
picoface works without it. Install it separately:

```bash
pip install mediaComp
```

**Classifying your own drawing.** Continuing from the Quickstart:

```python
from mediaComp import *
from picoface.pictures import crop_and_center, scale_down, save_images

def make_greyscale(picture):
    for pixel in getPixels(picture):
        gray = (getRed(pixel) + getGreen(pixel) + getBlue(pixel)) // 3
        setColor(pixel, makeColor(gray, gray, gray))
    return picture

picture = makeEmptyPicture(300, 200)             # white, like paper
addOvalFilled(picture, 40, 30, 120, 120, black)  # one figure, dark on light
picture = make_greyscale(picture)                # the model needs grayscale
picture = crop_and_center(picture)               # square, centred, framed like the training images
picture = scale_down(picture)                    # 28x28, the size the model was trained on
print(predict(classifier_model, picture))
```

Why each step is there:
- **Grayscale:** the models were trained on grayscale images, so `predict()`
  rejects a picture whose red, green, and blue values differ, and tells you to
  convert it first. Converting is your job, with a loop like `make_greyscale`
  above. It can come anywhere before `predict()`: `crop_and_center()` and
  `scale_down()` work on colour pictures too.
- **Dark on light:** the training images are dark figures on light
  backgrounds, like mediaComp's default drawing colours.
- **Framing:** every training figure fills about the same share of its image,
  centred. `crop_and_center()` finds your figure (draw just one) and frames it
  the same way, on a square canvas of your background colour.
- **Size:** `scale_down()` shrinks the picture to the model's size, averaging
  the pixels as the training images were, so thin lines fade rather than
  vanish. `predict()` never resizes a picture for you; one of the wrong size
  is rejected.

Both helpers return a new picture and leave yours unchanged, so you can
`show()` each step. A wrong prediction on your drawing is not a bug: your
drawing is crisper and flatter than the noisy training images, and finding
out why the model disagrees with you is part of the exercise.

**Looking at picoface's images in mediaComp.** `save_images()` writes image
arrays as enlarged PNG files and returns their paths:

```python
paths = save_images(generate(vae_model, 8), "generated")
picture = makePicture(paths[0])
show(picture)
```

It works the same for `activation_maximize()` images and a
`classify_generated()` report's `report.images`. Files with the same names in
that folder are overwritten.

**Thonny on Windows:** install `torch`, `picoface`, and `mediaComp` through
Tools → Manage packages. The Windows torch package on PyPI is already
CPU-only, so no special index URL is needed.

## What to expect

These are documented, informational ranges from the reference implementation
running against a real generated dataset (7 shape classes, 2,000 images per
class, dark figures on light backgrounds) — **nothing in this repo enforces
them**. Your own numbers will vary with your dataset, seeds, and
hyperparameters.

| Metric | Typical range |
|---|---|
| CNN (`build_classifier`) held-out accuracy | ~0.99–1.00 |
| VAE (`build_vae`) held-out accuracy | ~0.97–0.98 |
| Reconstruction agreement, most classes | ~0.75–1.0 |
| `classify_generated()` overall agreement | ~0.89–0.93 |

Reconstruction and generation quality is class-dependent: most classes land
solidly in the ranges above, but two classes (`star` and `smiley`) are
currently the weakest performers — a known, named open question in
[`openspec/ROADMAP.md`](openspec/ROADMAP.md), not a bug to work around.

## Layout

- `src/picoface/` — the installable student-facing package (classifier,
  generator, linkage, viz) plus `_internals/` for implementation details.
- `dataset_forge/` — instructor-only, unrestricted dataset generation
  tooling, kept outside the installable package.

Per-arm template notebooks plus a capstone notebook are a planned
nice-to-have, not shipped in this MVP — there is currently no notebook
anywhere in this repo, and the Quickstart above is the primary walkthrough.

## Install (development)

`picoface` only ever runs on CPU — no NVIDIA/CUDA or AMD/ROCm GPU is
required or used (see `openspec/ROADMAP.md`'s binding constraint: training
must complete in seconds-to-minutes on an old CPU-only laptop). Install the
CPU-only PyTorch wheel first, then install picoface:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Skipping the first line still works, but on Linux `pip install -e .` alone
pulls PyPI's default CUDA-enabled torch build — several hundred MB of
NVIDIA runtime libraries (cuBLAS, cuDNN, cuFFT, NCCL, ...) that go unused
on every machine, CPU-only or otherwise, since GPU-specific optimization
is explicitly out of scope for this project.

A PyPI release (`pip install picoface`, no git clone needed) is planned as
the primary long-term distribution channel, but isn't published yet — the
git-clone-or-zip path above is what works today. See
[`RELEASING.md`](RELEASING.md) for the maintainer-side release process.
