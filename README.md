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

# Arm 1: an independent classifier, later used as the capstone's judge.
classifier_model = build_classifier(data)
train(classifier_model, data)
print(evaluate(classifier_model, data))           # accuracy on `data`
print(predict(classifier_model, data.images[0]))  # predicted class name

# Arm 2: the student's model -- one model that both classifies and generates.
vae_model = build_vae(data)
train(vae_model, data)          # same train(), works on either model
print(evaluate(vae_model, data))
new_images = generate(vae_model, 8)  # sample 8 new images

# Capstone: tie the two arms together.
report = classify_generated(classifier_model, vae_model, data)
print(report.overall)  # fraction of generated images the classifier agrees with

am_image = activation_maximize(classifier_model, data.class_names[0])
```

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
N×H×W×C, `labels`: int N) plus a companion `classes.json` in the same
directory mapping label index to class name. This is a fixed contract, not a
guess — any `.npz`/`classes.json` pair matching it will load.

**picoface does not provide a real dataset itself.** Producing one is a
separate, instructor-side step (the `dataset_forge/` tool in this repo, not
part of the installable package) that happens before students receive
anything. Getting a real dataset into students' hands is explicitly outside
this library's scope.

## What to expect

These are documented, informational ranges from the reference implementation
running against a real generated dataset (7 shape classes, 2,000 images per
class) — **nothing in this repo enforces them**. Your own numbers will vary
with your dataset, seeds, and hyperparameters.

| Metric | Typical range |
|---|---|
| CNN (`build_classifier`) held-out accuracy | ~0.98–1.00 |
| VAE (`build_vae`) held-out accuracy | ~0.95–1.00 |
| Reconstruction agreement, most classes | ~0.8–1.0 |
| `classify_generated()` overall agreement | ~0.81 |

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
