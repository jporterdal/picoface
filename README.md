# picoface

A small, pip-installable, PyTorch-backed Python library for building and
training real image classifiers and generators from simple, well-named
function calls — designed for first-year non-CS students to train on an
ordinary laptop CPU in seconds to minutes.

Status: early scaffolding (Phase 0). See
[`openspec/ROADMAP.md`](openspec/ROADMAP.md) for the full project plan.

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

## Layout

- `src/picoface/` — the installable student-facing package (classifier,
  generator, linkage, viz) plus `_internals/` for implementation details.
- `dataset_forge/` — instructor-only, unrestricted dataset generation
  tooling, kept outside the installable package.
