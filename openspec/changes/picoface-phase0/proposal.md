## Why

`picoface` is a new pedagogical Python library (full project rationale, architecture, and phase plan in [`openspec/ROADMAP.md`](../../ROADMAP.md)). Before any of its capabilities can be built, the project needs a real, installable package skeleton: a `pyproject.toml`, the module layout the later phases will fill in, and Dataset Forge kept structurally separate from day one. This change is Phase 0 of that roadmap — scaffolding only, no capability logic.

## What Changes

- Introduce `picoface`, a new pip-installable Python package (import name matches distribution name, matches the GitHub repo `jporterdal/picoface`).
- Create the package skeleton (`src/picoface/{__init__.py, datasets.py, classifier.py, generator.py, linkage.py, viz.py, _internals/}`) with empty modules — no implementation yet, just the shape later phases build into.
- Create a top-level `dataset_forge/` directory, separate from the installable package, with its own dependency manifest.
- Add a root `.gitignore`, README stub, and license.

## Capabilities

### New Capabilities

- `packaging`: The installable-package structure and naming (`picoface` repo, import name, and PyPI distribution name aligned) that supports pip/zip/Colab-git-clone distribution without committing to one channel. This change fully satisfies `packaging`'s structural requirements (naming consistency, `pyproject.toml` + src layout, Dataset Forge excluded from the installable package); only *choosing and documenting* the actual distribution channel is left for Phase 7.

### Modified Capabilities

None — this is a new project with no existing specs.

## Impact

- New repository content only; no existing code or specs are affected (greenfield project, no commits yet).
- Establishes the physical module boundaries (`_internals/` vs. public modules) that Phases 1–4 will fill in, without implementing any logic inside them yet.
- Dependencies: PyTorch (CPU), numpy, matplotlib; packaging via `pyproject.toml` with a src layout.

See [`openspec/ROADMAP.md`](../../ROADMAP.md) for the full 8-phase plan, all six capabilities, and the design decisions that apply project-wide.
