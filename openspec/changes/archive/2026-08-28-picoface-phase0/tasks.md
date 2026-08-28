This change is scoped to **Phase 0 — Scaffolding** only. The full 8-phase project plan lives in [`openspec/ROADMAP.md`](../../ROADMAP.md); Phases 1–7 will each become their own OpenSpec change when they're ready to start.

## 1. Phase 0 — Scaffolding

- [x] 1.1 Initialize `pyproject.toml` with a src layout for the `picoface` package (name, build backend, PyTorch/numpy/matplotlib dependencies)
- [x] 1.2 Add a root `.gitignore` covering common Python/dev-environment artifacts (venvs, `__pycache__/`, `*.egg-info/`, build/dist output, notebook checkpoints, etc.)
- [x] 1.3 Create the package skeleton: `src/picoface/{__init__.py, datasets.py, classifier.py, generator.py, linkage.py, viz.py, _internals/__init__.py}`
- [x] 1.4 Create a top-level `dataset_forge/` directory, separate from the installable package, with its own dependency manifest
- [x] 1.5 Add a README stub and license
- [x] 1.6 Verify `pip install -e .` succeeds and `import picoface` works against the empty module skeleton — **depends on 1.5**: `pyproject.toml` declares `readme = "README.md"` and `license = { file = "LICENSE" }`, so this step fails until 1.5 creates those files. Use the existing `venv/` at the repo root for this verification rather than creating a new one.
