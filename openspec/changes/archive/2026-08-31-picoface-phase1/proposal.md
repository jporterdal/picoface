## Why

Phase 0 delivered an installable but empty package skeleton — `datasets.py` has no code, and nothing can read or represent an image dataset yet. Every downstream arm (classifier, generator, linkage) needs a stable, shared way to load a dataset before it can be built or tested, and none of them can start meaningfully until that contract exists. Phase 1 establishes that contract now, against disposable synthetic data, so the real dataset (Phase 5) and the arms that consume it (Phases 2–4) can be developed independently and in either order.

## What Changes

- Define the dataset interchange format: an `.npz` bundle (`images`: uint8 array, N×H×W×C; `labels`: integer array, N) plus a companion `classes.json` mapping label index → class name. Format is size- and class-count-agnostic — H, W, C, and number of classes are inferred at load time, never hardcoded.
- Add `load_dataset(path)` as the sole public entry point in `src/picoface/datasets.py`, hiding all file parsing behind one function call.
- Introduce a shared, frozen `@dataclass` return type (numpy-only fields: `images`, `labels`, `class_names`) with a custom `__repr__` (shapes/dtypes, not raw array contents) — returned by both `load_dataset()` and the new stub generator, so downstream code treats real and synthetic data identically.
- Add an internal, non-public synthetic stub-dataset generator (`_internals/stub_data.py`) producing a small, arbitrary-shaped, ≥2-fake-class dataset in memory, for Phases 2–4's own test suites to build and prove plumbing against before real content exists. Not part of the public API — no student sees pre-Phase-6 code.
- Introduce `pytest` as a dev dependency and a `tests/` convention (first phase with real logic to verify). Phase 1's own tests round-trip stub-generated data through a real `.npz`/`classes.json` write and `load_dataset()` read, to prove the file-format parser satisfies the size/class-agnostic requirement — downstream phases reuse the in-memory generator directly and don't need to repeat that proof.

## Capabilities

### New Capabilities
- `data-contract`: the shared image-dataset interchange format (`.npz` + `classes.json` schema), the `load_dataset()` entry point, the shared dataset return type, and the internal synthetic stub dataset used to validate downstream plumbing before real content exists.

### Modified Capabilities
(none — `packaging` from Phase 0 is unaffected; this change only adds new code under the existing package structure)

## Impact

- **Code**: `src/picoface/datasets.py` (new implementation), `src/picoface/_internals/stub_data.py` (new), `src/picoface/_internals/__init__.py` (unchanged interface, new submodule).
- **Dependencies**: adds `pytest` as a dev-only dependency in `pyproject.toml` (`[project.optional-dependencies]` or `[dependency-groups]`); no new runtime dependencies (numpy is already required).
- **Tests**: introduces `tests/` at the repo root — first test suite in the project.
- **Downstream**: unblocks Phase 2 (classifier), Phase 3 (generator), and Phase 4 (linkage), all of which will build and test against `load_dataset()` and the stub generator introduced here.
