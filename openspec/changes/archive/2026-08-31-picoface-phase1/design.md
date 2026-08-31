## Context

Phase 0 delivered an installable but empty package skeleton (`src/picoface/{datasets,classifier,generator,linkage,viz}.py`, `_internals/`). Phase 1 is the first phase to write real logic, and it sits on the critical path for every later phase: Phases 2–4 (classifier, generator, linkage) all need a dataset object to build and test against, and none of them can start meaningfully until `load_dataset()` and a stand-in dataset exist. Real content (shape taxonomy, resolution, noise policy) is deliberately not decided until Phase 5 (Dataset Forge); this phase must prove the plumbing works using only disposable synthetic data.

This design was worked out in an `/opsx:explore` session prior to this proposal. Three questions drove it: (1) what value type does `load_dataset()` return, (2) how does the "stub dataset" get produced and does it belong in the public API, (3) does Phase 1 split into more than one change. The session concluded: single change, frozen dataclass return type, stub generator is internal-only, and the stub proves the file-format parser once so downstream phases don't have to.

## Goals / Non-Goals

**Goals:**
- Establish `load_dataset()` as the one public, stable entry point for reading a dataset bundle from disk.
- Establish a single shared dataset value type used by both `load_dataset()` and internal test/dev tooling, so Phases 2–4 never need to special-case where their data came from.
- Prove the on-disk `.npz`/`classes.json` format is genuinely size- and class-agnostic, not just in principle.
- Give Phases 2–4 a cheap, in-memory way to get a plumbing-test dataset without needing real content or file I/O.
- Introduce a test-suite convention (`pytest`, `tests/`) for the project, since this is the first phase with logic worth testing.

**Non-Goals:**
- Deciding the real shape taxonomy, resolution, color depth, or noise/augmentation policy — deferred to Phase 5 (Dataset Forge).
- Any framework-specific (torch) data types or `DataLoader` integration — that belongs to Phase 2/3's `train()` implementations, inside `_internals`, not to the data-contract layer.
- A public/documented stub-dataset API — no student ever sees pre-Phase-6 code, so there is no audience for one.
- Dataset serialization/model save-load — out of scope for this phase and for the MVP generally (see roadmap non-goals).

## Decisions

### 1. Shared dataset value type: frozen `@dataclass`, numpy-only
Fields: `images: np.ndarray` (uint8, N×H×W×C), `labels: np.ndarray` (int, N), `class_names: list[str]`. Frozen (immutable) to prevent accidental in-place mutation of loaded data. Custom `__repr__` shows shapes/dtypes (e.g. `Dataset(images: uint8[120,16,16,3], labels: int[120], classes=['a','b','c'])`) instead of the dataclass default, which would print full array contents.

**Alternatives considered:**
- **Plain tuple** — rejected: positional-only, no named access, doesn't meet the spec's "ready-to-use dataset object" bar.
- **`NamedTuple`** — rejected: its free positional unpacking (`images, labels, classes = load_dataset(...)`) invites order-swap bugs for a first-time-Python audience, and it's a less natural home for future convenience methods (`len(ds)`, `ds.num_classes`) than a dataclass.
- **Wrapping `torch.utils.data.Dataset`** — rejected: leaks a framework-specific type into a layer that must stay framework-agnostic. Dataset Forge never imports torch; the choice of how Phase 2/3's `train()` consumes data (framework, `DataLoader` or otherwise) isn't this phase's decision to make.

### 2. `load_dataset()` and the stub generator return the same type
Both the public loader and the internal stub generator produce the dataclass from Decision 1. This means Phases 2–4 can write plumbing that accepts "a dataset value" without caring whether it came from a real file or synthetic in-memory generation — no adapter layer needed later.

### 3. Stub dataset generator is internal, non-public
Lives at `src/picoface/_internals/stub_data.py` (e.g. `make_stub_dataset(...)`), consistent with the existing `_internals` boundary from Phase 0 (implementation the student never touches). Not exported from the package's public surface, not documented as an API. Rationale: the project isn't given to students until Phase 5/6 delivers the real dataset, so there is no scenario where a student would call this — "built-in" (per the roadmap's phrasing) means "importable by this project's own development and test code across Phases 2–4," not "part of the student-facing API."

### 4. Stub generation is in-memory; file-format proof happens once, in this phase
The stub generator builds its dataclass output directly in memory — no disk I/O — so Phases 2–4's test suites can call it cheaply and repeatedly. Phase 1's own test suite additionally performs one explicit round-trip: write stub-generated data to a real `.npz` + `classes.json`, then read it back through `load_dataset()`, asserting the values match. This is what actually exercises and proves the "format is size- and class-agnostic" requirement (by doing this with at least two differently-shaped bundles). Downstream phases reuse the in-memory generator directly and don't need to repeat the file-format proof themselves.

### 5. Introduce `pytest` and a `tests/` convention now
No test framework exists in the repo yet. This is the first phase with real logic to verify (format parsing, shape-agnosticism, round-tripping), so `pytest` is added as a dev-only dependency (not a runtime dependency — must not leak into a student's `pip install picoface`) and `tests/` is established as the project's test-suite location, to be reused by Phases 2–4.

## Risks / Trade-offs

- **Internal-only stub generator could get reinvented or copy-pasted by a future phase instead of imported.** → Mitigate by naming and location convention (`_internals/stub_data.py`) documented here and in the module itself; Phase 2's design should explicitly reference importing it rather than rewriting it.
- **Frozen dataclass with numpy array fields has non-trivial equality semantics** (dataclass `__eq__` will call `==` on numpy arrays, which returns an elementwise array, not a bool, and will raise on truthiness checks). → Implementation must override `__eq__` (or document that equality isn't meaningful/supported) rather than rely on the dataclass default; the round-trip test in Decision 4 should compare fields explicitly (e.g. `np.array_equal`) rather than `==` on the whole object.
- **`pytest` as a dev dependency needs a clear separation mechanism** (e.g. `[dependency-groups]` / optional-extras in `pyproject.toml`) so it never gets pulled into a plain `pip install picoface`. → Verify with a clean-install check similar to Phase 0's `pip install -e .` verification step.

## Open Questions

None blocking implementation. Convenience additions to the dataclass beyond the three core fields (e.g. `__len__`, `num_classes` property) are left to be added opportunistically during implementation or by whichever later phase first needs them — not required by this phase's spec.
