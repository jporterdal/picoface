## 1. Test infrastructure

- [ ] 1.1 Add `pytest` as a dev-only dependency in `pyproject.toml` (e.g. `[dependency-groups]` or `[project.optional-dependencies]`), not a runtime dependency
- [ ] 1.2 Create the `tests/` directory at the repo root as the project's test-suite convention

## 2. Shared dataset value type

- [ ] 2.1 Implement a frozen `@dataclass` in `src/picoface/datasets.py` with fields `images` (uint8 ndarray, N×H×W×C), `labels` (int ndarray, N), `class_names` (list[str])
- [ ] 2.2 Implement a custom `__repr__` showing shapes/dtypes (e.g. `Dataset(images: uint8[120,16,16,3], labels: int[120], classes=['a','b','c'])`) instead of the dataclass default
- [ ] 2.3 Implement explicit `__eq__` using numpy-safe comparison (e.g. `np.array_equal` on array fields) rather than relying on the dataclass default, which breaks on array fields

## 3. load_dataset() entry point

- [ ] 3.1 Implement `.npz` (`images`, `labels`) + `classes.json` parsing in `load_dataset(path)` in `src/picoface/datasets.py`
- [ ] 3.2 Ensure H, W, C, and number of classes are inferred from the loaded arrays/JSON at call time, never hardcoded
- [ ] 3.3 Return the shared dataclass type from Task 2 — **depends on 2.1**

## 4. Internal synthetic stub dataset generator

- [ ] 4.1 Create `src/picoface/_internals/stub_data.py` with a `make_stub_dataset(...)` function generating a small, arbitrary-dimension, in-memory dataset with at least two fake classes
- [ ] 4.2 Ensure `make_stub_dataset(...)` returns the same shared dataclass type as `load_dataset()` — **depends on 2.1**
- [ ] 4.3 Do not export `stub_data` or `make_stub_dataset` from the package's public surface (`src/picoface/__init__.py` stays untouched by this task)

## 5. Tests

- [ ] 5.1 Round-trip test: write `make_stub_dataset(...)` output to a real `.npz` + `classes.json`, read it back with `load_dataset()`, assert fields match (via `np.array_equal`) — **depends on 3.3, 4.2**
- [ ] 5.2 Size/class-agnostic test: generate two `make_stub_dataset(...)` outputs with different shapes/class counts, round-trip both through `.npz`/`load_dataset()`, assert both load correctly with no code changes — **depends on 5.1**
- [ ] 5.3 Repr test: assert a loaded dataset's `repr()` shows shapes/dtypes and does not contain raw pixel values
- [ ] 5.4 Interchangeability test: assert `load_dataset()` output and `make_stub_dataset()` output are the same type and expose the same fields, so downstream code can treat them identically

## 6. Verification and cleanup

- [ ] 6.1 Verify `pip install -e .` still succeeds and `import picoface` works without pulling in `pytest` as a runtime dependency
- [ ] 6.2 Run the full test suite (`pytest`) and confirm all tests in Task 5 pass
- [ ] 6.3 Empty `openspec/draft-specs/data-contract/` now that its content has been absorbed into this change's `specs/data-contract/spec.md`, per the convention noted in `openspec/ROADMAP.md`
