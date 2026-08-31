## Why

Phase 1 delivered `load_dataset()` and an in-memory stub-dataset generator, but nothing in `picoface` can yet build, train, or run a model against them — `src/picoface/classifier.py` is an empty stub. Arm 1 (the student-facing classifier) is the simplest of the two student arms and has no dependency on Arm 2, so it can be built and proven against the Phase 1 stub dataset now, resolving the student-facing function signatures this project's later phases (linkage, end-to-end tuning) will build on.

## What Changes

- Add `build_classifier(data)` (plus a `build_classifier_from_shape(num_classes, input_shape)` variant for when no `Dataset` is yet available) to `src/picoface/classifier.py`: constructs a small CNN sized to the dataset's (or explicitly given) class count and image shape, returning a trainable model object, with no `nn.Module` authoring exposed to the caller. Both validate their input shape against the architecture's minimum viable size at build time.
- Add `train(model, data)`: runs the full training loop (forward/loss/backward/optimizer step/epoch iteration) against a `Dataset` (real or stub), returning a training-history object. Logs wall-clock training time to start giving early visibility into the "seconds-to-minutes on old CPU" budget (per roadmap risk, measured from Phase 2 onward).
- Add `evaluate(model, data)`: returns an accuracy metric on held-out data.
- Add `predict(model, image)`: returns a class label for a single new image.
- Add viz helper(s) in `src/picoface/viz.py` for the classifier arm (e.g. plotting training-history curves and/or a sample of predictions) — student-facing, no internals exposed.
- All `nn.Module` subclasses, loss functions, and training-loop code live in `src/picoface/_internals/` (new submodule for the classifier's internals), consistent with the Phase 0 public/`_internals` boundary; `classifier.py` and `viz.py` expose only the named entry-point functions.
- Add a test suite (`tests/test_classifier.py`) proving `build_classifier()` → `train()` → `evaluate()`/`predict()` against the Phase 1 stub dataset (via `make_stub_dataset()`), including a wall-clock training-time assertion under the CPU budget.
- Add `torch` as an exercised runtime dependency for the first time (already declared in `pyproject.toml` since Phase 0, but unused until now).

## Capabilities

### New Capabilities
- `shape-classifier`: the student-assembled classification arm — building, training, evaluating, and running inference with a small CNN classifier via simple function calls, within an old-CPU time budget.

### Modified Capabilities
(none — `data-contract` and `packaging` are consumed as-is, not changed)

## Impact

- **Code**: `src/picoface/classifier.py` (new implementation), `src/picoface/viz.py` (new classifier-facing helpers), `src/picoface/_internals/` (new submodule(s) for CNN model definition and training-loop code).
- **Dependencies**: none new — `torch`, `numpy`, `matplotlib` already declared in `pyproject.toml`; this is the first phase to actually import `torch`.
- **Tests**: adds `tests/test_classifier.py`, reusing `_internals/stub_data.make_stub_dataset()` from Phase 1 rather than generating its own fixtures.
- **Downstream**: unblocks Phase 4 (linkage, which needs a trained classifier to score generator output and to drive activation-maximization) and Phase 6 (end-to-end integration, which swaps the stub dataset for the real one against this same API).
