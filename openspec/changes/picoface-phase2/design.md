## Context

Phase 0 delivered the module skeleton (`classifier.py` exists but is empty); Phase 1 delivered `load_dataset()`, the frozen `Dataset` dataclass, and the internal `make_stub_dataset()` generator that Phases 2–4 build and test against. Phase 2 is the first phase to write model code, the first to actually import `torch`, and the first to exercise the project's core promise: a student assembles and trains a real network using only named function calls, no `nn.Module` authoring, no hand-written training loop.

Arm 1 (classifier) has no dependency on Arm 2 (generator) — the roadmap sequences them 2-then-3 for narrative reasons (classifier is the simpler arm), not a technical one. This design covers only the classifier arm's plumbing, proven against the Phase 1 stub dataset; real content (shape taxonomy, resolution) stays deferred to Phase 5.

## Goals / Non-Goals

**Goals:**
- Resolve concrete signatures for `build_classifier()`, `train()`, `evaluate()`, `predict()` that a first-year non-CS student can call without reading `_internals`.
- Pick a CNN architecture and hyperparameter defaults that reliably train on the stub dataset within a CPU time budget of minutes, not just "eventually converges."
- Establish the internals layout for classifier model/training code, consistent with the Phase 0/1 public/`_internals` boundary.
- Start measuring and logging wall-clock training time now (per the roadmap's "measure from Phase 2 onward" mitigation for the CPU-budget risk), even though the real accuracy/speed bar isn't set until Phase 6.
- Give Phase 4 (linkage) a trained-model object it can consume directly (in-memory, no serialization).

**Non-Goals:**
- Real shape taxonomy, resolution, or color depth — deferred to Phase 5; this phase only needs the stub dataset's arbitrary shape to work.
- Any generator/autoencoder/VAE code — that's Phase 3, and shares no model code with the classifier (though it will reuse the same `_internals` boundary convention).
- Model save/load or serialization — out of scope for the MVP per the roadmap; Phase 4's linkage consumes in-memory model objects from the same session.
- Setting a final accuracy/quality bar — that's explicitly a Phase 6 decision, measured against the real dataset, not the stub.
- Hyperparameter tuning UI or student-facing knobs beyond simple keyword arguments (e.g. no config files, no CLI).

## Decisions

### 1. `build_classifier()` signature: `build_classifier(num_classes, input_shape=None)`
Takes `num_classes` explicitly (mirrors the `shape-classifier` draft spec's scenario) and an optional `input_shape=(H, W, C)`. If a student instead has a `Dataset` in hand, they pass `num_classes=len(data.class_names)` and `input_shape=data.images.shape[1:]` — both trivially derivable one-liners, avoiding a hidden dependency of `build_classifier()` on the `Dataset` type itself (keeps `classifier.py` decoupled from `datasets.py`, matching how Phase 1 kept `Dataset` framework-agnostic). Returns an opaque model handle (a thin wrapper or the internal `nn.Module` itself, typed as `PicoClassifier` for readability in student code, but never subclassed or extended by students).

**Alternatives considered:**
- **`build_classifier(data)` taking the `Dataset` directly** — rejected: forces every caller to already have a full dataset in memory just to construct a model shape, and blurs the data-contract/classifier boundary for no real benefit.
- **Returning the raw `torch.nn.Module`** — rejected: leaks a framework type as the *primary* student-facing handle, inviting students to call `.forward()` or inspect layers directly, defeating the internals boundary's intent even if the class technically lives in `_internals`.

### 2. Fixed, small CNN architecture — no architecture search or config
Two conv blocks (conv → ReLU → maxpool) sized relative to `input_shape`, followed by a small fully-connected head to `num_classes`. No student-facing knobs for depth/width/kernel size for MVP — matches "no layer definitions" from the roadmap's goals. Channel counts and kernel sizes are fixed constants tuned during implementation against the stub dataset's CPU training time, not derived from a formula.

**Alternatives considered:**
- **Configurable depth/width via `build_classifier(..., hidden_channels=...)`** — deferred, not rejected outright: adds surface area and a way for a student to accidentally build something too slow for the CPU budget; can be added later as an opt-in advanced parameter without breaking the simple call.
- **Global-average-pooling head instead of FC** — considered for input-size flexibility, deferred: adds one more concept without a forcing requirement yet, since `input_shape` is already passed explicitly.

### 3. `train()` signature: `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)`
Runs the full loop internally (batching via `torch.utils.data.DataLoader` built inside `_internals`, cross-entropy loss, Adam optimizer, CPU device fixed — no `device=` parameter for MVP since GPU support is a non-goal project-wide). Returns a `TrainingHistory` object (simple dataclass: `loss` per epoch, `wall_clock_seconds`) so students and Phase 2's own tests can inspect and plot progress without exposing tensors. Defaults are chosen so a first call with no tuning succeeds on the stub dataset; real-dataset defaults get revisited in Phase 6.

**Alternatives considered:**
- **Student passes a `torch.optim.Optimizer` instance** — rejected: exposes a framework concept the "no hand-written training loop" goal is explicitly trying to avoid.
- **`train()` mutates `model` in place and returns `None`** — rejected: returning `TrainingHistory` gives a free, natural hook for the viz helper (`plot_training_history(history)`) without a second call back into internals.

### 4. Wall-clock time logged, not asserted, in `TrainingHistory`
`TrainingHistory.wall_clock_seconds` records actual training time; Phase 2's test suite asserts this stays under a generous ceiling (e.g. under 60s for the stub dataset's default size) as a regression tripwire, not as the project's real speed bar — that's Phase 6, against real data. This directly implements the roadmap risk mitigation ("measure and log wall-clock training time against the stub dataset from Phase 2 onward").

### 5. `evaluate()` and `predict()` share the same internal forward-pass helper
`evaluate(model, data) -> float` (accuracy on the given `Dataset`) and `predict(model, image) -> str` (single image in, class name out — not a raw index, since a first-year student has no use for an unlabeled integer) both call one internal `_forward(model, images) -> logits` helper in `_internals`, avoiding duplicated preprocessing/normalization logic between the two.

**Alternatives considered:**
- **`predict()` returns the integer class index** — rejected: forces the student to separately consult `class_names` themselves for a result that's only ever used for display; `predict()` already has `model`'s associated class-name list available internally to resolve this for free.

### 6. Classifier internals live under `_internals/classifier_internals.py` (or a `_internals/classifier/` package if the module grows)
Contains the `nn.Module` subclass, the `DataLoader`/batching setup, the training-loop function, and the shared `_forward` helper. Follows the Phase 1 precedent (`_internals/stub_data.py`) of one internals module per concern rather than one monolithic `_internals/model.py` that Phase 3's generator internals would also have to share.

### 7. Viz helper: `plot_training_history(history)` in `viz.py`
One matplotlib helper for MVP — a loss-vs-epoch curve. Prediction-sample visualization (e.g. showing a grid of images with predicted vs. true labels) is left as a candidate follow-up inside this same phase's tasks if time permits, not a hard requirement, since the spec's core scenarios (build/train/evaluate/predict) don't depend on it.

## Risks / Trade-offs

- **Fixed architecture may not train well once real (more visually complex) content arrives in Phase 5/6.** → Explicitly flagged in the roadmap as a Phase 6 integration/tuning concern; this phase only needs to prove the plumbing and a reasonable default on synthetic data, not the final architecture.
- **Hardcoded hyperparameter defaults (epochs/batch_size/lr) are guesses tuned against a small synthetic stub, not real data.** → Acceptable per roadmap sequencing (real-content decisions deferred to Phase 5); Phase 6 is the designated place to revisit these defaults.
- **No `device=` parameter now could force a breaking API change later if GPU support is ever added.** → Accepted: GPU-specific optimization is a project-wide non-goal for MVP; if ever added, it can default to CPU and be purely additive.
- **`TrainingHistory`'s wall-clock assertion in tests is timing-sensitive and could flake on a slow CI machine.** → Set a generous ceiling (an order of magnitude above expected local time) so it functions as a regression tripwire, not a tight performance gate.

## Open Questions

None blocking implementation. Exact CNN channel/kernel constants and default hyperparameter values are left to be tuned during implementation against the stub dataset's CPU training time, not required to be finalized in this design.
