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

### 1. Two `build_classifier*()` entry points, both wrapping a shared internal builder
`build_classifier(data)` is the primary, student-facing entry point: it takes a `Dataset` directly and derives `num_classes`/`input_shape` internally, so a first-year student never has to learn attribute access (`data.class_names`) or array slicing (`data.images.shape[1:]`) just to build a model — both are extra concepts this project's stated goal (minimize conceptual load before the student is doing real ML) shouldn't require this early.

`build_classifier_from_shape(num_classes, input_shape)` is a secondary entry point for building a model without data in hand (e.g. before a dataset exists, or to construct several models against the same shape spec). Both `input_shape` and `num_classes` are required here — there is no `=None` default to reason about, since the two-function split removes any ambiguity about what "no shape given" means.

Both are thin one-line wrappers over a shared, non-public `_build_classifier(num_classes, input_shape)` in `_internals/classifier_internals.py`, so the actual construction logic exists exactly once. Both return the same opaque `PicoClassifier` model handle (a thin wrapper or the internal `nn.Module` itself, typed for readability in student code, but never subclassed or extended by students).

**This reverses part of the original Decision 1 rationale**, which rejected a `Dataset`-taking constructor specifically to keep `classifier.py` decoupled from `datasets.py`. That coupling is now accepted deliberately: for a first-year, non-CS audience, `build_classifier(data)` being a single, no-derivation call is worth more than the module-boundary purity, and `build_classifier_from_shape()` still exists as an explicit escape hatch for anyone who wants the decoupled, data-free path.

**Shape validation moves to build time, in two steps.** `_build_classifier` first checks `input_shape` against an analytically-computed minimum spatial size — derived from the same padding/kernel/stride constants the conv/pool layers are built from (see Decision 2) — and raises an explicit `ShapeError` (Decision 8) naming that minimum if the given `input_shape` is at or below it. Once past that check, it determines the FC head's exact flattened input size by running a single dummy tensor (`torch.zeros(1, C, H, W)`) through the conv/pool stack, rather than deriving the flatten size via a hand-maintained formula — this stays correct regardless of future kernel/padding/pool-count tweaks. Together these replace a previously-silent failure mode (an `input_shape` too small for the architecture would otherwise crash deep inside a later `train()` call with an opaque PyTorch error) with an immediate, specific `ShapeError` raised at the point the student calls `build_classifier()`/`build_classifier_from_shape()`.

**Data/model consistency is checked at use time, split across two places.** The *image-shape* half — does this image/batch match `model.input_shape`? — lives inside the shared `_forward` helper (Decision 5), since `_forward` already receives the raw image tensor on every path (`train()`'s per-batch loop, `evaluate()`'s full-dataset pass, `predict()`'s single image) and already has `model` in scope. This makes the check automatic on all three without being reimplemented three times, mirroring `_forward`'s existing rationale for centralizing preprocessing. The *class-count* half — does `len(data.class_names)` match `model.num_classes`? — only applies to `train()` and `evaluate()`, which receive a full `Dataset`; `predict()` never receives an explicit class count to compare against (it only returns a class name resolved from the model's own stored list), so no class-count check applies there. `train()`/`evaluate()` run this check once, before their main loop starts, rather than per-batch/per-image. Both halves raise `ShapeError` (Decision 8) on mismatch, rather than failing inside a matmul.

**Alternatives considered:**
- **Single `build_classifier()` with `data`/`num_classes`/`input_shape` all optional, mode inferred from which were passed** — rejected: creates an ambiguous case (what happens if `data` and `num_classes`/`input_shape` are both given?) with no clean answer, and makes one signature's meaning depend on call site. Two separately-named, precisely-typed functions make each call's contract unambiguous by construction.
- **`build_classifier(data)` as the only entry point, no shape-only path** — rejected: removes the ability to construct a model before a dataset exists, or to build multiple models against one shape spec without holding a full `Dataset` in memory.
- **Returning the raw `torch.nn.Module`** — rejected: leaks a framework type as the *primary* student-facing handle, inviting students to call `.forward()` or inspect layers directly, defeating the internals boundary's intent even if the class technically lives in `_internals`.

### 2. Fixed, small CNN architecture — no architecture search or config
Two conv blocks (conv → ReLU → maxpool) sized relative to `input_shape`, followed by a small fully-connected head to `num_classes`. No student-facing knobs for depth/width/kernel size for MVP — matches "no layer definitions" from the roadmap's goals. Channel counts and kernel sizes are fixed constants tuned during implementation against the stub dataset's CPU training time, not derived from a formula.

Each conv uses `padding=1` as the starting default, chosen so a 3×3 kernel preserves H/W and only the maxpools shrink the spatial size — this keeps the minimum-viable-input-size calculation simple and leaves headroom before hitting degenerate feature maps. Like the channel/kernel constants, `padding` is a valid hyperparameter to tune later against real data (Phase 5/6), not a permanent commitment.

Kernel size, padding, and stride for every conv/pool layer are defined once as constants in `_internals/classifier_internals.py`, not duplicated between the layer-construction code and any validation code. An internal helper computes the smallest `input_shape` that survives the full conv/pool stack analytically from those same constants — for the current padding=1/kernel=3 conv plus two stride-2 maxpools, that floor is 4×4 (two halvings: 4 → 2 → 1) — and `_build_classifier` (Decision 1) raises an explicit `ShapeError` (Decision 8) naming that minimum if a smaller `input_shape` is given, before attempting the dummy pass-through that determines the FC head's exact flatten dimension.

**Alternatives considered:**
- **Configurable depth/width via `build_classifier(..., hidden_channels=...)`** — deferred, not rejected outright: adds surface area and a way for a student to accidentally build something too slow for the CPU budget; can be added later as an opt-in advanced parameter without breaking the simple call.
- **Global-average-pooling head instead of FC** — considered for input-size flexibility, deferred: adds one more concept without a forcing requirement yet, since `input_shape` is already passed explicitly.

### 3. `train()` signature: `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)`
Runs the full loop internally (batching via `torch.utils.data.DataLoader` built inside `_internals`, cross-entropy loss, Adam optimizer, CPU device fixed — no `device=` parameter for MVP since GPU support is a non-goal project-wide). Returns a `TrainingHistory` object (simple dataclass: `loss` per epoch, `wall_clock_seconds`) so students and Phase 2's own tests can inspect and plot progress without exposing tensors. Defaults are chosen so a first call with no tuning succeeds on the stub dataset; real-dataset defaults get revisited in Phase 6. Before the training loop starts, `train()` validates `len(data.class_names) == model.num_classes`, raising `ShapeError` (Decision 1/8) on mismatch; per-batch image-shape validation happens automatically inside the shared `_forward` helper (Decision 5) as `DataLoader` batches flow through it, so `train()` itself never touches preprocessing, framework tensor code, or per-image shape checks directly.

**Alternatives considered:**
- **Student passes a `torch.optim.Optimizer` instance** — rejected: exposes a framework concept the "no hand-written training loop" goal is explicitly trying to avoid.
- **`train()` mutates `model` in place and returns `None`** — rejected: returning `TrainingHistory` gives a free, natural hook for the viz helper (`plot_training_history(history)`) without a second call back into internals.

### 4. Wall-clock time logged, not asserted, in `TrainingHistory`
`TrainingHistory.wall_clock_seconds` records actual training time; Phase 2's test suite asserts this stays under a generous ceiling (e.g. under 60s for the stub dataset's default size) as a regression tripwire, not as the project's real speed bar — that's Phase 6, against real data. This directly implements the roadmap risk mitigation ("measure and log wall-clock training time against the stub dataset from Phase 2 onward").

### 5. `train()`, `evaluate()`, and `predict()` share the same internal forward-pass helper
`train()`'s per-batch loop, `evaluate(model, data) -> float` (accuracy on the given `Dataset`), and `predict(model, image) -> str` (single image in, class name out — not a raw index, since a first-year student has no use for an unlabeled integer) all call one internal `_forward(model, images) -> logits` helper in `_internals`, avoiding duplicated preprocessing/normalization/shape-validation logic across all three.

`_forward` is also where the one preprocessing step every path needs actually happens: `Dataset.images` is stored as `uint8` `NHWC` (per `datasets.py`'s on-disk/in-memory contract), but PyTorch conv layers expect `float` `NCHW`. `_forward` permutes to channels-first, casts to `float32`, and normalizes pixel values to `[0, 1]` via `/255.0`, then runs the model. Centralizing this in `_forward` — rather than once in the `DataLoader` for training and separately in `evaluate()`/`predict()` — guarantees training-time and inference-time preprocessing can never drift apart, a common source of silent accuracy bugs in real ML code (train/serve skew).

`_forward` also performs the image-shape half of the data/model consistency check (Decision 1): before normalizing, it compares the incoming image shape to `model.input_shape` and raises `ShapeError` (Decision 8) on mismatch. Because every path funnels through `_forward`, this check can't be skipped or reimplemented inconsistently by any one of the three public functions.

**Alternatives considered:**
- **`predict()` returns the integer class index** — rejected: forces the student to separately consult `class_names` themselves for a result that's only ever used for display; `predict()` already has `model`'s associated class-name list available internally to resolve this for free.
- **Per-channel mean/std standardization instead of `/255.0` min-max scaling** — deferred, not rejected: often trains marginally better, but requires computing and storing dataset statistics somewhere, adding state and complexity beyond what MVP's simple-normalization needs justify; worth revisiting in Phase 6 alongside other hyperparameter/preprocessing tuning against real data.

### 6. Classifier internals live under `_internals/classifier_internals.py` (or a `_internals/classifier/` package if the module grows)
Contains the `nn.Module` subclass, the `_build_classifier` construction helper (including the dummy-pass-through shape validation), the `DataLoader`/batching setup, the training-loop function, and the shared `_forward` helper. Follows the Phase 1 precedent (`_internals/stub_data.py`) of one internals module per concern rather than one monolithic `_internals/model.py` that Phase 3's generator internals would also have to share.

### 7. Viz helper: `plot_training_history(history)` in `viz.py`
One matplotlib helper for MVP — a loss-vs-epoch curve. Prediction-sample visualization (e.g. showing a grid of images with predicted vs. true labels) is left as a candidate follow-up inside this same phase's tasks if time permits, not a hard requirement, since the spec's core scenarios (build/train/evaluate/predict) don't depend on it.

### 8. Project-specific `ShapeError` exception for all shape/class-count validation failures
All validation failure points introduced by Decisions 1, 2, and 5 — build-time minimum-input-size, build-time dummy-pass-through backstop, use-time image-shape mismatch, and use-time class-count mismatch — raise a single public `picoface.classifier.ShapeError`, not a generic `ValueError`. It subclasses `ValueError` (so `except ValueError` still works for anyone who doesn't know the specific type), while `except ShapeError` lets a student catch shape/class-count problems specifically. Each call site supplies its own message tailored to what a first-year student needs to fix (e.g. naming the minimum viable input size, or naming which of shape/class-count is mismatched and what the model actually expects), rather than surfacing PyTorch's own error text. It lives in `classifier.py`, not `_internals`, so it's importable/catchable by students — consistent with the public/`_internals` boundary.

**Alternatives considered:**
- **Raising plain `ValueError`/`RuntimeError` at each site** — rejected per explicit direction: generic built-in exceptions don't let message wording be centralized or tuned for a first-year audience, and give students no specific type to catch.
- **A small exception hierarchy** (e.g. separate `MinimumSizeError`, `ShapeMismatchError`, `ClassCountError`) — deferred, not rejected: MVP's four failure points are distinguishable by message text alone; more granular types can be added later as subclasses of `ShapeError` without breaking existing `except ShapeError` handlers.

## Risks / Trade-offs

- **Fixed architecture may not train well once real (more visually complex) content arrives in Phase 5/6.** → Explicitly flagged in the roadmap as a Phase 6 integration/tuning concern; this phase only needs to prove the plumbing and a reasonable default on synthetic data, not the final architecture.
- **Hardcoded hyperparameter defaults (epochs/batch_size/lr) are guesses tuned against a small synthetic stub, not real data.** → Acceptable per roadmap sequencing (real-content decisions deferred to Phase 5); Phase 6 is the designated place to revisit these defaults.
- **No `device=` parameter now could force a breaking API change later if GPU support is ever added.** → Accepted: GPU-specific optimization is a project-wide non-goal for MVP; if ever added, it can default to CPU and be purely additive.
- **`TrainingHistory`'s wall-clock assertion in tests is timing-sensitive and could flake on a slow CI machine.** → Set a generous ceiling (an order of magnitude above expected local time) so it functions as a regression tripwire, not a tight performance gate.

## Open Questions

None blocking implementation. Exact CNN channel/kernel constants and default hyperparameter values are left to be tuned during implementation against the stub dataset's CPU training time, not required to be finalized in this design.
