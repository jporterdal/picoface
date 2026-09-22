## Context

Phase 3c made the VAE's latent space supervised: the classification head reads the reparameterized sample `z` during training and `mu` at inference, so a class's labeled images cluster in a region of latent space the model itself learned (`openspec/changes/archive/2026-09-22-picoface-phase3c/design.md`, Decision 1). That change's design doc explicitly parked conditional generation as a Non-Goal, but named the mechanism this design uses: "The supervised latent makes it cheap to add later by sampling a class's empirical cluster." This is that "later" — scoped narrowly to what the capstone needs, not as a public `generate(cls=...)` API change (see proposal.md and Decisions 1–2 below).

`_VAE` (`src/picoface/_internals/generator_internals.py`) already has `encode(x) -> (mu, logvar)`, `encode_mu(x)`, and a `decoder` submodule internally; `sample(n)` decodes from the prior `N(0, I)` only, with no way to decode an arbitrary `z`. None of this is reachable through the abstract `_Model` contract (`classify`, `sample` are the only declared capabilities), and `picoface.linkage` — like `picoface.classifier` and `picoface.generator` — is meant to work through that contract rather than reach into a specific model class's internals (see ROADMAP.md, "Public API / `_internals` boundary").

See proposal.md for the full motivation and scope; this document covers how `classify_generated()`'s class-targeted sampling is built and where it sits in the existing model-agnostic architecture.

## Goals / Non-Goals

**Goals:**
- Give `classify_generated()` a real "intended class" per generated image, using only the VAE's existing (already-trained, unchanged) encoder and decoder.
- Keep the new capability generic at the `_Model` level, consistent with how `classify`/`sample` already work, rather than special-casing `_VAE` inside `linkage.py`.
- Keep `activation_maximize()` usable on any classifying model, not hardcoded to the CNN, per the existing capability protocol.

**Non-Goals:**
- A public, student-facing `generate(model, n, class_name=...)` or similar conditional-generation verb on `picoface.generator`. That was explicitly deferred by Phase 3c and remains deferred; nothing here reopens it, since the mechanism is confined to `classify_generated()`'s internals (explicit decision from the `/opsx:explore` session that produced this change — see Decision 2).
- A label-conditioned decoder architecture (a decoder that takes a class embedding as input). That is a genuinely different, heavier approach the roadmap parks for Phase 6; this design changes no model architecture and requires no retraining.
- Any regularization (L2 on gradient norm, blur, etc.) on `activation_maximize()`'s ascent to keep it away from adversarial-looking noise. Explicitly deferred to a later phase, once real data exists, per user decision during exploration.
- Tuning what "empirical cluster" means beyond a simple, defensible statistic (Decision 1) — this is stub-dataset-proven plumbing, like every prior phase; real-data behavior is a Phase 6 concern.

## Decisions

### 1. Class-conditional sampling sources labeled real data's `mu`, not a stored per-class statistic on the model

`classify_generated(classifier_model, generator_model, data, n)` takes `data` — the same labeled `Dataset` the VAE was trained on — as an explicit argument, and computes each class's empirical latent cluster at call time: `encode_mu()` on `data`'s images of that class, then the mean of the resulting vectors (and their standard deviation, for sampling spread) per class.

This is preferred over having `train()` or `build_vae()` compute and cache per-class latent statistics on the model object, because:
- It needs no change to `train()`, `TrainingHistory`, or `build_vae()` — the model-agnostic training loop stays exactly as Phase 3b/3c left it.
- `data` is already sitting in the student's notebook right after training; passing it to `classify_generated()` costs nothing a student would notice, and matches the existing pattern where `evaluate(model, data)` also takes the dataset explicitly rather than the model remembering it.
- It keeps "what counts as a class's cluster" entirely inside `linkage.py`, where it can change (e.g. mean vs. a smarter estimate) without touching the model contract again.

**Alternatives considered:**
- **Cache per-class `mu` statistics on the model during `train()`** — rejected: couples the training loop to a capstone-only need, and goes stale if the model is retrained or evaluated against different data later in the same session.
- **Require the student to compute and pass in per-class statistics themselves** — rejected: reintroduces exactly the "glue code between the two models" the draft spec's original scenario explicitly says students shouldn't have to write.

### 2. `encode_mu()`/`decode()` are a new `_Model` capability, not `linkage.py` reaching into `_VAE`

`_Model` gains two capability-gated methods — `encode_mu(x) -> mu` and `decode(z) -> image`, both raising `NotImplementedError` by default — declared by a new `latent_access` capability that only `_VAE` sets (alongside its existing `classify`, `sample`). `linkage.py` calls these through the abstract contract, the same way `model_api.py`'s `generate()` calls `model.sample(n)` without knowing the concrete model class.

This mirrors exactly how `classify`/`sample` already work (Phase 3b's refactor) and keeps `linkage.py` from importing `_VAE` directly or using `isinstance` checks against a concrete internals class — which would violate the "arm modules expose only named entry-point functions; `_internals` details are not reached into from outside" boundary the project has held since Phase 0/3b.

**Alternatives considered:**
- **`linkage.py` imports `_VAE` from `generator_internals` and calls `.encode_mu()`/`.decoder(...)` directly** — rejected: works today, but ties `classify_generated()` to one concrete class, breaks silently if a future model type wants to support the same capstone trick, and crosses the internals boundary the rest of the project treats as load-bearing.
- **Fold `encode_mu`/`decode` into the existing `classify`/`sample` capabilities** (e.g. overload `sample()` to accept an optional `z`) — rejected: changes an existing capability's contract (`sample(n)`'s signature) rather than adding a new one, which is a larger and less obviously backward-compatible change for a feature only the capstone needs.

### 3. `classify_generated()` validates matching class taxonomies before doing any work

Before generating or classifying anything, `classify_generated()` checks `classifier_model.class_names == generator_model.class_names` (as sets, order-independent — see spec scenario) and raises a named error on mismatch. This follows the same pattern `_Model.check_data` already uses for image-shape/class-count mismatches (`ShapeError`-style, "clear error naming the mismatch, rather than failing inside internals" — ROADMAP.md's established convention since Phase 2).

Nothing else enforces this today: the CNN and VAE are trained independently, each against whatever `Dataset` was passed to their own `train()` call, and both simply store whatever `class_names` that dataset provided. A capstone comparing predictions across the two models needs them to mean the same 12 classes in the same way, or the report is meaningless in a way that's easy to miss rather than crash on.

**Alternatives considered:**
- **No validation; let a mismatch silently produce a confusing report** — rejected: contradicts the project's established pattern of explicit, named errors over silent wrong answers.
- **Require exact list order, not just set equality** — rejected: nothing else in the project requires two independently built `Dataset`s to enumerate classes in the same order, and requiring it here would be a surprising, easy-to-trip constraint for no real benefit.

### 4. `activation_maximize()` starts from random noise and uses fixed internal ascent hyperparameters

Starting point is a `uint8` random-noise image (converted to the model's preprocessed float range), not a real image, so the visualization reflects the class prototype rather than being anchored to whatever a chosen starting image already looks like. Step count, learning rate, and pixel clamping to `[0, 1]` after each step are internal constants — no new student-facing parameters, matching every prior phase's convention (ROADMAP.md, Key Design Decisions).

No regularization is applied to the ascent (Non-Goals). This is a known, accepted risk (see Risks below), deliberately deferred.

**Alternatives considered:**
- **Start from a real image of a different class** — rejected: makes the result depend on which starting image was chosen, which is exactly the kind of hidden, unreproducible parameter the project avoids exposing (or even having) elsewhere.
- **Expose step count / learning rate as `activation_maximize()` parameters** — rejected: no other function in the library exposes internal optimization hyperparameters to students; doing it here alone would be inconsistent.

## Risks / Trade-offs

- **`classify_generated()`'s per-class report will likely be weakly informative before Phase 6's real-data tuning** — already flagged in ROADMAP.md under Phase 4. A VAE whose reconstruction is starved by uncertainty weighting (Phase 3c's known risk) will produce blurry, class-ambiguous samples regardless of how well the sampling targets a class's cluster. → Documented as expected; Phase 4's job is correct plumbing against the stub dataset, not a quality bar (consistent with how every prior phase treated the stub).
- **Sampling near a class's empirical mean, rather than from its full learned distribution, could under- or over-represent how varied the VAE's generations for that class actually are.** → Acceptable for a capstone exercise whose point is a legible intended-vs-predicted comparison, not a rigorous characterization of the class-conditional distribution; the "spread" statistic (Decision 1) exists precisely so this isn't just a single point estimate.
- **`activation_maximize()` without regularization may produce images that read as adversarial noise to a human rather than a recognizable class prototype, especially on a small CNN with sharp decision boundaries.** → Explicitly accepted for now (Non-Goals); revisited once real data and a tuned classifier exist (Phase 6), where the risk is easier to evaluate honestly than against the stub.
- **The new `latent_access` capability adds a second axis (alongside `classify`/`sample`) that a future model type must think about.** → Kept optional and narrowly scoped (only `_VAE` declares it); a model that doesn't support it simply can't be passed as `classify_generated()`'s `generator_model`, with a clear capability error rather than a silent gap.
