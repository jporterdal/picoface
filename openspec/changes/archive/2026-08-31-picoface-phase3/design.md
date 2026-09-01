## Context

Phase 2 delivered the classifier arm's plumbing (`build_classifier()`, `train()`, `evaluate()`, `predict()`) and established the conventions this phase follows: a public/`_internals` boundary, a project-specific `ShapeError` for shape/consistency failures, a shared internal `_forward`-style preprocessing helper, wall-clock training time logged in a `TrainingHistory` return value, and proof against the Phase 1 stub dataset rather than real content. `generator.py` is still the empty Phase 0 stub.

Arm 2 (generator) has no code dependency on Arm 1 (classifier) — the roadmap sequences them 2-then-3 for narrative reasons, not a technical one, and Phase 2's design explicitly reserved "shares no model code with the classifier" for this phase. This design covers only the generator arm's plumbing, proven against the stub dataset; real content stays deferred to Phase 5, and final hyperparameter tuning against real data stays deferred to Phase 6.

A preceding `/opsx:explore` session resolved several open points against the draft spec (`openspec/draft-specs/shape-generator/spec.md`) before this design was written; those resolutions are recorded as decisions below and in `openspec/ROADMAP.md`'s "Student-Visible Scope" section and Phase 3/6 descriptions.

## Goals / Non-Goals

**Goals:**
- Resolve concrete signatures for `build_autoencoder()`, `build_vae()`, `train()`, `generate()`, `show_latent_space()` that a first-year non-CS student can call without reading `_internals`.
- Pick an encoder/decoder architecture and hyperparameter defaults (including the KL weight β) that reliably train a VAE on the stub dataset within the CPU time budget.
- Keep the AE→VAE progression a true "swap one call, re-run `train()`" experience, per the roadmap's key design decision.
- Establish the internals layout for generator model/training code, consistent with the Phase 0–2 public/`_internals` boundary, without importing classifier model code.
- Give Phase 4 (linkage) a trained VAE object it can consume directly (in-memory, no serialization) for `classify_generated()` / `activation_maximize()`.

**Non-Goals:**
- Real shape taxonomy, resolution, or color depth — deferred to Phase 5.
- GAN, diffusion, or autoregressive generation — non-goals project-wide.
- Model save/load or serialization — out of scope for MVP.
- Configurable `latent_dim` or student-facing β — both fixed internal constants for MVP (Decisions 2–3).
- A CPU time budget, or notebook-ready polish, for `build_autoencoder()` — it must work correctly and share `build_vae()`'s call shape, nothing more (Decision 1; see also `openspec/ROADMAP.md`'s "Student-Visible Scope").
- `show_latent_space()` support for AE models — the spec only requires it for VAE models; a deterministic AE bottleneck could technically be plotted the same way, but that's not a tested requirement.

## Decisions

### 1. `build_autoencoder(data)` and `build_vae(data)` — same call shape, no `_from_shape` variants, shared internal encoder/decoder builder
Both take a `Dataset` directly (deriving `num_classes`/`input_shape` internally, as `build_classifier(data)` does) and return an opaque model handle. Unlike Phase 2's classifier arm, there is no `build_*_from_shape()` escape hatch for either function — no forcing use case surfaced for constructing a generator model without data in hand, and the smaller API surface is worth more here than the symmetry with Phase 2 would buy.

Both wrap a shared, non-public `_build_encoder(input_shape, latent_dim)` / `_build_decoder(latent_dim, output_shape)` pair in `_internals/generator_internals.py`. `build_autoencoder()` connects them directly; `build_vae()` connects the same encoder trunk to two additional linear heads (`mu`, `logvar`) and adds the reparameterization sampling step before the shared decoder. This is what makes the "swap the build call, keep training" promise real: the two models differ only in the small latent-parameterization step, not in the bulk of the conv/deconv architecture.

**Alternatives considered:**
- **`_from_shape()` variants mirroring Phase 2** — deferred, not rejected: no current need; can be added later without breaking `build_autoencoder(data)`/`build_vae(data)` if one surfaces.
- **Fully independent AE/VAE architectures with no shared builder** — rejected: would duplicate the conv/deconv stack twice for no benefit, and would undercut the "one new idea at a time" pedagogical intent from the roadmap.

### 2. Fixed 2-dimensional latent space, not student-configurable
`latent_dim` is a fixed internal constant (`= 2`), not a parameter of `build_vae()`/`build_autoencoder()`. This keeps `show_latent_space()` a direct (x, y) scatter plot with no dimensionality-reduction step, avoiding a new dependency (no `scikit-learn`/PCA/t-SNE) beyond the project's existing `torch`/`numpy`/`matplotlib`. It also keeps the arm's total conceptual surface smaller for a first-year audience, consistent with Phase 2's "no architecture-search knobs" precedent.

This is a placeholder value tuned for stub-data plumbing, not a final choice — `openspec/ROADMAP.md`'s Phase 6 description now explicitly calls out revisiting `latent_dim` against real data.

**Alternatives considered:**
- **Configurable `latent_dim` + PCA projection in `show_latent_space()` when `latent_dim > 2`** — rejected for MVP: adds a new dependency and a code path (projection) that only exists to work around a self-imposed configurability requirement nothing in the spec actually needs yet.

### 3. Reconstruction/KL loss weight (β) is a fixed internal constant, flagged for Phase 6 validation
`train()`'s VAE loss is `reconstruction_loss + beta * kl_divergence`, with `beta` a module-level constant in `_internals/generator_internals.py`, not a `train()` keyword argument. This follows Phase 2's precedent of fixing training-sensitive constants internally rather than exposing them to students (Phase 2 Decision 2's channel/kernel constants).

Unlike Phase 2's architecture constants, β is unusually load-bearing for whether VAE output looks like anything (too high → posterior collapse / uniform blur; too low → disorganized latent space) — the roadmap's own risk list already flags "VAE output may look too blurry to feel motivating." The initial value is a guess tuned qualitatively against the stub dataset's reconstructions, not validated against real content. `openspec/ROADMAP.md` now states explicitly (Phase 6 description, Open Questions) that β and `latent_dim` both get revisited once real data exists — this design intentionally does not claim to have picked a final value.

**Alternatives considered:**
- **Expose `beta` as a `train(..., beta=...)` parameter** — rejected for MVP: adds a knob whose correct value a first-year student has no way to reason about, working against the "no architecture-search knobs" pattern; can be promoted to a parameter later without breaking the simple call if a real need emerges.
- **KL annealing schedule (ramp β from 0 during training)** — deferred, not rejected: a real technique for exactly this failure mode, but adds a training-loop concept beyond MVP scope; worth trying during Phase 6's real-data tuning pass if a fixed β proves insufficient.

### 4. Reconstruction loss: MSE, not BCE
`_forward`-equivalent preprocessing normalizes pixel values to `[0, 1]` (matching Phase 2's classifier preprocessing convention), and the decoder's final layer applies a sigmoid to match that range. Reconstruction loss uses per-pixel MSE rather than binary cross-entropy. BCE is the more common choice in VAE tutorials, but it implicitly treats pixel values as independent Bernoulli probabilities, which fits near-binary content (e.g. MNIST) better than the stub dataset's (and likely Phase 5's) continuous-valued, noisy synthetic images. MSE is simpler to reason about and a more honest fit for this project's actual pixel distributions.

**Alternatives considered:**
- **BCE reconstruction loss** — deferred, not rejected: worth an empirical comparison once real image content exists (Phase 6), since BCE can produce sharper edges on near-binary content.

### 5. `train()` is one public entry point for both AE and VAE models, dispatching internally on model type
Mirrors Phase 2's single `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)` signature and its "runs the full loop internally" contract — a student never sees a different call shape when they swap `build_autoencoder()` for `build_vae()`. Internally, `_train_loop` in `generator_internals.py` checks the model's type once and selects the reconstruction-only loss (AE) or reconstruction+KL loss (VAE, Decision 3), same as `_build_classifier`/`_forward` centralize logic in Phase 2 rather than duplicating it per call site.

Returns a `TrainingHistory` with `loss` (total, per epoch) and `wall_clock_seconds`, matching the classifier arm's field names for consistency. For VAE models, `_train_loop` additionally records `reconstruction_loss` and `kl_loss` per epoch (empty lists for AE) — not part of the public spec's required behavior, but cheap to capture now and directly useful for the Phase 6 β-validation work already flagged in Decisions 3 and in `openspec/ROADMAP.md`.

**Alternatives considered:**
- **Separate `train_autoencoder()`/`train_vae()` functions** — rejected: breaks the "swap the build call, keep training" promise that's the whole pedagogical point of the AE→VAE progression.

### 6. `generate()` is VAE-only; misuse raises `GeneratorError`
`generate(model, n)` only accepts models built by `build_vae()`. Calling it with a `build_autoencoder()` model raises a new public `picoface.generator.GeneratorError` (subclasses `ValueError`, importable/catchable like Phase 2's `ShapeError`) naming the mismatch explicitly (e.g. "generate() requires a model built by build_vae(); got a build_autoencoder() model — build_autoencoder() models have no probabilistic prior to sample from"), rather than failing on a missing sampling method somewhere inside `_internals`. This mirrors Phase 2 Decision 8's rationale: turn an otherwise-opaque internals failure into a specific, catchable, first-year-readable error at the point of misuse.

`generate()` itself samples `n` latent vectors from the standard normal prior `N(0, I)` (the VAE's training-time prior, since latent_dim is fixed at 2 per Decision 2) and runs them through the shared decoder.

**Alternatives considered:**
- **Silently support `generate()` on AE models by sampling from the empirical encoding distribution (mean/std of encoded training data)** — rejected: AE's latent space has no encouraged structure (no KL term), so "sampling" from it produces meaningless output; a clear error is more honest than a function that appears to work but doesn't.
- **Reuse `ShapeError` instead of a new `GeneratorError`** — rejected: this isn't a shape mismatch, it's a wrong-model-type/wrong-capability call; a distinctly named exception makes `except GeneratorError` communicate the actual problem.

### 7. Decoder output shape is validated at build time via dummy pass-through, using a generator-local `ShapeError`
Following Phase 2 Decision 1's pattern, `_build_encoder`/`_build_decoder` run a dummy tensor through the full encode→decode path at build time and assert the decoder's output shape exactly matches `input_shape`, raising `picoface.generator.ShapeError` (a separate class from `picoface.classifier.ShapeError`, not imported from it — consistent with the "no shared model code between arms" decision) if a transpose-conv/upsample mismatch would otherwise surface as a cryptic tensor-shape error deep in training. `ShapeError` also covers the same image-shape-vs-`model.input_shape` consistency check `_forward` performs in the classifier arm, reused here as its own local check.

**Alternatives considered:**
- **Import and reuse `picoface.classifier.ShapeError`** — rejected: would create a cross-arm dependency the project has otherwise deliberately avoided (Phase 2 Context: "Arm 1 has no dependency on Arm 2" and vice versa); a same-named, independently-defined class costs nothing and keeps the arms decoupled.

### 8. `show_latent_space()` plots the encoder's mean (`mu`), not a sampled `z`
For a `build_vae()` model, `show_latent_space(vae_model, data)` encodes `data.images` through the encoder trunk and plots `mu` (the mean of the latent distribution) rather than a stochastic sample — this is the standard VAE visualization convention, since `mu` is deterministic given the input and gives a stable, reproducible plot rather than one that jitters between calls. Points are colored by `data.labels`/`class_names`, matching the pattern in `evaluate()`/`predict()` of always surfacing class names rather than raw indices where a student would read them.

### 9. Generator internals live under `_internals/generator_internals.py`, independent of `classifier_internals.py`
Same one-module-per-arm convention Phase 2 established (`_internals/classifier_internals.py`) and explicitly reserved this phase for. Contains the `nn.Module` subclasses (AE, VAE), `_build_encoder`/`_build_decoder`, the shared preprocessing helper (NHWC uint8 → NCHW float `[0, 1]`, duplicated from — not imported from — the classifier arm's equivalent, per Decision 7's rationale), and `_train_loop`.

## Risks / Trade-offs

- **β is a guess, not a validated value.** → Explicitly flagged as a Phase 6 revisit in `openspec/ROADMAP.md`; Decision 5's per-epoch `reconstruction_loss`/`kl_loss` logging gives Phase 6 something to look at when tuning it against real data.
- **Fixed `latent_dim=2` may be too constraining once real (more visually complex) content arrives in Phase 5.** → Also flagged for Phase 6 revisit; if 2D proves insufficient, promoting it to a configurable-but-defaulted parameter is an additive change, not a breaking one.
- **MSE reconstruction loss may produce blurrier output than BCE on some content.** → Deferred empirical comparison to Phase 6 (Decision 4); blur is already a documented, accepted risk project-wide (roadmap Risks/Trade-offs).
- **Decoder output-shape mismatches (transpose-conv/upsample arithmetic) are a common source of cryptic PyTorch errors.** → Mitigated by Decision 7's build-time dummy pass-through and dedicated `ShapeError`, mirroring Phase 2's proven approach for the same class of bug.
- **`build_autoencoder()` receiving no time-budget enforcement (Non-Goal) means it could regress to something impractically slow without a test catching it.** → Accepted per the roadmap's Student-Visible Scope: AE is a nice-to-have that only needs to work, not perform; if it becomes part of a Phase 7 notebook after all, a budget can be added then.

## Open Questions

None blocking implementation. Exact encoder/decoder channel counts, kernel sizes, and the initial β value are left to be tuned during implementation against the stub dataset, not required to be finalized in this design — consistent with how Phase 2 left its own architecture constants to implementation-time tuning.
