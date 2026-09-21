## Context

Phase 2 (classifier arm) and Phase 3 (generator arm) each ship their own model classes, `TrainingHistory`, `ShapeError`, preprocessing helper, and `_train_loop`, deliberately sharing no code (Phase 2 Context; Phase 3 Decisions 7 and 9). The public verbs are correspondingly arm-bound: `classifier.train/evaluate/predict` call `model(x)` expecting logits and a CE loop; `generator.train` type-switches on `model.is_variational` inside its loop; `generator.generate` and `viz.show_latent_space` gate on the same flag. See `proposal.md` — Why for motivation.

Current shape of the generator models ([generator_internals.py](../../../src/picoface/_internals/generator_internals.py)): `_ConvEncoderTrunk` (two stride-2 conv→ReLU, flatten) feeds either `_Encoder`'s single `to_latent` linear (AE) or `_VAE`'s `fc_mu`/`fc_logvar` (VAE), then a shared `_Decoder`. `_VAE.forward` returns `(recon, mu, logvar)`.

An exploratory spike (throwaway scripts, not in the repo) informed the head placement and loss weighting. Setup: 4 synthetic shape classes (circle, square, triangle, plus) rendered at 16×16×1 with random position, size, brightness and noise; 300 train / 150 test per class; 3 seeds; the real trunk and decoder reused; Arm 1's standalone CNN as a reference. Key results (100 epochs unless noted):

| Variant | Test acc. | Recon MSE | 2D-latent 5-NN acc. (chance 0.25) |
|---|---|---|---|
| Arm 1 standalone CNN (30 ep) | 0.996 | — | — |
| Plain VAE | — | 0.043 | 0.40 |
| Head on trunk, λ=1 | 0.976 | 0.044 | 0.42 |
| Head on trunk, λ=0.1 / 1 / 10 (30 ep) | 0.956 / 0.952 / 0.951 | 0.045 | 0.39–0.40 |
| Head on `mu`, λ=1 | 0.949 | 0.059 | 0.95 |
| Head on trunk + label into decoder | 0.977 | 0.039 | 0.33; class-conditional samples judged correct only 27% of the time (≈chance) |

(Mean-image reconstruction baseline on this set: 0.063.) The spike set is not the Phase 5 dataset; treat as directional.

## Goals / Non-Goals

**Goals:**
- One network that classifies and generates, trained by the same single `train()` call, with no degradation of reconstruction.
- Public verbs whose call shape never depends on model kind, and an internal contract simple enough that a new model kind needs no new public functions.
- A refactor that is verifiably behavior-preserving before any joint-model behavior is added.

**Non-Goals:**
- Class-clustered latent plots, a class-conditional decoder, or `generate(cls=...)` (see proposal — Explicit non-goals; the spike table above is the evidence).
- Model save/load, GPU, or any new dependency.
- Deciding whether/how students do Arm 1 separately (Phase 7).
- Tuning λ, the head, or β against real data (Phase 6).

## Decisions

### 1. An internal abstract model type with a template-method training contract
A new arm-neutral module `_internals/model_api.py` defines an abstract `_Model(nn.Module)`:

- **Required:** `training_step(inputs, labels) -> (loss_tensor, metrics: dict[str, float])` — inputs are already-preprocessed NCHW float tensors in [0, 1]; the model returns its own total loss and named metrics.
- **Provided with a default:** `check_data(data)` — validates `data`'s image shape against `input_shape` and class count against `num_classes`, raising the model's stored `shape_error_cls`; models may override.
- **Optional capabilities**, declared in a class-level `capabilities: frozenset[str]` (a subset of `{"classify", "sample", "latent_mean"}`) and implemented as methods: `classify(x) -> logits`, `sample(n) -> images tensor`, `encode_mu(x) -> mu`. Base-class defaults raise `NotImplementedError`; callers check `capabilities` first and never rely on that.
- **Common attributes:** `input_shape`, and for classifying models `num_classes` and `class_names`.

Concrete models: `_CNNClassifier` (capabilities `{classify}`; `training_step` = CE), `_Autoencoder` (`{classify}`; MSE + λ·CE), `_VAE` (`{classify, sample, latent_mean}`; MSE + β·KL + λ·CE).

**Why this over each model owning its loop:** the two existing `_train_loop`s already duplicate DataLoader/Adam/epoch/timing code; a per-model loop would triple it. A template method keeps one loop and lets the model define only what actually differs — the loss. It also removes the `if model.is_variational` switch: the branch becomes polymorphism.

**Why an explicit `capabilities` set over `hasattr`/`isinstance`:** it is explicit, testable, and gives the error message something to name ("this model cannot classify"). `isinstance` checks against concrete classes would re-couple the verbs to specific models, defeating the point.

**Alternatives considered:**
- *Separate `Trainable`/`Classifies`/`Generates` mixin ABCs* — rejected: more types than the two current capabilities justify; the frozenset can be replaced by mixins later without touching callers.
- *Each model exposes `fit(data, epochs, ...)`* — rejected for the duplication above.

### 2. One implementation of the public verbs, re-exported from each arm's module
`train`, `evaluate`, `predict`, `generate` live in `_internals/model_api.py`. `classifier.py` re-exports `train`, `evaluate`, `predict`; `generator.py` re-exports `train`, `generate` (both list them in `__all__`). `classifier.train is generator.train`. The public modules remain thin and expose no model classes or loop code, as required by "Training-loop internals hidden".

`model_api` imports nothing from the public modules (no cycle); public modules and `viz` import from it. Each `build_*` function still receives its arm's error class and stores it on the model as `shape_error_cls`, preserving the existing pattern (this is how the shared loop raises the *model's* arm's `ShapeError` without importing the public module).

**Alternatives considered:**
- *Top-level `picoface.train` only* — rejected: breaks existing imports and the per-arm student vocabulary.
- *`generator` importing from `classifier`* — rejected: recreates the cross-arm dependency this codebase deliberately avoided.
- *Thin per-module wrappers* — rejected: duplicated signatures/docstrings would drift.

**Note — this reverses Phase 3 Decision 9** ("preprocessing duplicated, not imported"). That decision protected arm decoupling; with the shared code living in an arm-neutral module that neither arm imports from the other, the reason no longer applies, and the two duplicate `_preprocess`/`_forward` helpers collapse into one.

### 3. Shared training loop and unified `TrainingHistory`
`model_api.train` performs: `model.check_data(data)`; set `model.class_names = data.class_names` (as `classifier.train` does today, needed for `build_classifier_from_shape` models); preprocess each batch (validate shape, NHWC uint8 → NCHW float32 /255) using the one shared helper; DataLoader (shuffle), Adam, CPU; call `model.training_step`; `loss.backward()` and `step()`; average `loss` and each metric per epoch; record wall-clock time.

One `TrainingHistory` dataclass: `loss`, `wall_clock_seconds` (as today), `reconstruction_loss`, `kl_loss`, plus new `classification_loss` and `accuracy` — list fields defaulting to empty. A model reporting a metric key with no matching field is a developer error, not a silent drop. `accuracy` is the mean of per-batch training accuracy, i.e. a training-time indicator, not held-out performance (`evaluate()` on held-out data is that).

The CNN classifier continues to report `loss` only, so the refactor step is strictly behavior-preserving and its plot is unchanged; adding accuracy for the CNN is a possible later, separate improvement.

### 4. Classification branch attached to the trunk features, not to the latent
The new head is `Linear(flatten_dim, 32) → ReLU → Linear(32, num_classes)`, mirroring Arm 1's fc head, reading the flattened `_ConvEncoderTrunk` features in parallel with `fc_mu`/`fc_logvar` (VAE) or `to_latent` (AE).

**Why the trunk:** in the spike, a head on the trunk gave the best classifier (0.976 vs 0.949 on `mu`), left reconstruction untouched (0.044 vs 0.043 without the head), and avoided forcing the 2D bottleneck to carry both class and appearance. A head on `mu` clustered the latent (0.95) but cost ~35% reconstruction (0.059, close to the mean-image baseline 0.063), which would make the generator materially worse. A head on the sampled `z` adds noise to the classifier for no gain.

The AE is restructured to hold `trunk`, `to_latent`, `decoder`, and `head` directly (as the VAE already does), replacing the `_Encoder` wrapper, because the head needs the trunk features and not just the bottleneck output. `_build_encoder`/`_build_decoder` from Phase 3 Decision 1 are otherwise unchanged; `_build_decoder` continues to be shared by both.

### 4b. `classify` is differentiable and mode-free
`classify(x)` performs trunk → head only (no decoder, no sampling) and does not wrap itself in `no_grad` or toggle train/eval. `evaluate`/`predict` add `no_grad()` and `model.eval()` around it. This keeps `classify` usable for Phase 4's `activation_maximize()`, which needs gradients with respect to pixels, and keeps evaluation deterministic (no reparameterization noise).

### 5. Loss composition and weights
- VAE: `MSE_mean(recon, x) + BETA · KL + CLASSIFICATION_WEIGHT · CE(logits, y)`.
- AE: `MSE_mean(recon, x) + CLASSIFICATION_WEIGHT · CE(logits, y)`.
- `BETA = 0.01` unchanged. `CLASSIFICATION_WEIGHT = 1.0`, a module-level constant beside `BETA` with a comment recording the spike rationale, not a student-facing parameter (Phase 2/3 precedent).
- Reductions unchanged: MSE and CE are batch/pixel means; KL is summed over latent dims and averaged over the batch.

This is the classification term of Kingma et al.'s M2 semi-supervised VAE (α-weighted −log q(y|x) added to the objective), without M2's label-conditioned decoder. **Why λ = 1.0:** on the spike set, accuracy and reconstruction were insensitive to λ between 0.1 and 10 (the numerical mismatch between CE ≈ 0.7–1.4 initially and mean-MSE ≈ 0.01–0.05 did not matter in practice — Adam normalizes per parameter, so the decoder and mu/logvar heads are unaffected by CE's scale, and only the shared trunk sees the mix). 1.0 is therefore the plain, defensible default. It is a placeholder validated on synthetic data only.

**Alternatives considered:** a label-conditioned decoder (rejected for MVP: at β=0.01 the decoder ignored the label — 27% conditional hit rate ≈ chance — and it would change `generate()`'s signature); rescaling reconstruction to summed-pixel SSE with a proper ELBO (out of scope; would retune a Phase 3 decision, and is a Phase 6 candidate).

### 6. Errors
A small `_internals/errors.py` defines `BaseShapeError(ValueError)`, `CapabilityError(ValueError)`, and `GeneratorError(CapabilityError)` (defined there, and re-exported by `generator.py`, because the shared `generate()` in `model_api.py` must raise it without importing a public module). `classifier.ShapeError` and `generator.ShapeError` both subclass `BaseShapeError`, stay defined in their arm modules (as now), and remain individually importable. `generate()` raises `GeneratorError` for any model without the `sample` capability (AE, CNN); a verb failing on any other missing capability (e.g. `evaluate` on a hypothetical non-classifying model) raises `CapabilityError` directly, with a message naming the verb and what the model cannot do. `BaseShapeError` and `CapabilityError` are re-exported from both arm modules.

`viz.show_latent_space()` keeps raising `GeneratorError`, now keyed on the `latent_mean` capability instead of `is_variational` (AE lacks it, so behavior is unchanged for AE models).

### 7. `plot_training_history` and `viz` imports
`plot_training_history` imports the unified `TrainingHistory` from `model_api`. When `history.accuracy` is non-empty it returns a two-panel figure (loss; accuracy); otherwise the existing single-axes figure, unchanged.

## Risks / Trade-offs

- **The refactor could silently change classifier or generator behavior.** → It is sequenced first and gated on the existing suite passing unmodified (tests exercise only the public API); the CPU wall-clock regression tests remain as speed guards.
- **λ, the head architecture, and the head placement were validated only on a self-made synthetic set.** → Placeholders flagged in the roadmap for Phase 6, alongside β and `latent_dim`; nothing here is exposed to students.
- **Joint classifier is weaker than the standalone CNN (0.976 vs 0.996 in the spike).** → Accepted; Arm 1 remains for maximum accuracy. If the gap matters on real data, a deeper head is an internal-only fix.
- **The latent-space plot will not show clean class clusters**, which a student may expect once the network "knows" the classes. → Stated as a non-goal in the spec; recorded as a Phase 6 risk in the roadmap; no student-facing claim is made.
- **Generator output is weak at β=0.01** (spike: samples judged as circle/square only; recon 0.043 vs mean-image 0.063), so Phase 4's `classify_generated()` will be uninformative until β is tuned, and the draft "intended class" wording does not fit unconditional generation. → Recorded in the roadmap for Phases 4 and 6; not fixed here.
- **Abstract layer over three models could be over-engineering.** → It is one small ABC with two required members and removes real duplication (two loops, two preprocessors, a type switch); the capability set can be swapped for mixins later without touching callers.
- **Plot behavior changes shape (two panels) for joint-model histories.** → Only when accuracy is present; existing classifier plots are unchanged and the existing smoke test guards this.

## Migration Plan

1. Land the refactor (tasks §1–§2) first; run the full existing test suite unmodified as the gate — no joint-model behavior is introduced until it is green.
2. Only then add the classification heads and loss terms (tasks §3 onward), each of which depends on §1–§2.
3. Roadmap and docs updates last (tasks §7).

Rollback is a plain git revert of this change; there is no data or serialization format to migrate (no model save/load exists).

## Open Questions

- Exact names for the shared exception types and the internal module are settled here (`BaseShapeError`, `CapabilityError`, `model_api.py`) but are internal-facing; renaming later is cheap and does not affect the specs.
