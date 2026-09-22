## Why

Phase 3 delivered the generator arm as a fully **unsupervised** VAE: `train()` reads `data.images` and never touches `data.labels`, and the loss is reconstruction + a fixed `BETA * KL`. Labels reach the arm only as scatter-plot colors in `show_latent_space()`, applied after training.

That is not the project's intent. The intended deliverable is a **single supervised model** a student trains once and then uses two ways: first to classify novel images with better-than-chance accuracy, then to generate novel images of reasonable quality. Under that framing the generator arm's VAE gains a classification head fed from the same latent sample the decoder consumes, and the roadmap's Phase 4 "classifier portion" work belongs with the model it attaches to rather than after it.

The unsupervised design also leaves three concrete problems this change resolves together, because each one's fix constrains the others:

- **Nothing balances the loss terms.** A hand-tuned `BETA = 0.01` is the only control, and adding a classification term makes a third hand-tuned weight untenable. Uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018) replaces the hand-tuning for the two task losses.
- **A fixed `BETA` invites posterior collapse**, already flagged as a project risk. Phase 3's design deferred KL annealing explicitly ("a real technique for exactly this failure mode"); a supervised latent makes it necessary rather than optional.
- **`latent_dim = 2` was chosen only to keep `show_latent_space()` a projection-free scatter plot.** Routing classification through that bottleneck makes 2 dimensions the cap on both accuracy and sample quality — a visualization convenience become the model's binding constraint.

Nothing here is student-visible. Students receive a model object, feed it labeled data, call `train()`, and observe the two outcomes.

## Relationship to Phase 3b

This change was drafted from a checkout that predated Phase 3b (`picoface-phase3b-joint-model`, archived 2026-09-21), but it supersedes 3b and is authoritative where the two disagree. The "unsupervised" description above is Phase 3's end state; implementation starts from Phase 3b's code. 3b's two halves are treated differently:

- **3b's model decisions are overwritten.** The classification head moves from the shared conv-trunk features to the latent (Decision 1); the fixed `CLASSIFICATION_WEIGHT = 1.0` is replaced by learned uncertainty weighting (Decision 4); and the plain autoencoder loses its classification head and `classify` capability, returning to reconstruction-only (Decision 7).
- **3b's model-agnostic refactor is kept** and is the substrate this change builds on: the abstract `_Model` with per-model `training_step` and declared capabilities, the single `train`/`evaluate`/`predict`/`generate` in `_internals/model_api.py`, the unified `TrainingHistory`, and the `BaseShapeError`/`CapabilityError`/`GeneratorError` hierarchy (Decision 6). Everything below is expressed as changes to a `_Model` subclass or to that shared module, not as a new training loop or new per-arm verbs.

## What Changes

**The VAE becomes supervised and multi-task.**

- `build_vae(data)` gains a classification head sized to `data`'s class count. The reparameterized latent sample `z` feeds the decoder and the classifier head alike — classification is routed *through* the probabilistic bottleneck, not around it, so the supervisory signal shapes the same distribution `generate()` samples from.
- `train(model, data)` now requires labels for VAE models and validates class count, matching the classifier arm. The call shape is unchanged.
- `evaluate(model, data)` and `predict(model, image)` become available for VAE models, so one trained model answers both "how accurately does it classify?" and "what does it generate?".
- `generate(vae_model, n)` is unchanged.

**The loss gets three coordinated fixes.**

- **KL annealing** replaces the fixed `BETA`: the KL weight ramps from zero to 1 — the ELBO weight, given the per-pixel scale below — over an internal schedule, so the model learns to reconstruct and classify before the prior starts pulling. The end point is derived, not tuned.
- **Uncertainty weighting** (Kendall et al., 2018) balances the two task losses — reconstruction and classification — via learned homoscedastic parameters, removing the hand-tuned weight. The reconstruction parameter is the decoder's per-pixel noise scale (a calibrated Gaussian decoder), so it is also what calibrates reconstruction against the KL term. Weighting is deliberately **not** applied to the KL term, which is a regularizer with no observation-noise parameter to learn; annealing governs that instead.
- **Reconstruction and KL become a per-pixel ELBO.** Today reconstruction is mean-reduced per pixel while KL is summed per image, so relative to the ELBO the effective KL weight is `BETA * H*W*C` — `BETA = 0.01` is a KL weight of about 7.68 at 16x16x3, and would shift silently when Phase 5 picks a resolution. The per-pixel mean reduction is kept; reconstruction becomes the per-pixel Gaussian negative log-likelihood with the learned noise scale, and KL is divided by `H*W*C`. Together they are exactly the ELBO divided by the pixel count, so the reconstruction:KL balance is the likelihood's at every resolution, and reconstruction and classification pull with equal, resolution-independent strength.

**Latent dimensionality is unpinned, and the latent plot is dropped.**

- `show_latent_space()` is removed, along with its spec requirement. It was a nice-to-have whose only structural cost was pinning `latent_dim = 2`.
- `latent_dim` becomes an internal constant with no fixed value mandated by spec, free to be set for accuracy and sample quality. Still not student-facing.

**Companion models are kept alive with explicit, reduced roles.**

- The **CNN classifier** (Phase 2) stays and stays interoperable. It is no longer the student's classification path, but it becomes the capstone's *independent* judge: one model grading its own output is near-tautological, whereas an independently-trained CNN scoring the VAE's samples is a real measurement.
- The **plain autoencoder** stays, reconstruction-only: Phase 3b's classification head is removed from it. Its docs change to state plainly that it is not a required pedagogical step — students never see the model's architecture, so there is nothing for an AE-then-VAE progression to teach them.

**Training diagnostics become recorded, not asserted.**

- `TrainingHistory` records per-epoch reconstruction loss, KL, classification loss, the annealed KL weight, and the learned uncertainty weights; tests record held-out accuracy and wall-clock time alongside them.
- This is the evidence base for Phase 6: rather than pinning an accuracy number now, the change produces a table Phase 6 reads to decide how much tuning is warranted. A single loose better-than-chance assertion guards against silent breakage without pinning a number.

**The stub dataset gains spatially-distinguished classes.**

- Today's stub separates classes by mean brightness alone and its own docstring calls them "trivially separable". A latent bottleneck would score near-perfectly on it while telling us nothing about real shapes, which differ by *spatial arrangement* at similar brightness — a falsely reassuring number is worse than none.
- The generator gains matched-brightness, shape-bearing classes (a few lines of numpy). This is the mitigation ROADMAP.md already prescribes and never implemented: "the stub dataset should include at least one deliberately non-trivial synthetic class." It commits to no taxonomy; Phase 5 still owns real content.

## Capabilities

### New Capabilities
(none — no new capability; this change modifies four existing ones)

### Modified Capabilities
- `shape-generator`: the VAE becomes a supervised, multi-task model (classification head on the latent sample, labeled data required, `evaluate()`/`predict()` support), its loss becomes a per-pixel ELBO with KL annealing to the ELBO weight, plus uncertainty weighting of reconstruction against classification, `latent_dim`'s fixed value and `show_latent_space()` are removed, and training diagnostics become an explicit recorded output.
- `shape-classifier`: no behavior of the CNN changes; the capability gains a requirement fixing its new role as the independently-trained external validator that must remain interoperable with the rest of the project.
- `data-contract`: the internal stub-dataset generator must be able to produce classes distinguishable only by spatial arrangement, not by brightness alone.
- `model-interface`: the abstract model contract and shared verbs are unchanged in shape, but the `build_autoencoder()` model no longer has the classification capability, the `latent_mean` capability goes with `show_latent_space()`, `evaluate`/`predict` become importable from `picoface.generator` as well, the shared `TrainingHistory` gains the new per-epoch diagnostic fields, and the training loop tells a model where it is in training so it can schedule its own loss terms.

## Impact

- `src/picoface/_internals/generator_internals.py` — `_VAE`: classification head moved from the trunk to the latent (`z` in training, `mu` via `classify()`), `training_step` gains annealed KL, uncertainty weighting, and normalized loss reductions, `LATENT_DIM` revalued, `CLASSIFICATION_WEIGHT` removed. `_Autoencoder`: classification head and `classify` capability removed; `training_step` is reconstruction-only.
- `src/picoface/_internals/model_api.py` — `TrainingHistory` gains the new diagnostic fields; `train()` informs the model of training progress (epoch index and total) so the VAE can schedule its KL weight; the `latent_mean` capability and `_latent_mean()` helper are removed with `show_latent_space()`. The shared verbs are otherwise unchanged.
- `src/picoface/generator.py` — re-exports the shared `evaluate`/`predict` alongside `train`/`generate`; docstrings describe the VAE as one supervised model and state `build_autoencoder()` is not a required step.
- `src/picoface/viz.py` — `show_latent_space()` removed. `plot_training_history()` is untouched.
- `src/picoface/_internals/stub_data.py` — matched-brightness shape-bearing class generation.
- `tests/test_generator.py` — supervised path, held-out split, seeded runs, recorded diagnostics; `show_latent_space()` tests removed. `tests/test_joint_model.py` (from 3b) — tests asserting the autoencoder classifies or that `show_latent_space()` works are removed or narrowed to the VAE; the model-agnostic interface tests are kept. `tests/test_classifier.py` — vacuous `0.0 <= accuracy <= 1.0` assertions replaced.
- `openspec/ROADMAP.md` — Phase 3c row, revised arm description, Phase 4 scope clarified, `latent_dim`/beta open questions resolved or restated, and Phase 3b's model-decision entries (head on the trunk, fixed λ, autoencoder classifies) marked as superseded.
- `README.md` — no student-facing claims change; layout description unaffected.
- No new dependencies. Removing `show_latent_space()` removes the last reason `latent_dim` was constrained to avoid a PCA/t-SNE library, so the constraint disappears without the dependency ever being added.
- **Deferred to Phase 6, deliberately:** every numeric value here (latent dimensionality, annealing schedule length and shape, uncertainty-parameter floor, accuracy bar) — the annealing end point is not among them, being fixed at the ELBO weight of 1. This change delivers the mechanisms and the diagnostics that make Phase 6's tuning decisions evidence-based.
