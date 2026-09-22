## Why

Phase 3 delivered the generator arm as a fully **unsupervised** VAE: `train()` reads `data.images` and never touches `data.labels`, and the loss is reconstruction + a fixed `BETA * KL`. Labels reach the arm only as scatter-plot colors in `show_latent_space()`, applied after training.

That is not the project's intent. The intended deliverable is a **single supervised model** a student trains once and then uses two ways: first to classify novel images with better-than-chance accuracy, then to generate novel images of reasonable quality. Under that framing the generator arm's VAE gains a classification head fed from the same latent sample the decoder consumes, and the roadmap's Phase 4 "classifier portion" work belongs with the model it attaches to rather than after it.

The unsupervised design also leaves three concrete problems this change resolves together, because each one's fix constrains the others:

- **Nothing balances the loss terms.** A hand-tuned `BETA = 0.01` is the only control, and adding a classification term makes a third hand-tuned weight untenable. Uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018) replaces the hand-tuning for the two task losses.
- **A fixed `BETA` invites posterior collapse**, already flagged as a project risk. Phase 3's design deferred KL annealing explicitly ("a real technique for exactly this failure mode"); a supervised latent makes it necessary rather than optional.
- **`latent_dim = 2` was chosen only to keep `show_latent_space()` a projection-free scatter plot.** Routing classification through that bottleneck makes 2 dimensions the cap on both accuracy and sample quality — a visualization convenience become the model's binding constraint.

Nothing here is student-visible. Students receive a model object, feed it labeled data, call `train()`, and observe the two outcomes.

## What Changes

**The VAE becomes supervised and multi-task.**

- `build_vae(data)` gains a classification head sized to `data`'s class count. The reparameterized latent sample `z` feeds the decoder and the classifier head alike — classification is routed *through* the probabilistic bottleneck, not around it, so the supervisory signal shapes the same distribution `generate()` samples from.
- `train(model, data)` now requires labels for VAE models and validates class count, matching the classifier arm. The call shape is unchanged.
- `evaluate(model, data)` and `predict(model, image)` become available for VAE models, so one trained model answers both "how accurately does it classify?" and "what does it generate?".
- `generate(vae_model, n)` is unchanged.

**The loss gets three coordinated fixes.**

- **KL annealing** replaces the fixed `BETA`: the KL weight ramps from zero over an internal schedule, so the model learns to reconstruct and classify before the prior starts pulling.
- **Uncertainty weighting** (Kendall et al., 2018) balances the two task losses — reconstruction and classification — via learned homoscedastic parameters, removing the hand-tuned weight. It is deliberately **not** applied to the KL term, which is a regularizer with no observation-noise parameter to learn; annealing governs that instead.
- **Loss reductions are normalized to a common per-image scale.** Today reconstruction is mean-reduced per pixel while KL is summed per image, so the effective KL weight scales with `H*W*C` — `BETA = 0.01` is roughly the equivalent of a conventional beta near 7.7 at 16x16x3, and would shift silently when Phase 5 picks a resolution. Left unfixed, the learned uncertainty parameters would absorb this artifact and hide it.

**Latent dimensionality is unpinned, and the latent plot is dropped.**

- `show_latent_space()` is removed, along with its spec requirement. It was a nice-to-have whose only structural cost was pinning `latent_dim = 2`.
- `latent_dim` becomes an internal constant with no fixed value mandated by spec, free to be set for accuracy and sample quality. Still not student-facing.

**Companion models are kept alive with explicit, reduced roles.**

- The **CNN classifier** (Phase 2) stays and stays interoperable. It is no longer the student's classification path, but it becomes the capstone's *independent* judge: one model grading its own output is near-tautological, whereas an independently-trained CNN scoring the VAE's samples is a real measurement.
- The **plain autoencoder** stays, frozen. Its docs change to state plainly that it is not a required pedagogical step — students never see the model's architecture, so there is nothing for an AE-then-VAE progression to teach them.

**Training diagnostics become recorded, not asserted.**

- `TrainingHistory` records per-epoch reconstruction loss, KL, classification loss, the annealed KL weight, and the learned uncertainty weights; tests record held-out accuracy and wall-clock time alongside them.
- This is the evidence base for Phase 6: rather than pinning an accuracy number now, the change produces a table Phase 6 reads to decide how much tuning is warranted. A single loose better-than-chance assertion guards against silent breakage without pinning a number.

**The stub dataset gains spatially-distinguished classes.**

- Today's stub separates classes by mean brightness alone and its own docstring calls them "trivially separable". A latent bottleneck would score near-perfectly on it while telling us nothing about real shapes, which differ by *spatial arrangement* at similar brightness — a falsely reassuring number is worse than none.
- The generator gains matched-brightness, shape-bearing classes (a few lines of numpy). This is the mitigation ROADMAP.md already prescribes and never implemented: "the stub dataset should include at least one deliberately non-trivial synthetic class." It commits to no taxonomy; Phase 5 still owns real content.

## Capabilities

### New Capabilities
(none — no new capability; this change modifies three existing ones)

### Modified Capabilities
- `shape-generator`: the VAE becomes a supervised, multi-task model (classification head on the latent sample, labeled data required, `evaluate()`/`predict()` support), its loss gains KL annealing + uncertainty weighting + scale normalization, `latent_dim`'s fixed value and `show_latent_space()` are removed, and training diagnostics become an explicit recorded output.
- `shape-classifier`: no behavior of the CNN changes; the capability gains a requirement fixing its new role as the independently-trained external validator that must remain interoperable with the rest of the project.
- `data-contract`: the internal stub-dataset generator must be able to produce classes distinguishable only by spatial arrangement, not by brightness alone.

## Impact

- `src/picoface/_internals/generator_internals.py` — classification head, supervised training loop, annealing schedule, uncertainty-weighting parameters, normalized loss reductions, extended `TrainingHistory`, `LATENT_DIM` revalued. `_encode_mu` is retained and repurposed for `evaluate()`/`predict()` rather than plotting.
- `src/picoface/generator.py` — `build_vae()` sized to class count; `train()` label and class-count validation; new `evaluate()`/`predict()`; `build_autoencoder()` docstring states it is not a required step.
- `src/picoface/viz.py` — `show_latent_space()` removed. `plot_training_history()` is untouched.
- `src/picoface/_internals/stub_data.py` — matched-brightness shape-bearing class generation.
- `tests/test_generator.py` — supervised path, held-out split, seeded runs, recorded diagnostics; `show_latent_space()` tests removed. `tests/test_classifier.py` — vacuous `0.0 <= accuracy <= 1.0` assertions replaced.
- `openspec/ROADMAP.md` — Phase 3c row, revised arm description, Phase 4 scope clarified, `latent_dim`/beta open questions resolved or restated.
- `README.md` — no student-facing claims change; layout description unaffected.
- No new dependencies. Removing `show_latent_space()` removes the last reason `latent_dim` was constrained to avoid a PCA/t-SNE library, so the constraint disappears without the dependency ever being added.
- **Deferred to Phase 6, deliberately:** every numeric value here (latent dimensionality, annealing schedule length and ceiling, uncertainty-parameter clamp, accuracy bar). This change delivers the mechanisms and the diagnostics that make Phase 6's tuning decisions evidence-based.
