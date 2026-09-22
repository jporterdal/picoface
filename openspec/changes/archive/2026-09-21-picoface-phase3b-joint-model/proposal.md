## Why

The Phase 3 VAE only generates: a student who has trained one has nothing to *see it do* until `generate()` runs, and pedagogically the network should be seen to classify before it is seen to generate. Giving the autoencoder/VAE a classifier branch trained jointly with reconstruction lets one network classify out of the box while it learns to generate.

Doing this cleanly exposes a structural problem: `train`, `evaluate`, `predict`, and `generate` are currently tied to specific model types (`classifier.train` assumes a CNN and CE loss; `generator._train_loop` type-switches on `is_variational`). A model that both classifies and generates does not fit either arm's functions as they stand. The fix is a small abstract model interface so the public verbs are model-agnostic and each model defines how it is trained — which also keeps any future model type from needing new public functions.

## What Changes

**Refactor (must land first; no behavior change):**
- Introduce an internal abstract model type. Each model supplies its own `training_step` (loss computation), inherits a default `check_data` (shape/class-count validation), and declares optional capabilities: `classify` (logits from a preprocessed image tensor), `sample` (draw images from a prior), `latent_mean` (for `show_latent_space`).
- Replace the two duplicated `_train_loop`s with one shared loop (DataLoader, Adam, epoch loop, wall-clock timing, per-epoch metric averaging) that calls `model.training_step`. The `if model.is_variational` type switch disappears into the model subclasses.
- Make `train`, `evaluate`, `predict`, and `generate` model-agnostic: one implementation in an arm-neutral internal module, re-exported from `picoface.classifier` and/or `picoface.generator` so existing imports keep working (`classifier.train is generator.train`).
- Unify the two `TrainingHistory` classes into one (keeping `loss`, `wall_clock_seconds`, `reconstruction_loss`, `kl_loss` field names). Give the two `ShapeError` classes a shared base so one `except` catches either. Keep `GeneratorError`; add a capability-error subclass for "this model can't do that".

**Joint model (depends on the refactor):**
- Add a discriminative branch to both `build_vae()` and `build_autoencoder()` models: a small head (`Linear → ReLU → Linear`, hidden size 32, mirroring Arm 1) reading from the **shared conv trunk features**, parallel to the mean/log-variance heads (VAE) or latent projection (AE). It is not attached to the latent `mu`/`z`.
- VAE loss becomes `MSE(recon) + β·KL + λ·CE(logits, labels)`; AE loss becomes `MSE(recon) + λ·CE`. β stays 0.01; λ is a fixed internal constant of 1.0 (not student-facing), flagged with β and `latent_dim` for Phase 6 revalidation.
- `build_vae(data)` / `build_autoencoder(data)` keep their signatures and now also record `num_classes` and `class_names` from `data`. `train(model, data, ...)` keeps its signature; per-epoch history additionally records `classification_loss` and `accuracy`.
- `classifier.evaluate(model, data)` and `classifier.predict(model, image)` now work on any model with a `classify` capability, including the joint VAE/AE (they become the shared model-agnostic implementations, replacing the CNN-only `_forward` path).
- `plot_training_history` also plots accuracy when the history contains it.
- Update `openspec/ROADMAP.md`: fix the stale phase-status table, describe Arm 2 as also classifying, add this change to the plan, and add λ / head architecture to Phase 6's revisit list and the new risks below.

**Explicit non-goals:**
- Making `show_latent_space()` show class clusters. A head on the trunk does not organize the 2D latent (a plain VAE's latent is only weakly class-organized once shapes vary in position/size); forcing it would require a head on `mu`, which measurably degrades reconstruction. Left to Phase 6.
- A class-conditional decoder (feeding the label into the decoder / `generate(cls=...)`). Deferred: a spike showed it does not yet work at β=0.01, and it would change `generate()`'s signature.
- Any change to whether/how students do Arm 1 as a separate stage — a Phase 7 curriculum decision. Arm 1's code is only touched by the refactor.
- Any harder test data than the existing stub dataset — Phase 6.

**Known consequences for later phases (recorded in the roadmap, not fixed here):**
- Phase 4's `classify_generated()` will report mostly circle/square-like results until β is tuned on real data (Phase 6); and the draft capstone spec's "intended class" wording does not fit an unconditional `generate()`.

## Capabilities

### New Capabilities
- `model-interface`: the model-agnostic public verbs (`train`, `evaluate`, `predict`, `generate`) and the abstract model contract behind them — each model defines its own training step and data checks; classification and sampling are optional capabilities; calling a verb on a model lacking the required capability raises a clear, named error; one shared `TrainingHistory`.

### Modified Capabilities
- `shape-classifier`: `train()`, `evaluate()`, `predict()` are now specified as model-agnostic (working on any model with the required capability, not only the CNN); the data/model consistency requirement applies to any such model; the training-history visualization also plots accuracy when present.
- `shape-generator`: `build_autoencoder()` and `build_vae()` models gain a classifier branch and train on reconstruction (+ KL for the VAE) plus cross-entropy; `train()` history records classification loss and accuracy; `generate()` is specified in terms of the `sample` capability (existing AE-mismatch behavior preserved); a trained joint model classifies via `evaluate`/`predict`.

## Impact

- Code: `src/picoface/_internals/` (new abstract model + shared `train`/`evaluate`/`predict`/`generate` module; `classifier_internals.py` and `generator_internals.py` refactored onto it), `classifier.py` and `generator.py` (thin re-exports), `viz.py` (`plot_training_history`, `show_latent_space` capability check).
- APIs: no breaking changes to existing student-facing signatures. `TrainingHistory` gains fields; exception classes gain a shared base/subclass. `is_variational` (internal) is replaced by capabilities.
- Tests: all existing tests must pass unchanged after the refactor step; new tests for the joint heads, chance-beating classification accuracy on the stub dataset, `classifier.evaluate`/`predict` on joint models, and capability errors.
- Time budget: the added heads are negligible against the existing CPU budget; the existing wall-clock regression test remains the guard.
- Dependencies: none added.
- Docs: `openspec/ROADMAP.md` updated.
