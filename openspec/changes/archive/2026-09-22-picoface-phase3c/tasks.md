## 1. Per-pixel ELBO (do this first)

Decision 5 requires the reconstruction and KL terms to be on the per-pixel ELBO scale before annealing or uncertainty weighting is layered on, so the learned `sigma`s are introduced on the scale they are specified on. All edits are in `_VAE.training_step` in `generator_internals.py`. `D` below is `H*W*C` of the model's `input_shape`.

- [x] 1.1 Keep the reconstruction term's per-pixel mean reduction (`mse_loss(..., reduction="mean")`); do not switch to a summed reduction (Decision 5, alternatives)
- [x] 1.2 Divide the per-image KL term (sum over latent dimensions, mean over batch — unchanged) by `D`, so reconstruction and KL together form the ELBO divided by `D`
- [x] 1.3 Before changing anything, record reconstruction loss and KL magnitudes on the stub dataset under the current objective; after 1.2, record them again, and note in the diagnostics record (Section 8) that the old `BETA = 0.01` corresponds to a KL weight of `0.01 * D` (7.68 at 16x16x3) on the new scale, versus the ELBO weight of 1
- [x] 1.4 Remove `BETA` and its comment block (the "deliberately kept small" reasoning compared it against the wrong scale); the KL weight now comes from the annealing schedule in 4.2, ending at 1
- [x] 1.5 Confirm on two stub datasets differing only in resolution (e.g. 16x16 and 32x32) that, after training, the learned reconstruction `log(sigma_r^2)` values agree closely and the reconstruction and classification task weights are comparable — i.e. the task balance does not scale with `D`. Train for enough optimizer steps that `log(sigma_r^2)` has reached equilibrium with `log(MSE)` in both runs (at the 10× multiplier, about 340 steps; on the stub that means more images per class or more epochs than the defaults), and record both runs. (Depends on Section 4; run after 4.3 and 4.10.)

## 2. Stub dataset: spatially-distinguished classes

- [x] 2.1 Extend `make_stub_dataset()` with a mode producing simple filled figures (e.g. square, circle, cross) rendered with numpy, positioned to fill comparable image area
- [x] 2.2 Normalize the rendered classes to matched mean brightness, so no single global statistic separates them
- [x] 2.3 Keep the existing brightness-separated mode available and selectable, for plumbing and shape-agnosticism tests that do not need difficulty
- [x] 2.4 Add a test asserting that classifying the spatial classes by per-image mean brightness alone performs no better than chance, while the classes remain recoverable — this is the guard that keeps the dataset honest
- [x] 2.5 Confirm the generator still honors arbitrary `height`/`width`/`channels`/`class_names` and remains non-public

## 3. Internals: the supervised VAE model

Starting point is Phase 3b's code: `_VAE` and `_Autoencoder` are `_Model` subclasses in `generator_internals.py`, each with a classification head on the conv-trunk features (Proposal: Relationship to Phase 3b). 3b's refactor is kept; its model decisions are overwritten.

- [x] 3.1 Revalue `LATENT_DIM` above 2 in `generator_internals.py`, record the value tried and its effect in the diagnostics table (final value is a Phase 6 decision)
- [x] 3.2 Move `_VAE`'s classification head off the trunk features: a small stack from `latent_dim` to `num_classes`, reading the reparameterized sample `z` in `forward()` — the same tensor the decoder consumes (Decision 1)
- [x] 3.3 Change `_VAE.classify()` to read `mu` (via `encode_mu()`) through the latent head rather than the trunk, so `evaluate()`/`predict()` are deterministic (Decision 1); confirm it remains differentiable with respect to input pixels (Phase 4's `activation_maximize()` needs this)
- [x] 3.4 Remove `"latent_mean"` from `_VAE.capabilities`; keep `encode_mu()` as the internal the latent head's inference path uses
- [x] 3.5 Remove `_Autoencoder`'s classification head and its `classify` capability; its `num_classes` becomes `None` so the inherited `check_data` skips the class-count check (Decision 7)
- [x] 3.6 Remove `CLASSIFICATION_WEIGHT` and its comment block; replace 3b's head-placement comment with one citing Decision 1
- [x] 3.7 Keep `_validate_decode_shape()` working for `_VAE.forward()`'s return tuple and the autoencoder's now single-tensor return; leave the shared conv trunk and decoder architecture unchanged

## 4. Internals: VAE loss and the shared training loop

- [x] 4.1 Add a training-progress hook to `_Model` in `model_api.py` (no-op by default) and call it from the shared `train()` before each epoch with the epoch index and the caller's total `epochs` (Decision 6); do not add any model-type branching to `train()`
- [x] 4.2 Implement the KL annealing schedule in `_VAE`, driven by that hook: ramp the KL weight from zero to 1 (the ELBO weight — not a tuned ceiling, Decision 3) as a function of progress relative to total epochs; only the schedule's length and shape are provisional, with no student-facing parameter
- [x] 4.3 Add learned log-variances `log(sigma_r^2)` and `log(sigma_c^2)` as `nn.Parameter`s on `_VAE`, and combine the task losses in `_VAE.training_step` per Decision 4:
  - reconstruction: `MSE / (2 * sigma_r^2) + 0.5 * log(sigma_r^2)` (per-pixel Gaussian NLL; `MSE` is the per-pixel mean from 1.1)
  - classification: `CE / sigma_c^2 + 0.5 * log(sigma_c^2)` (no factor of 2 in the denominator)
  - total: reconstruction + classification + `kl_weight * KL / D`
- [x] 4.4 Do NOT apply uncertainty weighting to the KL term (Decision 3) — it is governed by 4.2's schedule alone
- [x] 4.5 Add a floor on each learned `log(sigma^2)` value, capping how large a task's weight can grow — the mitigation lever for the lower-loss-earns-higher-weight starvation dynamic (Decision 4); leave its value provisional and record it. Enforce it on the stored parameter: clamp each log-variance in place to at least the floor in the post-step hook from 4.10, so a parameter at the floor keeps its gradient and can recover (Decision 4)
- [x] 4.6 Decide by experiment whether the uncertainty parameters are frozen during the annealing warmup or allowed to re-equilibrate; freezing is the conservative default (Decision 3), implemented via the 4.1 hook. Record which was chosen and why
- [x] 4.7 Confirm the shared `train()`'s optimizer covers every model parameter exactly once, with the log-variances at `LOG_VAR_LR_MULTIPLIER` times the caller's `learning_rate` and every other parameter at that rate, and with no weight decay in any group (Decision 4)
- [x] 4.8 Extend the shared `TrainingHistory` with per-epoch `kl_weight` and the learned `log(sigma_r^2)` and `log(sigma_c^2)` values, reported through `training_step`'s metrics dict, alongside the existing `loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`, `accuracy`, and `wall_clock_seconds`; fields a model does not report stay empty lists
- [x] 4.9 Reduce `_Autoencoder.training_step` to reconstruction loss only (labels received and ignored)
- [x] 4.10 Add two no-op hooks to `_Model` in `model_api.py` and use them from the shared `train()` without model-type branching (Decision 6): `optimizer_param_groups(learning_rate)`, returning one group of all parameters at that rate by default, which `train()` passes to `torch.optim.Adam`; and `on_step_end()`, which `train()` calls after every `optimizer.step()`. Override both in `_VAE`: a second parameter group holding the two log-variances at `LOG_VAR_LR_MULTIPLIER = 10` times the rate (a provisional constant with a comment citing phase3c Decision 4), and the floor clamp from 4.5

## 5. Public API: `src/picoface/generator.py`

- [x] 5.1 Keep `build_vae(data)` recording `num_classes`/`class_names` from `data` (as 3b does); change `build_autoencoder(data)` to stop recording `num_classes`
- [x] 5.2 Re-export the shared `evaluate` and `predict` from `model_api` in `picoface.generator` and add them to `__all__`, so `generator.evaluate is classifier.evaluate`
- [x] 5.3 Update the module docstring: the VAE is one supervised model serving both workflows; drop the "autoencoder-to-VAE progression" framing
- [x] 5.4 Update `build_autoencoder()`'s docstring to state it is optional and not a prerequisite for `build_vae()`, and that it does not classify — remove the "pedagogical stepping stone" framing and 3b's "it also learns to classify" line (Decision 7)
- [x] 5.5 Update `build_vae()`'s docstring to describe classification through the latent rather than a parallel branch
- [x] 5.6 Confirm no `nn.Module` subclasses, loss functions, or training-loop code became importable from `picoface.generator`

## 6. Remove the latent-space visualization

- [x] 6.1 Remove `show_latent_space()` from `src/picoface/viz.py` and from its `__all__`, and remove its now-unused imports (`_latent_mean`, `Dataset`, `numpy`), leaving `plot_training_history()` untouched
- [x] 6.2 Remove `_latent_mean()` and the `"latent_mean"` entry in `_CAPABILITY_PHRASES` from `model_api.py`
- [x] 6.3 Remove `test_show_latent_space_smoke_and_ae_rejection` from `tests/test_generator.py` and `test_show_latent_space_still_plots_one_point_per_image_on_a_joint_model` from `tests/test_joint_model.py`

## 7. Tests

- [x] 7.1 Add `torch.manual_seed()` to tests that report or assert on accuracy, so recorded numbers are comparable across runs and assertions do not flake
- [x] 7.2 Add a held-out split helper: a second stub dataset generated with a different `seed`, used for every accuracy measurement
- [x] 7.3 Replace the three vacuous `assert 0.0 <= accuracy <= 1.0` assertions in `tests/test_classifier.py` with held-out accuracy above a loose bound
- [x] 7.4 Add a supervised end-to-end test: `build_vae(data)` → `train(model, data)` → `evaluate(model, held_out)` → `predict(model, image)` → `generate(model, n=5)`, confirming all four work from one trained model
- [x] 7.5 Add a labels-actually-matter test: train on shuffled labels and confirm held-out accuracy degrades relative to correct labels — the regression tripwire against a silently unsupervised training path
- [x] 7.6 Add a loose held-out accuracy assertion on the spatially-distinguished stub: above chance by a margin, bar set well below observed (Decision 8)
- [x] 7.7 Keep 3b's class-count mismatch test (`test_joint_model_class_count_mismatch_raises_shape_error`) narrowed to the VAE, confirming `ShapeError`; confirm the autoencoder does not raise on a class-count difference
- [x] 7.8 Add `evaluate()`/`predict()`-on-autoencoder tests, confirming `GeneratorError` (and that it is still caught by `except CapabilityError`); update 3b's `test_evaluate_rejects_a_model_without_classification` accordingly
- [x] 7.9 Add an annealing test: confirm the recorded `kl_weight` for the first epoch is below that of the final epoch, that the final epoch's `kl_weight` is 1, and that both hold for a short run (e.g. `epochs=3`) and a longer one — the schedule is relative to the requested epochs
- [x] 7.10 Add a diagnostics-completeness test: confirm `TrainingHistory` exposes per-epoch reconstruction, KL, classification, KL weight, and learned task weights for a VAE, and remains empty-listed for an autoencoder
- [x] 7.11 Re-validate the CPU time budget against the full combined objective — the existing 300-second ceiling was measured for reconstruction + KL alone
- [x] 7.12 Confirm the shape-agnosticism test still passes for both model types across differing image shapes and class counts
- [x] 7.13 Confirm the autoencoder path still trains with labels present but ignored
- [x] 7.14 Reconcile Phase 3b's `tests/test_joint_model.py` with this change. Narrow to the VAE (drop the autoencoder parametrization): classifies-above-chance (superseded by 7.4/7.6 on held-out data), predict-returns-class-name, records-num-classes, evaluate-is-deterministic, classify-is-differentiable, plot-shows-accuracy. Update: VAE/autoencoder history-series tests to the new field sets (autoencoder: reconstruction only). Keep unchanged: shape-mismatch, one-handler-catches-both-arms, generate-rejects, train-rejects-non-models, same-function-from-both-arms (extend to `evaluate`/`predict` from `picoface.generator`), signature tests, shape-agnosticism, wall-clock ceiling (fold into 7.11)
- [x] 7.15 Re-check 3b's `test_joint_vae_reconstruction_loss_decreases_with_classification_branch` under annealing: reconstruction may rise as the KL weight ramps, so assert on the warmup window or on first-vs-best rather than first-vs-last if needed, and record which
- [x] 7.16 Add hook tests: `optimizer_param_groups` on every model kind covers each parameter exactly once; the VAE's log-variances are in a group at `LOG_VAR_LR_MULTIPLIER` times the given rate; and after `train()`, each stored log-variance is at or above `LOG_VAR_FLOOR`, including when a log-variance is set below the floor before a step

## 8. Diagnostics record for Phase 6

- [x] 8.1 Produce a recorded table from a seeded run on the spatially-distinguished stub: held-out accuracy, final reconstruction loss, final KL, final classification loss, learned task weights per epoch, wall-clock seconds, and the `LATENT_DIM` used
- [x] 8.2 Record the same table for at least two latent dimensionalities, so Phase 6 inherits a trend rather than a single point
- [x] 8.3 Note in the record whether the learned weights showed the starvation dynamic from Decision 4, whether the `log(sigma^2)` floor was reached, and whether each run's learned weights had reached equilibrium with their task losses by the final epoch (Decision 4: at 10×, short stub runs will not have)
- [x] 8.4 Commit this record where Phase 6 will find it (change directory or `openspec/ROADMAP.md` reference), and state plainly that these are stub-dataset numbers, not predictions about real content

## 9. Documentation

- [x] 9.1 Update `openspec/ROADMAP.md`'s three-arm description: Arms 1 and 2 are no longer independent student-facing paths — the supervised VAE is the student's model, the CNN is the independent validator
- [x] 9.2 Add a Phase 3c row to the phase table and a Phase 3c section describing this change
- [x] 9.3 Clarify Phase 4's scope in the roadmap: the classification capability now ships with the model rather than after it, leaving Phase 4 the linkage exercise (`classify_generated()` scored by the independent CNN, `activation_maximize()`)
- [x] 9.4 Resolve the `latent_dim=2` open question (unpinned) and the beta open question (the KL weight is now the ELBO weight of 1 on a per-pixel scale, reached by annealing); restate what remains open as the annealing schedule's length/shape, the `log(sigma^2)` floor, the log-variance learning-rate multiplier (conservatively 10×; retune once real steps per epoch are known), and whether to add weight decay (decoupled `AdamW`, excluding the log-variances; Decision 4)
- [x] 9.5 Update the Phase 3 bullet list in the roadmap, which still states `latent_dim` is fixed at 2 and beta is a fixed constant
- [x] 9.6 Record the Kendall-weighting starvation dynamic (Decision 4) in the roadmap's Risks section, so it is visible to Phase 6 rather than living only in this change's design
- [x] 9.7 Update the roadmap's stub-dataset risk entry to reflect that the "deliberately non-trivial synthetic class" mitigation is now implemented
- [x] 9.8 Confirm `README.md` needs no change — no student-facing claim it makes is affected
- [x] 9.9 Mark Phase 3b's superseded model decisions in the roadmap: the Arm 2 description ("classification branch ... added in Phase 3b"), the "Joint classifier branch sits on the shared conv-trunk features" decision, the Phase 3b section's head-placement/λ/autoencoder text, the Phase 4 note that the judge may be "a joint model's own head", the Phase 6 revisit list's λ and head-placement items, and the Risks entries on the 2D latent plot and the trunk head's accuracy gap. Keep 3b's `model-interface` entries — that refactor stands (Decision 6)
- [x] 9.10 Update the phase table's Phase 3b row from "In progress" to archived
- [x] 9.11 Update the `## Purpose` paragraph of `openspec/specs/shape-generator/spec.md`, which still describes "an autoencoder-to-VAE progression, plus latent-space visualization" — spec deltas cannot change it, so edit it directly: the capability is the supervised VAE (classifies and generates) plus an optional reconstruction-only autoencoder. In the same edit, rename the `build_vae()` scenario "The swap adds only the probabilistic latent" to "The swap adds the probabilistic latent and classification" — its body was rewritten by this change's delta, but OpenSpec deltas cannot rename a scenario, so the old name is carried until then. (Purpose paragraph updated during apply; the scenario rename must wait until after archive, because the delta still carries the old name and renaming it in the main spec first would break the archive merge.)

## 10. Verification

- [x] 10.1 Run the full test suite on CPU and confirm it passes
- [x] 10.2 Run `openspec validate picoface-phase3c --strict` and resolve any findings
- [x] 10.3 Confirm no new dependency entered `pyproject.toml` — removing the latent plot removed the reason the projection library was ever a consideration
- [x] 10.4 Confirm the public surface of `picoface.generator` and `picoface.viz` matches what the specs describe, with no leaked internals
