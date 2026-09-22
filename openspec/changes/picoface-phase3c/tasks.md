## 1. Loss scale normalization (do this first)

Decision 5 requires this to land before annealing or uncertainty weighting, so the learned parameters cannot silently absorb a resolution-scaling artifact.

- [ ] 1.1 In `_train_loop`, change the reconstruction term from per-pixel mean reduction to a per-image scale (sum over pixels, mean over batch), matching the KL term's existing per-image reduction
- [ ] 1.2 Record the pre-change and post-change loss magnitudes on the stub dataset, and re-derive the KL weight that reproduces the current effective regularization (`BETA = 0.01` at 16x16x3 is roughly equivalent to a conventional beta near 7.7) — this becomes the starting point for the annealing ceiling in 4.2
- [ ] 1.3 Delete the now-incorrect `BETA` comment block in `generator_internals.py` explaining the value as "deliberately kept small"; its reasoning compared beta against the mean-reduced reconstruction scale rather than the per-image scale the KL term lives on
- [ ] 1.4 Confirm on two differently-sized stub datasets that the reconstruction/KL balance no longer varies with `H*W*C`

## 2. Stub dataset: spatially-distinguished classes

- [ ] 2.1 Extend `make_stub_dataset()` with a mode producing simple filled figures (e.g. square, circle, cross) rendered with numpy, positioned to fill comparable image area
- [ ] 2.2 Normalize the rendered classes to matched mean brightness, so no single global statistic separates them
- [ ] 2.3 Keep the existing brightness-separated mode available and selectable, for plumbing and shape-agnosticism tests that do not need difficulty
- [ ] 2.4 Add a test asserting that classifying the spatial classes by per-image mean brightness alone performs no better than chance, while the classes remain recoverable — this is the guard that keeps the dataset honest
- [ ] 2.5 Confirm the generator still honors arbitrary `height`/`width`/`channels`/`class_names` and remains non-public

## 3. Internals: the supervised VAE model

- [ ] 3.1 Revalue `LATENT_DIM` above 2 in `generator_internals.py`, record the value tried and its effect in the diagnostics table (final value is a Phase 6 decision)
- [ ] 3.2 Add a classification head to `_VAE`: a small linear stack from `latent_dim` to `num_classes`, reading the reparameterized sample `z` — the same tensor the decoder consumes (Decision 1)
- [ ] 3.3 Thread `num_classes` through `_build_vae()` and `_VAE.__init__`; store `class_names` on the model as `build_classifier()` does
- [ ] 3.4 Extend `_VAE.forward()` to return class logits alongside the reconstruction, `mu`, and `logvar`
- [ ] 3.5 Update `_validate_decode_shape()` for the widened forward-pass return signature, keeping the build-time dummy pass-through intact
- [ ] 3.6 Add an internal inference helper that returns class logits from `mu` rather than a sample, so `evaluate()`/`predict()` are deterministic (Decision 1); repurpose `_encode_mu` for this rather than deleting it
- [ ] 3.7 Leave `_Autoencoder`, `_Encoder`, and the shared conv trunk unchanged

## 4. Internals: training loop

- [ ] 4.1 Change `_train_loop` to build a `TensorDataset(images, labels)` and to accept labels, with a three-way dispatch: autoencoder (reconstruction only, labels ignored), VAE (reconstruction + annealed KL + weighted classification, labels required)
- [ ] 4.2 Implement the KL annealing schedule: ramp the KL weight from zero to a ceiling across training, as an internal function of epoch with no student-facing parameter
- [ ] 4.3 Add learned homoscedastic uncertainty parameters for the two task losses (reconstruction, classification) as `nn.Parameter` log-variances on the model, combined as `(1 / (2 * sigma^2)) * L + log(sigma)` per Kendall et al. (2018)
- [ ] 4.4 Do NOT apply uncertainty weighting to the KL term (Decision 3) — it is governed by 4.2's schedule alone
- [ ] 4.5 Add a floor on the learned `log(sigma^2)` values as the mitigation lever for the `1 / (2 * L)` starvation dynamic (Decision 4); leave its value provisional and record it
- [ ] 4.6 Decide by experiment whether the uncertainty parameters are frozen during the annealing warmup or allowed to re-equilibrate; freezing is the conservative default (Decision 3). Record which was chosen and why
- [ ] 4.7 Ensure the uncertainty parameters are registered with the optimizer alongside the model's other parameters
- [ ] 4.8 Extend `TrainingHistory` with per-epoch `classification_loss`, `kl_weight`, and the learned task weights, alongside the existing `loss`, `reconstruction_loss`, `kl_loss`, and `wall_clock_seconds`

## 5. Public API: `src/picoface/generator.py`

- [ ] 5.1 Update `build_vae(data)` to derive `num_classes` from `data.class_names` and pass it through; keep the `build_vae(data)` call shape unchanged
- [ ] 5.2 Add a class-count consistency check to `train()` for VAE models, raising the generator arm's `ShapeError`, mirroring `_check_class_count()` in `classifier.py`
- [ ] 5.3 Pass `data.labels` into `_train_loop`; set `model.class_names` on the way through, as `classifier.train()` does
- [ ] 5.4 Add `evaluate(model, data) -> float` for VAE models, returning accuracy in `[0, 1]`, raising `GeneratorError` for autoencoder models
- [ ] 5.5 Add `predict(model, image) -> str` for VAE models, returning a class *name*, raising `GeneratorError` for autoencoder models
- [ ] 5.6 Update `__all__` and the module docstring: the VAE is one supervised model serving both workflows, not a generation-only model
- [ ] 5.7 Update `build_autoencoder()`'s docstring to state it is optional and not a prerequisite for `build_vae()` — remove the "pedagogical stepping stone" framing (Decision 7)
- [ ] 5.8 Confirm no `nn.Module` subclasses, loss functions, or training-loop code became importable from `picoface.generator`

## 6. Remove the latent-space visualization

- [ ] 6.1 Remove `show_latent_space()` from `src/picoface/viz.py` and from its `__all__`
- [ ] 6.2 Remove the now-unused imports in `viz.py` (`_encode_mu`, `Dataset`, `GeneratorError`, `ShapeError`, `numpy`), leaving `plot_training_history()` untouched
- [ ] 6.3 Remove `test_show_latent_space_smoke_and_ae_rejection` from `tests/test_generator.py`
- [ ] 6.4 Confirm `_encode_mu` survives in internals for its new inference role (3.6) rather than being removed with its former caller

## 7. Tests

- [ ] 7.1 Add `torch.manual_seed()` to tests that report or assert on accuracy, so recorded numbers are comparable across runs and assertions do not flake
- [ ] 7.2 Add a held-out split helper: a second stub dataset generated with a different `seed`, used for every accuracy measurement
- [ ] 7.3 Replace the three vacuous `assert 0.0 <= accuracy <= 1.0` assertions in `tests/test_classifier.py` with held-out accuracy above a loose bound
- [ ] 7.4 Add a supervised end-to-end test: `build_vae(data)` → `train(model, data)` → `evaluate(model, held_out)` → `predict(model, image)` → `generate(model, n=5)`, confirming all four work from one trained model
- [ ] 7.5 Add a labels-actually-matter test: train on shuffled labels and confirm held-out accuracy degrades relative to correct labels — the regression tripwire against a silently unsupervised training path
- [ ] 7.6 Add a loose held-out accuracy assertion on the spatially-distinguished stub: above chance by a margin, bar set well below observed (Decision 8)
- [ ] 7.7 Add a class-count mismatch test for the generator's `train()`, confirming `ShapeError`
- [ ] 7.8 Add `evaluate()`/`predict()`-on-autoencoder tests, confirming `GeneratorError`
- [ ] 7.9 Add an annealing test: confirm the recorded `kl_weight` for the first epoch is below that of the final epoch
- [ ] 7.10 Add a diagnostics-completeness test: confirm `TrainingHistory` exposes per-epoch reconstruction, KL, classification, KL weight, and learned task weights for a VAE, and remains empty-listed for an autoencoder
- [ ] 7.11 Re-validate the CPU time budget against the full combined objective — the existing 300-second ceiling was measured for reconstruction + KL alone
- [ ] 7.12 Confirm the shape-agnosticism test still passes for both model types across differing image shapes and class counts
- [ ] 7.13 Confirm the autoencoder path still trains with labels present but ignored

## 8. Diagnostics record for Phase 6

- [ ] 8.1 Produce a recorded table from a seeded run on the spatially-distinguished stub: held-out accuracy, final reconstruction loss, final KL, final classification loss, learned task weights per epoch, wall-clock seconds, and the `LATENT_DIM` used
- [ ] 8.2 Record the same table for at least two latent dimensionalities, so Phase 6 inherits a trend rather than a single point
- [ ] 8.3 Note in the record whether the learned weights showed the starvation dynamic from Decision 4, and whether the `log(sigma^2)` floor was reached
- [ ] 8.4 Commit this record where Phase 6 will find it (change directory or `openspec/ROADMAP.md` reference), and state plainly that these are stub-dataset numbers, not predictions about real content

## 9. Documentation

- [ ] 9.1 Update `openspec/ROADMAP.md`'s three-arm description: Arms 1 and 2 are no longer independent student-facing paths — the supervised VAE is the student's model, the CNN is the independent validator
- [ ] 9.2 Add a Phase 3c row to the phase table and a Phase 3c section describing this change
- [ ] 9.3 Clarify Phase 4's scope in the roadmap: the classification capability now ships with the model rather than after it, leaving Phase 4 the linkage exercise (`classify_generated()` scored by the independent CNN, `activation_maximize()`)
- [ ] 9.4 Resolve the `latent_dim=2` open question (unpinned) and restate the beta open question as the annealing ceiling plus the uncertainty-weight floor
- [ ] 9.5 Update the Phase 3 bullet list in the roadmap, which still states `latent_dim` is fixed at 2 and beta is a fixed constant
- [ ] 9.6 Record the Kendall-weighting starvation dynamic (Decision 4) in the roadmap's Risks section, so it is visible to Phase 6 rather than living only in this change's design
- [ ] 9.7 Update the roadmap's stub-dataset risk entry to reflect that the "deliberately non-trivial synthetic class" mitigation is now implemented
- [ ] 9.8 Confirm `README.md` needs no change — no student-facing claim it makes is affected

## 10. Verification

- [ ] 10.1 Run the full test suite on CPU and confirm it passes
- [ ] 10.2 Run `openspec validate picoface-phase3c --strict` and resolve any findings
- [ ] 10.3 Confirm no new dependency entered `pyproject.toml` — removing the latent plot removed the reason the projection library was ever a consideration
- [ ] 10.4 Confirm the public surface of `picoface.generator` and `picoface.viz` matches what the specs describe, with no leaked internals
