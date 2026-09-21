> **Ordering:** §1 is a behavior-preserving refactor and is a **prerequisite for every other section** (§2–§6). Do not start §2 until §1.9's gate passes. Each later section restates this dependency.

## 1. Refactor onto an abstract model interface (PREREQUISITE — no behavior change)

- [ ] 1.1 Baseline: run the full existing test suite (`pytest`) before touching any code and record the pass count and the two wall-clock-ceiling test results; verify all pass. Also record each existing model's `state_dict` keys and tensor shapes (CNN, AE, VAE on the stub dataset) to a scratch file, for the architecture-unchanged check in 1.4/1.5
- [ ] 1.2 Add `src/picoface/_internals/errors.py` with `BaseShapeError(ValueError)`, `CapabilityError(ValueError)`, and `GeneratorError(CapabilityError)`; make `classifier.ShapeError` and `generator.ShapeError` subclass `BaseShapeError`; have `generator.py` import (not redefine) `GeneratorError`; re-export `BaseShapeError` and `CapabilityError` from both arm modules. Verify: existing tests still pass, and `issubclass` checks for each relationship hold
- [ ] 1.3 Add `src/picoface/_internals/model_api.py` with: the abstract `_Model(nn.Module)` (required `training_step(inputs, labels) -> (loss, metrics)`, default `check_data(data)`, `capabilities` frozenset, optional `classify`/`sample`/`encode_mu` raising `NotImplementedError` by default, stored `shape_error_cls`); the single shared preprocessing helper (NHWC uint8 → NCHW float32 /255, shape validated against `model.input_shape`); and the unified `TrainingHistory` (`loss`, `wall_clock_seconds`, `reconstruction_loss`, `kl_loss`, `classification_loss`, `accuracy`; list fields default empty). Verify: module imports cleanly and imports nothing from `picoface.classifier`/`picoface.generator`/`picoface.viz`
- [ ] 1.4 Refactor `_CNNClassifier` onto `_Model`: `capabilities = {"classify"}`, `classify(x)` = the current feature+FC path, `training_step` = cross-entropy (reporting `loss` only), store `shape_error_cls`; leave architecture and constants untouched. Verify: `tests/test_classifier.py` passes unmodified, and the CNN's `state_dict` keys/shapes match the 1.1 record exactly
- [ ] 1.5 Refactor `_Autoencoder` and `_VAE` onto `_Model` **without adding classification yet**: AE `training_step` = MSE, VAE `training_step` = MSE + BETA·KL reporting `reconstruction_loss`/`kl_loss`; VAE `capabilities = {"sample", "latent_mean"}` with `sample`/`encode_mu` methods (replacing `_sample_generate`/`_encode_mu`), AE `capabilities = {}`; remove `is_variational`. Verify: `tests/test_generator.py` passes unmodified, and both models' `state_dict` keys/shapes match the 1.1 record
- [ ] 1.6 Implement the one shared `train(model, data, epochs, batch_size, learning_rate)` in `model_api.py` (`check_data`, set `model.class_names`, shared preprocessing, DataLoader/Adam/CPU, `model.training_step`, per-epoch metric averaging, wall-clock timing; non-model input rejected with a clear error), replacing both `_train_loop`s, then delete them. Verify: both existing test files pass unmodified and both wall-clock-ceiling tests still pass
- [ ] 1.7 Implement shared `evaluate`, `predict`, and `generate` in `model_api.py` (`evaluate`/`predict` wrap `classify` in `no_grad()` + `eval()`; `generate` requires the `sample` capability, raising `GeneratorError` otherwise; a missing `classify` raises `CapabilityError` naming the verb). Verify: existing classifier tests (`evaluate`/`predict`) pass unmodified, and existing `generate()`-on-AE test still raises `GeneratorError`
- [ ] 1.8 Make `classifier.py` re-export `train`, `evaluate`, `predict` and `generator.py` re-export `train`, `generate` (both in `__all__`); repoint `viz.py` at the unified `TrainingHistory` and key `show_latent_space`'s guard on the `latent_mean` capability (still raising `GeneratorError`); remove the now-dead `_forward`, `_preprocess`, `_encode_mu`, `_sample_generate`, and per-arm `TrainingHistory`. Verify: `classifier.train is generator.train`, and no `nn.Module` subclass, loss function, or loop code is importable from either public module
- [ ] 1.9 **Gate:** run the full existing test suite; verify every pre-existing test passes with **no modification to any existing test** (`git diff` shows only additions under `tests/`), both wall-clock ceilings pass, and the recorded `state_dict` shapes are unchanged. §2–§6 must not begin until this passes

## 2. Joint classification branch (DEPENDS ON §1)

- [ ] 2.1 Add `CLASSIFICATION_WEIGHT = 1.0` in `generator_internals.py` beside `BETA`, with a comment recording the rationale (head on trunk; spike showed accuracy/recon insensitive to λ in 0.1–10; not validated on real data; flagged for Phase 6). Verify: constant present and referenced by both AE and VAE `training_step`
- [ ] 2.2 Add a shared `_build_classification_head(flatten_dim, num_classes)` (`Linear(flatten_dim, 32) → ReLU → Linear(32, num_classes)`) and attach it to the trunk features of both models; restructure `_Autoencoder` to hold `trunk`, `to_latent`, `decoder`, `head` directly (replacing `_Encoder`); `_build_decoder` unchanged. Verify: `_validate_decode_shape` still passes for both models on the existing shape-agnostic stub shapes
- [ ] 2.3 `build_autoencoder(data)` / `build_vae(data)` (signatures unchanged) derive `num_classes` and `class_names` from `data` and store them on the model, along with `shape_error_cls`. Verify: built models expose `num_classes == len(data.class_names)` and matching `class_names`
- [ ] 2.4 Implement `classify(x)` on both models as trunk → head only (no decoder, no sampling; no internal `no_grad`/mode toggling) and add `"classify"` to both `capabilities`. Verify: output shape is `(N, num_classes)` and a gradient flows back to the input pixels
- [ ] 2.5 Update `training_step`: AE = `MSE + CLASSIFICATION_WEIGHT·CE`; VAE = `MSE + BETA·KL + CLASSIFICATION_WEIGHT·CE`; report `reconstruction_loss`, `kl_loss` (VAE only), `classification_loss`, and `accuracy`. Update the models' `forward` to also return logits (VAE: `(recon, mu, logvar, logits)`; AE: `(recon, logits)`). Verify: after `train()`, a VAE history has five equal-length per-epoch series and an AE history has empty `kl_loss`

## 3. Visualization (DEPENDS ON §1; accuracy series from §2)

- [ ] 3.1 Update `plot_training_history` to return a two-panel figure (loss, accuracy) when `history.accuracy` is non-empty and the existing single-axes figure otherwise. Verify: the existing classifier smoke test passes unmodified (one axes), and a new test on a joint-model history sees two axes
- [ ] 3.2 Confirm `show_latent_space()` on a trained joint VAE still plots one point per image with no projection step, and still raises `GeneratorError` for an AE model. Verify: existing latent-space test passes unmodified

## 4. Tests (DEPENDS ON §1–§3)

- [ ] 4.1 Add a joint-VAE end-to-end test: `build_vae(data)` → `train` → `evaluate(model, data)` (imported from `picoface.classifier`) is above chance (`1/num_classes`) on the stub dataset, with seeds fixed; same for `build_autoencoder`
- [ ] 4.2 Add `predict` tests on both joint models: returns a member of `data.class_names`
- [ ] 4.3 Add mismatch tests on joint models: class-count mismatch in `evaluate()` and image-shape mismatch in `predict()` raise the shape error, and both a CNN mismatch and a joint-model mismatch are caught by `BaseShapeError`
- [ ] 4.4 Add a history test: VAE history has per-epoch `loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`, `accuracy` (accuracy values in [0, 1]); CNN history has empty reconstruction/KL lists
- [ ] 4.5 Add a "joint training doesn't degrade reconstruction" test: VAE `reconstruction_loss[-1] < reconstruction_loss[0]` with the branch active (mirrors the existing reconstruction test)
- [ ] 4.6 Add capability-error tests: `generate(cnn_model, 5)` and `generate(ae_model, 5)` both raise `GeneratorError`, both caught by a single `CapabilityError` handler; `train(object(), data)` raises a clear error (not `AttributeError`)
- [ ] 4.7 Add identity/interface tests: `classifier.train is generator.train`; `evaluate` is deterministic across two calls and leaves parameters unchanged; `classify` is differentiable w.r.t. input pixels
- [ ] 4.8 Add a signature test: `build_autoencoder`, `build_vae`, and `train` accept no classification-weight parameter
- [ ] 4.9 Add joint-model shape-agnosticism tests (two differently shaped/class-count stub datasets, AE and VAE), and confirm both wall-clock-ceiling tests still pass with the branch active

## 5. Roadmap (DEPENDS ON §1–§2; may be done in parallel with §4)

- [ ] 5.1 Update `openspec/ROADMAP.md`: fix the stale phase-status table (Phases 2 and 3 → Done, with archive paths) and add a Phase 3b row for this change
- [ ] 5.2 Update the Three-Arm Architecture / Arm 2 description and the Phase 3 text to say Arm 2's model also classifies, and note the Arm 1 curriculum question (whether Arm 1 remains a separate student stage) as a new Phase 7 open question
- [ ] 5.3 Add to Phase 4: `classify_generated()` will be uninformative until β is tuned on real data, and the draft "intended class" wording does not fit unconditional `generate()` (report predicted-class distribution instead, or revisit a class-conditional decoder)
- [ ] 5.4 Add to Phase 6's revisit list and Open Questions: `CLASSIFICATION_WEIGHT`, the head architecture/placement, and β's heavy-regularization effect on reconstruction; add risks for "latent plot won't show class clusters" and "joint head weaker than standalone CNN"
- [ ] 5.5 Verify by re-reading the roadmap that the status table, phase descriptions, risks, and open questions are mutually consistent

## 6. Verification (DEPENDS ON §1–§5)

- [ ] 6.1 Run the full test suite (`pytest`) and confirm it passes on CPU
- [ ] 6.2 Run `openspec validate picoface-phase3b-joint-model --strict` and confirm it passes
- [ ] 6.3 Manually train a `build_vae()` model on the stub dataset and record in the implementation notes: held-out-style `evaluate()` accuracy, final reconstruction loss versus a pre-change baseline, and wall-clock time, continuing the roadmap's measure-from-Phase-2 mitigation
- [ ] 6.4 Confirm the design's stated non-goals hold in code: no class argument on `generate()`, no classification-weight parameter, and no head attached to `mu`/`z`
