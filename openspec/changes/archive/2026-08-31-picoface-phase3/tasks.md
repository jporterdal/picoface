## 1. Internals: encoder/decoder and training loop

- [x] 1.1 Create `src/picoface/_internals/generator_internals.py` with a shared `_build_encoder(input_shape, latent_dim)` / `_build_decoder(latent_dim, output_shape)` pair (conv→ReLU stride-2 blocks down, transpose-conv/upsample blocks back up, sigmoid output activation) and a module-level `LATENT_DIM = 2` constant
- [x] 1.2 Implement `_build_autoencoder(input_shape)`: connects encoder trunk directly to decoder, validates the decode output shape matches `input_shape` via a dummy pass-through (`torch.zeros(1, C, H, W)`), raising `ShapeError` naming the mismatch if it doesn't
- [x] 1.3 Implement `_build_vae(input_shape)`: connects the same encoder trunk to `mu`/`logvar` linear heads, adds the reparameterization sampling step, then the shared decoder; same dummy-pass-through shape validation as 1.2
- [x] 1.4 Implement a shared internal preprocessing helper (NHWC uint8 → NCHW float32, `/255.0` normalization) local to this module — do not import the classifier arm's equivalent
- [x] 1.5 Add a module-level `BETA` constant (KL-divergence weight) for VAE loss, tuned qualitatively against stub-dataset reconstructions during implementation
- [x] 1.6 Implement an internal training-loop function dispatching on model type: MSE reconstruction loss only (AE) vs. `reconstruction_loss + BETA * kl_divergence` (VAE); DataLoader batching, Adam optimizer, CPU device, per-epoch loss + wall-clock timing
- [x] 1.7 Add a `TrainingHistory` dataclass (per-epoch `loss`, `wall_clock_seconds`, plus per-epoch `reconstruction_loss`/`kl_loss` populated only for VAE models)
- [x] 1.8 Implement an internal `_encode_mu(model, images) -> np.ndarray` helper returning the encoder's mean output (not a sampled `z`), for use by `show_latent_space()`
- [x] 1.9 Implement an internal sampling helper for `generate()`: draw `n` vectors from `N(0, I)` in the fixed 2D latent space and run them through the decoder

## 2. Public API: `src/picoface/generator.py`

- [x] 2.1 Define `ShapeError(ValueError)`: generator-arm-local exception (not imported from `picoface.classifier`) for build-time and use-time shape mismatches
- [x] 2.2 Define `GeneratorError(ValueError)`: raised when a function requiring a `build_vae()` model is given a `build_autoencoder()` model, with a message naming the mismatch and pointing at `build_vae()`
- [x] 2.3 Implement `build_autoencoder(data)`: derives `input_shape` from the given `Dataset`, delegates to `_build_autoencoder`
- [x] 2.4 Implement `build_vae(data)`: derives `input_shape` from the given `Dataset`, delegates to `_build_vae`; same call shape as `build_autoencoder(data)`
- [x] 2.5 Implement `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3) -> TrainingHistory`: works for both AE and VAE models built above, delegating loss selection to the internal training loop
- [x] 2.6 Implement `generate(vae_model, n) -> np.ndarray`: raises `GeneratorError` if `vae_model` was not built by `build_vae()`; otherwise samples and decodes `n` new images
- [x] 2.7 Verify no `nn.Module` subclasses, loss functions, or training-loop code are importable from `picoface.generator` itself

## 3. Visualization

- [x] 3.1 Add `show_latent_space(vae_model, data)` to `src/picoface/viz.py`: encodes `data.images` via `_encode_mu`, scatter-plots the resulting 2D points colored by `data.labels`/`class_names`; raises `GeneratorError` if given a non-VAE model

## 4. Tests

- [x] 4.1 Add `tests/test_generator.py`: `build_autoencoder(data)` → `train` → confirm reconstructions run end to end without error, against `make_stub_dataset()`
- [x] 4.2 Add a VAE end-to-end test: `build_vae(data)` → `train` → `generate(model, n=5)` → confirm 5 images of the correct shape/dtype are returned
- [x] 4.3 Add a "swap the build call" test: confirm the same `train(model, data, ...)` call succeeds unchanged for both a `build_autoencoder()` and a `build_vae()` model
- [x] 4.4 Add a `generate()`-on-AE-model test: confirm `GeneratorError` is raised, naming the AE/VAE mismatch
- [x] 4.5 Add a `show_latent_space()` test: confirm it runs without error on a trained VAE model and produces one point per input image (non-interactive matplotlib backend); confirm it raises `GeneratorError` on an AE model
- [x] 4.6 Add a VAE training-time regression test: assert `TrainingHistory.wall_clock_seconds` stays under a generous ceiling (e.g. under 5 minutes, per the spec's budget) on the default-sized stub dataset — no equivalent assertion for the AE path
- [x] 4.7 Add a shape-agnosticism test: build/train (both AE and VAE) against two differently-shaped stub datasets to confirm no dimension is hardcoded
- [x] 4.8 Add a decode-shape-mismatch test: confirm `ShapeError` is raised at build time if a deliberately-broken encoder/decoder pairing would produce a decoded shape mismatch (e.g. via a temporarily misconfigured internal constant in the test, or a degenerately small `input_shape`)
- [x] 4.9 Add a reconstruction-sanity test: after training a VAE on the stub dataset, assert reconstruction loss has decreased from its first-epoch value (regression tripwire against a silently broken loss/training loop)

## 5. Verification

- [x] 5.1 Run the full test suite (`pytest`) and confirm it passes on a CPU-only environment
- [x] 5.2 Manually inspect a few VAE reconstructions and latent-space plots against the stub dataset; record the qualitative outcome and the chosen `BETA` value's rationale in the implementation notes, per the roadmap's flag that this value needs later validation against real data (Phase 6)

  **Implementation notes:** `BETA = 0.01` (see rationale comment at `generator_internals.BETA`). Compared against `BETA` of 0.1 and 1.0 on the stub dataset: higher values drove the KL term toward collapse (near zero) faster/further without improving reconstruction loss, so 0.01 was kept as a light regularizer. Reconstructions after default (10-epoch) training are visibly close to each class's mean color/pattern but retain a soft, blurry quality (expected: MSE reconstruction loss + a 2D latent bottleneck can't reproduce per-pixel noise, and shouldn't try to). `show_latent_space()` plots produced one distinct, separated cluster per class on the stub dataset (classes are trivially separable by construction). Both are consistent with the roadmap's already-flagged "output may look too blurry to feel motivating" risk and are expected to need retuning once real (Phase 5) data exists — deferred to Phase 6 per the design's Non-Goals.
- [x] 5.3 Manually record observed wall-clock VAE training time against the stub dataset, continuing the roadmap's "measure from Phase 2 onward" mitigation

  **Observed:** ~0.2s wall-clock for a full default `train()` call (10 epochs, `n_per_class=8`, 16x16x3 stub images) on this CPU-only dev machine — far under the 5-minute budget, consistent with Phase 2's classifier-arm timings at this dataset size.
