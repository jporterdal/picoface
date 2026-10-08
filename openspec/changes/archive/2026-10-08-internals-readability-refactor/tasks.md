## 1. Baseline capture

- [x] 1.1 Write an uncommitted comparison script, outside the repo or in a gitignored path, as design.md Decision 6 describes. For `_build_vae`, `_build_autoencoder`, and `_build_classifier`, at 16×16×3 and one other valid shape, under fixed seeds, it saves: each model's `parameters()` list after building, one `training_step` loss and metrics dict, and VAE `sample(4)` output. It also saves the error type and message each builder raises for an invalid shape. Verify by running it on the current code and confirming the baseline file is written.
- [x] 1.2 Run the fast suite on the current code and record the pass count: `pytest tests/ --ignore=tests/test_real_dataset.py` (design.md, Decision 7). Verify it is all green before any edits.

## 2. Generator internals

- [x] 2.1 Replace `_ConvEncoderTrunk`, `_conv_output_size`, and `_encoded_feature_shape` with `_ConvEncoder`. It is a `Sequential` trunk built with a local `conv_block` helper, and it measures `feature_shape` and `flatten_dim` by a dummy forward pass. Verify that `pytest tests/test_generator.py tests/test_joint_model.py` passes.
- [x] 2.2 Rewrite `_Decoder` as a `Sequential` (Linear, `_Reshape`, `deconv_block`, final ConvTranspose2d, Sigmoid) that takes the encoder's `feature_shape`. Write the middle width inline with a pointer comment, and delete `_build_decoder`. Verify the same tests pass.
- [x] 2.3 Add `_reparameterize`, `_kl_divergence`, and `_VAEOutput`, and add `_LearnedTaskWeights` holding both log-variances, the two weighting formulas, and `clamp_to_floor()`. Verify that `pytest tests/test_joint_model.py` passes. It covers the floor clamp and the parameter groups.
- [x] 2.4 Reassemble `_VAE` and `_Autoencoder` from the new parts. Keep the layer creation and `parameters()` order from design.md Decision 6, and keep `reconstruction_log_var` / `classification_log_var` as read-only properties. Shorten `training_step()` to the loss equation plus metrics, and group the training hooks under one heading. Verify that `pytest tests/test_generator.py tests/test_joint_model.py` passes.
- [x] 2.5 Prune comments and docstrings in `generator_internals.py` following the style guide and the pointer table in design.md Decision 4. Verify that no comment block is longer than two lines except the docstrings that state a contract, and that every pointer path exists under `openspec/changes/`.

## 3. Classifier internals

- [x] 3.1 Rewrite `_CNNClassifier` with a `features` `Sequential` (conv → ReLU → pool, twice) and a `head` `Sequential` (Flatten → Linear → ReLU → Linear). Write the conv settings inline. Verify that `pytest tests/test_classifier.py` passes.
- [x] 3.2 Make `minimum_input_spatial_size()` and the pool layers derive from the same pool-count and pool-size constants, and remove the other constants nobody reads. Verify that the shape-rejection tests in `tests/test_classifier.py` and `tests/test_wrong_input.py` pass with unchanged messages.
- [x] 3.3 Prune comments and docstrings in `classifier_internals.py` following the style guide. Verify by reading the file against design.md's style guide.

## 4. Verification and cleanup

- [x] 4.1 Run the task 1.1 comparison script against the refactored code. Verify exact equality for every saved tensor, loss, metric, and error message.
- [x] 4.2 Run the fast suite from task 1.2 again. Verify the pass count matches task 1.2 and that `git diff --stat tests/` is empty. The full suite is deferred (design.md, Decision 7).
- [x] 4.3 Delete the comparison script and its baseline output. Verify that `git status` shows only the two `_internals` files as modified.
