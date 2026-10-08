## Why

The model code in `_internals/` works, but it is hard to read. `_VAE` holds encoding, sampling, the classification head, the loss, and the learned task weights all at once. Layers are wired together by hand from module-level constants that exist only so the shape sums stay in sync. Long design-history comments push the code itself off the screen. This change makes the code easier for a person to follow. The model's design and behaviour stay exactly as they are.

## What Changes

- **The VAE and autoencoder are built from small named parts.** These are a conv encoder, a decoder, a classification head, a reparameterization step, and a module for the learned task weights. `_VAE` and `_Autoencoder` assemble those parts. Their `forward()` methods read as the data flow.
- **Fixed layer stacks become `nn.Sequential` pipelines.** Each pipeline is listed in data-flow order and built with small local block helpers. Layer settings such as kernel size and stride are written next to the layer, not held in module-level constants.
- **Shapes are measured, not worked out by hand.** The encoder measures its output shape with a dummy forward pass, as the classifier already does, and the decoder receives that shape from the encoder. The hand-written shape helpers go.
- **The loss reads as an equation.** The KL divergence, the KL schedule, and the uncertainty weighting become named helpers, so `training_step()` is a few lines. The VAE's multi-part output becomes a named tuple.
- **The classifier gets the same treatment.** Its conv/pool stack and fully connected head become `Sequential` pipelines. Its minimum-input-size check still has a single source of truth.
- **Comments are pruned.** Multi-paragraph rationale and history comments shrink to a short note or a one-line pointer to the OpenSpec change or decision that records the reasoning.
- **Unchanged:** every public function, every error type and message, every tuned value (latent size, channel widths, KL ramp, log-variance floor and learning-rate multiplier), the training hooks on the model, and the leading-underscore naming of internal names. With the same `torch.manual_seed`, a model builds with identical initial weights and trains to identical losses before and after the change.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

(none) This is a pure refactor with no change in behaviour. The change sets `skip_specs: true`.

## Impact

- `src/picoface/_internals/generator_internals.py`: restructured into named components, with comments pruned.
- `src/picoface/_internals/classifier_internals.py`: restructured the same way, with comments pruned.
- `src/picoface/_internals/model_api.py`, `linkage_internals.py`, public modules: unchanged. The abstract `_Model` interface and the capability methods they call (`classify`, `encode_mu`, `decode`, `sample`, `training_step`, and the hooks) keep their names and signatures.
- Tests: unchanged. `tests/test_joint_model.py` reads `model.reconstruction_log_var`, `model.classification_log_var`, `LOG_VAR_FLOOR`, and `LOG_VAR_LR_MULTIPLIER`. All four stay reachable under the same names.
- `state_dict` key names change (for example `trunk.conv1.weight` becomes a `Sequential` index). Nothing in the project saves or loads weights, so no stored artifacts are affected.
- No new dependencies.
