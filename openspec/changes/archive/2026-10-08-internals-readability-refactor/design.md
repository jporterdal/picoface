## Context

See proposal.md for the motivation. Today's structure:

```
generator_internals.py                         classifier_internals.py
───────────────────────────────────────        ──────────────────────────────────
~10 module constants (_CONV_KERNEL_SIZE,       8 module constants (_CONV_*, _POOL_*,
  _ENC1_OUT_CHANNELS, ...) feeding both          _NUM_POOLS, channel and hidden widths)
  layers and the shape sums
_conv_output_size / _encoded_feature_shape     _CNNClassifier: conv1, conv2, pool, relu,
_ConvEncoderTrunk: conv1, conv2, relu            fc1, fc2 attributes, wired in
_Decoder: fc, deconv1, deconv2; computes the     _features() / classify()
  encoder's shape again from output_shape      minimum_input_spatial_size(): loop over
_VAE: trunk + fc_mu/fc_logvar + decoder +        _NUM_POOLS x _POOL_STRIDE
  head + two log-var Parameters + floor clamp
  + param groups + KL schedule + ~30-line
  training_step returning an unnamed 4-tuple
```

Constraints:

- Everything outside these two files must keep working without edits. That covers `model_api.py`, `linkage_internals.py`, the public modules, and the tests. The parts that reach into the internals are the `_Model` capability methods, `model.reconstruction_log_var` / `model.classification_log_var`, `LOG_VAR_FLOOR`, `LOG_VAR_LR_MULTIPLIER`, `LATENT_DIM`, `_build_vae`, `_build_autoencoder`, `_build_classifier`, and `minimum_input_spatial_size`.
- Behaviour must not change, down to the random numbers drawn (see Decision 6).

## Style guide

This is the target style for model code in `_internals/`. It applies to this change and to later model code there.

1. **Build a model from small named parts.** Each `nn.Module` has one job: encode, decode, sample, classify, or weight the losses. The top-level model's `__init__` lists the parts it assembles. Its `forward()` reads as the data flow between them.
2. **Write fixed layer stacks as `nn.Sequential`, in data-flow order.** Reading the `Sequential` from top to bottom should match the architecture diagram. Wire layers together by hand in `forward()` only when the flow branches or is reused.
3. **Repeated layer patterns get a small local builder.** Use a helper such as `conv_block(in_channels, out_channels)`, defined inside the `__init__` (or module) that uses it. It removes the repeated keyword arguments without hiding what a layer is.
4. **Write layer settings next to the layer.** For example: `kernel_size=3, stride=2, padding=1`. Use a module-level constant only for (a) a tuned value with a recorded history, or (b) a value that more than one place, or a test, must agree on.
5. **Measure shapes, don't work them out by hand.** Find a part's output shape with a dummy forward pass. Pass it on to the next part (the decoder takes the encoder's feature shape) rather than having each part re-derive it.
6. **Keep `forward()` short.** It should be a few lines that name their intermediate values (`mu`, `logvar`, `z`, `features`).
7. **Name the pieces of the loss.** Give each loss term a named helper that says what it computes. `training_step()` should then read like the loss equation, followed by its metrics.
8. **Return named tuples, not unnamed ones.** When a method returns several tensors, use a `NamedTuple`, so callers read `output.reconstruction` and not `output[0]`.
9. **Keep comments short and placed beside the code.** A comment says what is not obvious from the code, for example "log-variance, not standard deviation". Rationale and history become a one-line pointer to the OpenSpec change or decision that records them. Docstrings are one line unless they state a contract a caller depends on. Do not leave layers commented out.
10. **Keep the project's naming conventions.** Classes and functions in `_internals/` keep their leading underscore. Type hints stay as they are.
11. **Components don't read module globals that stand in for their arguments.** Everything a part needs comes in through its constructor or call. Tuned constants are passed in as default arguments.

## Goals / Non-Goals

**Goals:**

- Both files follow the style guide above.
- Behaviour is identical with the same seed, and a one-off comparison against the code before the change proves it (Decision 6).
- The test suite passes unmodified.

**Non-Goals:**

- Renaming, removing, or retuning anything that affects behaviour or the API.
- Changing `model_api.py`, `linkage_internals.py`, or the public modules. `linkage_internals.py` has long comments of its own and is left for a follow-up.
- Dropping the leading underscores.
- Moving the training hooks (`on_epoch_start`, `optimizer_param_groups`, `on_step_end`) off the model.
- Keeping `state_dict` key names stable.

## Decisions

### 1. Generator components

```
_ConvEncoder(input_shape)            Sequential: conv_block(C,8) → conv_block(8,16) → Flatten
    .feature_shape, .flatten_dim     measured by one dummy forward pass in __init__
_Decoder(latent_dim, feature_shape,  Sequential: Linear → _Reshape(feature_shape)
         out_channels)                 → deconv_block(16,128) → ConvTranspose2d(128,C) → Sigmoid
_build_classification_head(...)      Sequential: Linear → ReLU → Linear   (kept as is)
_reparameterize(mu, logvar)          plain function
_LearnedTaskWeights()                reconstruction_log_var, classification_log_var,
    .weigh_reconstruction(mse)         mse / (2·exp(lv)) + lv/2
    .weigh_classification(ce)          ce / exp(lv) + lv/2
    .clamp_to_floor()                  in-place clamp at LOG_VAR_FLOOR
_kl_divergence(mu, logvar)           per-image KL, batch mean
_kl_weight(epoch, total_epochs)      unchanged
_VAEOutput(NamedTuple)               reconstruction, mu, logvar, logits

_Autoencoder = _ConvEncoder → Linear(to_latent) → _Decoder
_VAE         = _ConvEncoder → fc_mu / fc_logvar → _reparameterize → _Decoder
                                                              └──→ head
```

The encoder trunk stays shared between `_Autoencoder` and `_VAE`, and each model builds its own instance. The `mu`/`logvar` projections stay on `_VAE` and are not folded into the encoder. That keeps the trunk usable by the autoencoder unchanged, and it keeps the order in which parameters are created (Decision 6).

`_VAE` keeps `reconstruction_log_var` and `classification_log_var` as read-only properties that return the `_LearnedTaskWeights` parameters. The tests (and anyone debugging) can then still reach them under their current names. The alternative was updating the tests, but this change relies on an unmodified test suite as evidence that nothing changed.

The training hooks stay on `_VAE`, grouped under a `# Training hooks, called by train()` heading. Each becomes one or two lines that delegate to `_LearnedTaskWeights` or `_kl_weight`.

### 2. Classifier components

`_CNNClassifier` gets a `features` `Sequential` (conv → ReLU → pool, twice) and a `head` `Sequential` (Flatten → Linear → ReLU → Linear). Its flatten size is still measured by a dummy forward pass. `classify()` becomes `head(features(x))`.

`minimum_input_spatial_size()` and the pool layers must not drift apart. Both derive from one number of pooling stages and one pool size, kept as two module constants (style guide rule 4b). The convolutions' kernel, padding, and stride move inline, because nothing else reads them. The "too small" error message stays word for word.

### 3. Shape measurement replaces shape arithmetic

`_conv_output_size` and `_encoded_feature_shape` are deleted. The encoder measures its output with `torch.zeros(1, C, H, W)` under `no_grad`, which draws no random numbers. The decoder takes `encoder.feature_shape` as an argument instead of recomputing it from `output_shape`. `_validate_decode_shape` stays, and it still runs at build time. It is what catches image sizes whose encode/decode round trip doesn't return the original size, such as sizes not divisible by 4.

The alternative was to keep the arithmetic and only tidy it. The arithmetic is the sole reason for five constants and two helpers, and the classifier already uses measurement.

### 4. Comment policy as applied

Each long comment becomes, at most, one line of what the value is plus one line pointing to where it was decided:

| Item | Pointer target |
|---|---|
| `LATENT_DIM` | `archive/2026-09-23-picoface-phase6/diagnostics.md`, "`latent_dim` retuning" |
| `KL_ANNEAL_FRACTION`, `_kl_weight` | `archive/2026-09-22-picoface-phase3c/design.md`, Decision 3 |
| `LOG_VAR_FLOOR`, `LOG_VAR_LR_MULTIPLIER`, `_LearnedTaskWeights` | phase3c design.md, Decision 4 |
| Per-pixel loss scale in `training_step` | phase3c design.md, Decision 5 |
| Decoder middle width (128) | phase6 diagnostics.md, "Decoder capacity escalation" |
| Head reads `z`, not `mu` | phase3c design.md, Decision 1 |
| Autoencoder is optional / reconstruction-only | phase3c design.md, Decision 7 |

Pointers are relative to `openspec/changes/`. The module docstrings shrink to two or three lines.

The one comment kept in full is the reason the log-variances are clamped in place, on the stored parameter and not only on the value the loss reads. That is a correctness constraint a future editor could break, so it stays as one or two lines on `clamp_to_floor()`.

### 5. Tuned values stay module constants and are passed in

`LATENT_DIM`, `KL_ANNEAL_FRACTION`, `LOG_VAR_FLOOR`, and `LOG_VAR_LR_MULTIPLIER` stay public module constants at their current values. The decoder's middle width and the head's hidden width become inline literals with a pointer comment. Nothing outside the file reads them, so rule 4a doesn't require a constant. The builders `_build_vae`, `_build_autoencoder`, and `_build_classifier` keep their signatures.

### 6. Behaviour preservation is proven, not assumed

Same seed → same initial weights requires the parameterized layers to be **created** in the same order:

- VAE: conv1, conv2, fc_mu, fc_logvar, decoder Linear, deconv1, deconv2, head Linear×2.
- Autoencoder: conv1, conv2, to_latent, decoder.
- Classifier: conv1, conv2, fc1, fc2.

Comparing weights simply as lists also requires `parameters()` to come out in that same order. The log-variances are zero-initialized, so where they are created doesn't matter for random numbers. They were direct parameters of `_VAE`, and `parameters()` lists a module's own parameters before its children's, so they came first. `_LearnedTaskWeights` is therefore registered as `_VAE`'s first child, which keeps them first.

`_validate_decode_shape` runs `forward()`, and the VAE's `forward()` draws random noise in `_reparameterize`. It must stay in the builder at the same point, so later draws line up.

A temporary, uncommitted comparison script records the following on the current code, then compares them against the refactored code:

- each model's `parameters()` right after building, under a fixed seed;
- one `training_step` loss and metrics dict under a fixed seed;
- `sample()` output for the VAE.

It covers two input shapes, the stub shape (16×16×3) and one other valid shape (for example 32×32×1). It also checks that each builder raises the same error type with the same message for an invalid shape. The comparison requires exact equality, not a tolerance. The same operations run in the same order, and `Sequential` adds no arithmetic.

### 7. The before/after test run uses the fast suite; the full suite is deferred

A plain `pytest` run collects `tests/test_real_dataset.py` and `dataset_forge/tests/`. Those build real Dataset Forge exports and fully train the classifier and VAE on CPU. The baseline run of the full suite went past 40 minutes without finishing. So this change's before/after check runs only the fast suite, which uses stub data:

```
pytest tests/ --ignore=tests/test_real_dataset.py
```

That is enough evidence for a refactor that doesn't change behaviour. The Decision 6 comparison already shows exact equality of weights, gradients, losses, metrics, samples, and error messages. The fast suite then checks that every caller still works. The real-dataset tests check model quality and time budgets, and those depend only on behaviour that Decision 6 proves unchanged.

More readability refactors may follow, for example on `linkage_internals.py`. The full suite should be run once after that series, not once per change.

## Risks / Trade-offs

- [A `Sequential` hides a layer that later needs its own name, for example for a diagnostic hook] → Index into the `Sequential`, or split it. Diagnostics don't currently reach individual layers.
- [Pointer comments go stale if archived change directories are renamed] → Archive paths are dated and never renamed under the OpenSpec workflow. The pointers name a section or decision number, not a line number.
- [The equality check passes only on the machine and PyTorch build it ran on] → That is enough. It compares before against after on one environment, and it is not kept as a regression test.
- [The read-only log-var properties are a small extra layer of indirection] → Accepted, to keep the test suite unmodified. They can go in a later change that also updates the tests.
- [`state_dict` keys change] → No saved weights exist in the project. This is noted in the proposal.
