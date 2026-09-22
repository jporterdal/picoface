## Why

Phases 2–3c built the classifier arm (independent judge) and the generator arm (the student's model) but never tied them together. The roadmap's capstone exercise — checking the VAE's generated images against the independently trained CNN, and visualizing what the classifier "imagines" for a class — has no implementation yet (`linkage.py` is an empty stub). Phase 4 closes that gap so the three-arm architecture actually resolves into the closing exercise it was designed for.

## What Changes

- Add `classify_generated(classifier_model, generator_model, data, n)`: generates images intended for each of `data`'s classes by sampling the VAE's latent space near each class's empirical cluster (computed by encoding `data`'s labeled images), decodes them, classifies each with the independently trained CNN, and reports per-class and overall agreement between the intended class and the CNN's prediction.
- Add `activation_maximize(model, target_class)`: gradient ascent on pixel values, starting from random noise, that maximizes `target_class`'s score under any model with the `classify` capability — visualizing what the classifier has learned to associate with that class.
- Add `encode_mu()` and `decode()` as new capability-gated primitives on the abstract `_Model` base (`model-interface`). These are internal-only additions used by `classify_generated()`'s latent-cluster sampling; they are not new student-facing verbs and do not change `train`/`evaluate`/`predict`/`generate`'s signatures or behavior.
- `classify_generated()` validates that the classifier's and generator's `class_names` match before doing anything, raising a clear, named error on mismatch.

## Capabilities

### New Capabilities
- `capstone-linkage`: the functions that tie the classifier and generator arms together — `classify_generated()` (score class-targeted generated images with the independently trained CNN) and `activation_maximize()` (gradient-ascent visualization of a classifier's learned class representation).

### Modified Capabilities
- `model-interface`: adds two new optional model capabilities/primitives (`encode_mu()`, `decode()`) to the abstract `_Model` contract, gated the same way `classify`/`sample` already are. No existing requirement's behavior changes; this is additive.

## Impact

- `src/picoface/linkage.py`: implemented (currently an empty stub). New public functions `classify_generated()`, `activation_maximize()`.
- `src/picoface/_internals/model_api.py`: `_Model` gains `encode_mu()`/`decode()` capability-gated methods (default: not implemented / no capability declared).
- `src/picoface/_internals/generator_internals.py`: `_VAE` declares the new capabilities and implements `encode_mu()`/`decode()` using its existing, unchanged encoder/decoder (already has `encode_mu`/`decoder` internally — this only exposes them through the capability protocol).
- `src/picoface/_internals/classifier_internals.py`: the CNN classifier is unaffected (it has no latent space; it does not declare the new capabilities).
- No changes to `picoface.classifier`, `picoface.generator`'s public signatures, no new training, no serialization format.
- New tests exercising both linkage functions against the stub dataset, plus the classifier/generator taxonomy-mismatch error path.
