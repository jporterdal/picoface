## Why

`classify_generated()`'s class-targeted sampling (`_class_latent_clusters` / `_sample_near_clusters`, `src/picoface/_internals/linkage_internals.py:45-77`) draws each class's generated images from `mean + CLUSTER_SAMPLE_SPREAD * std * N(0, I)` — an isotropic Gaussian fit to that class's encoded training images. This works fine for round/simple classes but leaves `star` and `smiley` badly behind: at the current shipped defaults, averaged over 5 training seeds, `star` scores 0.081 and `smiley` 0.365 (`classify_generated()` agreement) against an overall of 0.587, even though both classes reconstruct fine from their own real latents. Phase 6 archived with this as a named open question (ROADMAP.md's Open Questions); this session (Phase 7) picked it up, ran the leading hypothesis and two follow-up fixes to a real measured conclusion, and found a fix that resolves it without regressing any other class.

## What Changes

- Replace `_class_latent_clusters`' return value: instead of each class's `(mean, std)` of encoded `mu`, return the class's raw encoded `mu` tensor (every real training image's latent mean, per class).
- Replace `_sample_near_clusters`' draw: instead of a parametric Gaussian around the class mean, pick a random real encoded point from that class and add a small Gaussian jitter (`jitter * per-dimension std * N(0, I)`) before decoding.
- Retire the `CLUSTER_SAMPLE_SPREAD` constant (0.75, tuned in Phase 6 for the old parametric method) and replace it with a jitter-magnitude constant at `0.1` — a different quantity, not a retuning of the old one.
- No public API changes: `_class_latent_clusters`/`_sample_near_clusters` are internal (imported by name into `picoface/linkage.py`); `classify_generated()`'s signature and `GeneratedImagesReport`'s shape are untouched.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
None. `capstone-linkage`'s existing requirement ("images shall be drawn from the region of the generator's latent space that class's labeled images actually occupy, not an unconditioned sample of the whole space") is already satisfied by the current parametric sampler and remains satisfied — arguably more literally — by resampling real points from that exact region. The spec does not mandate a parametric vs. non-parametric mechanism; this mirrors Phase 6's own precedent of tuning `CLUSTER_SAMPLE_SPREAD` (and decoder capacity, log-variance floor, etc.) without a `capstone-linkage` spec delta, since none of those are internal constants named in spec text.

## Impact

- `src/picoface/_internals/linkage_internals.py`: `_class_latent_clusters`, `_sample_near_clusters`, and the `CLUSTER_SAMPLE_SPREAD` constant are rewritten/replaced as described above. `classify_generated()` in `src/picoface/linkage.py` calls these by name and needs no changes itself.
- No changes to `picoface/generator.py`, `picoface/classifier.py`, or any public module.
- Existing `capstone-linkage` tests that exercise `classify_generated()`'s contract (shapes, error paths, taxonomy checks) are unaffected, since those don't assert on the sampling mechanism. Any test that happens to assert on `CLUSTER_SAMPLE_SPREAD` by name needs updating.
- This change's diagnostics (geometry check, sampling-mechanism comparison, jitter sweep, final 5-seed validation) were run this session via monkeypatching, not against the real source — see `design.md` for the full measured results and the exact numbers behind the decision.
