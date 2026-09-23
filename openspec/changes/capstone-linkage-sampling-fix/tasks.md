## 1. Implementation

- [ ] 1.1 Rewrite `_class_latent_clusters` in `src/picoface/_internals/linkage_internals.py` to return each class's raw encoded `mu` tensor (not `(mean, std)`), and verify by calling it directly against a small trained VAE and checking the returned dict's values are `(n_class, latent_dim)` tensors.
- [ ] 1.2 Rewrite `_sample_near_clusters` to draw `n` samples per class by picking a random real point (with replacement) from that class's tensor and adding `JITTER * per-dimension std * torch.randn(...)`, then decoding; verify by calling it with `n=4` against stub data and checking output shape/dtype (`(4 * num_classes, H, W, C)`, `uint8`) matches the current contract.
- [ ] 1.3 Replace the `CLUSTER_SAMPLE_SPREAD = 0.75` module constant with a new jitter-magnitude constant set to `0.1`, and update its docstring comment to describe what it now controls (jitter around a real resampled point, not spread around a parametric mean) and reference this change's diagnostics for the value.
- [ ] 1.4 Read through `linkage_internals.py`'s and `linkage.py`'s docstrings/comments for any other references to the old parametric-cluster description (e.g. `classify_generated()`'s own docstring in `linkage.py`) and update wording to match the new mechanism.

## 2. Tests

- [ ] 2.1 Update `test_class_clusters_are_latent_vectors_that_separate_the_classes` in `tests/test_linkage.py` for `_class_latent_clusters`'s new return shape (raw per-class `mu` tensors instead of `(mean, std)` pairs) — verify by running `pytest tests/test_linkage.py -k class_clusters` and confirming it exercises the new shape meaningfully (e.g. still checks that classes are separated in latent space, using per-class tensor means/spreads computed inline rather than relying on the function to have already done so).
- [ ] 2.2 Confirm `test_sampling_near_clusters_gives_n_uint8_images_per_class` still passes unmodified against the new implementation (it only asserts on image shape/dtype/intended-list, not cluster internals) — verify by running `pytest tests/test_linkage.py -k sampling_near_clusters`.
- [ ] 2.3 Run the full `tests/test_linkage.py` file and confirm every test passes, verified by `pytest tests/test_linkage.py -v` showing no failures.
- [ ] 2.4 Run the full project test suite and confirm nothing outside `test_linkage.py` references the old `CLUSTER_SAMPLE_SPREAD` name or the old cluster shape, verified by `pytest` (full run) passing and `grep -rn CLUSTER_SAMPLE_SPREAD src/ tests/` returning no stale hits.

## 3. Real-data validation

- [ ] 3.1 Run `python -m dataset_forge.smoke dataset_forge/output/default-2x-seed0 --seed 0 --n 50 --figures <scratch-dir>/figures` against the changed code (not monkeypatched) and confirm `star`/`smiley`'s `classify_generated()` agreement lands in the range this change's design.md measured (star ~0.5-0.7, smiley ~0.5-0.6 at this single seed), with no class regressing below its baseline number in design.md's table.
- [ ] 3.2 Repeat 3.1 across at least 2 more training seeds (matching this change's design.md methodology of 5 training seeds) and confirm the aggregate lands close to design.md's recorded 0.814±0.019 overall, verified by comparing the printed per-class table to design.md's.
- [ ] 3.3 Visually inspect the `generated.png` figure from 3.1 for `star` and `smiley` and confirm the images read as recognizable stars/smileys, not the smeared/malformed shapes seen at baseline — a qualitative check, not just a score, per Phase 6's established practice of not trusting a number without also looking at the image.
- [ ] 3.4 Write a `diagnostics.md` in this change's directory recording the real-code validation numbers from 3.1-3.2 and the before/after figures, distinct from (but consistent with) the monkeypatched numbers already recorded in design.md.

## 4. Cleanup and roadmap

- [ ] 4.1 Update `openspec/ROADMAP.md`'s Open Questions entry for `star`'s `classify_generated()` weakness to reflect the resolution (real-point resampling adopted, jitter=0.1), following the same strikethrough-and-resolve pattern already used for other resolved entries in that section.
- [ ] 4.2 Delete or archive `openspec/phase7-star-sampling-handoff.md` per its own header ("fold its conclusions into a real change proposal once there's something to propose") now that this change captures its conclusions — confirm with the user before deleting, since it's an untracked file they may want to keep as a record.
