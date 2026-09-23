## Context

See `proposal.md` - Why/What Changes for the motivation and the shape of the fix. This section covers what this session measured to get there, since the path wasn't a straight line from the inherited hypothesis to the adopted fix, and the numbers behind that path matter for anyone revisiting this later.

**Starting point:** `openspec/phase7-star-sampling-handoff.md` (this repo, untracked scratch doc from a prior `/opsx:explore` session) diagnosed `star`'s weak `classify_generated()` (0.147 gen vs. 0.660 recon at Phase 6's shipped defaults) as likely caused by `star`'s near-rotational-symmetry making its real-image latent cluster a closed loop/ring, which the current isotropic-Gaussian-per-class model (`_class_latent_clusters`/`_sample_near_clusters`) can't represent — landing the sampling mean off-manifold, in the ring's hole. The handoff proposed three fixes to try in order: (1) shrink `CLUSTER_SAMPLE_SPREAD`, (2) full covariance instead of diagonal, (3) resample real encoded points.

**What this session measured, against a freshly trained VAE+CNN on the real dataset (`dataset_forge/output/default-2x-seed0`, matching Phase 6's shipped `default.json` config):**

1. **The ring hypothesis didn't hold up.** PCA on `star`'s encoded `mu`s (top-2 components) gave a radius coefficient-of-variation (0.557) statistically indistinguishable from `circle`'s (0.532) or `square`'s (0.564) — a true ring would show a much *lower* CV (near-constant radius) than an ordinary blob. `star`'s sampling mean was also *closer* to its nearest real point (1.2x the typical real-to-real neighbor gap) than `circle`'s or `ring`'s (1.6x each) — the opposite of "the mean sits in an off-manifold hole."
2. **But generated star images were visibly malformed** (smeared/merged points) and specifically misclassified as `triangle` 80% of the time (n=50), not diffusely across classes. A control check confirmed the independent CNN classifies 100% of *real* held-out star test images correctly — so this is purely a generation artifact, not `star`/`triangle` being naturally close classes for the classifier.
3. **Step 1 (shrink spread 0.75→0.6→0.5→0.4) failed, and failed informatively:** it made `star`/`smiley` monotonically *worse* (star: 0.132→0.044→0.004→0.000) while helping the round classes. Mechanism: shrinking spread pulls every sample toward the class mean, and for a class whose real images vary strongly along a global nuisance parameter (rotation), that mean is a smeared multi-orientation composite that doesn't decode cleanly — tightening spread just gets you to the bad point faster.
4. **Step 2 (full covariance, same spread) also failed to meaningfully help** (star 0.084-0.200 depending on spread tried). Diagonal vs. full covariance barely mattered — confirms the *shape* of the parametric distribution was never the real problem; its *center* was.
5. **Step 3 (resample real encoded points + small jitter) won decisively**, and the jitter magnitude turned out to barely matter across 0.05-0.15 (statistically indistinguishable once averaged over 5 training seeds x 3 sampling seeds each: overall 0.813/0.814/0.814). Settled on **jitter=0.1** as the plateau's midpoint — no data favors 0.05 or 0.15 over it.

**Final validation** (5 training seeds x 3 sampling seeds, n=50/class, same real dataset export, shipped baseline vs. adopted fix):

| class | baseline mean ± std | fix (jitter=0.1) mean ± std |
|---|---|---|
| square | 0.915 ± 0.035 | 0.928 ± 0.028 |
| ring | 0.799 ± 0.147 | 0.995 ± 0.007 |
| circle | 0.727 ± 0.100 | 0.999 ± 0.003 |
| triangle | 0.565 ± 0.155 | 0.844 ± 0.090 |
| star | 0.081 ± 0.041 | 0.571 ± 0.070 |
| smiley | 0.365 ± 0.219 | 0.584 ± 0.093 |
| negative_smiley | 0.656 ± 0.140 | 0.771 ± 0.103 |
| **overall** | **0.587 ± 0.038** | **0.814 ± 0.019** |

Every class improves in both mean and seed-to-seed consistency — no regressions. The consistency gain is notable on its own: `ring`/`circle`'s baseline std (0.10-0.15) means some unlucky training seeds shipped a nearly-broken capstone for those classes too, not just `star`/`smiley`; overall std roughly halves (0.038→0.019). Since each student's notebook run gets its own random seed, this matters pedagogically, not just as an average-case number.

All of the above was measured by monkeypatching `picoface.linkage`'s own name-bindings in memory (the same mechanism, and the same already-caught gotcha, as Phase 6's constant sweeps — see the Decisions section below), never editing `src/picoface/`. Scripts live in this session's scratch directory, not the repo, and are not a deliverable of this change.

## Goals / Non-Goals

**Goals:**
- Close `star`/`smiley`'s `classify_generated()` gap without regressing any of the other five classes.
- Keep the fix confined to `_class_latent_clusters`/`_sample_near_clusters`'s internals — no public API or `GeneratedImagesReport` shape changes.

**Non-Goals:**
- Re-deciding Phase 7's other open items (accuracy/quality bar, notebook content, distribution channel) — untouched by this change.
- Label-conditioned decoding or any decoder/encoder architecture change — ROADMAP.md lists this as a candidate *if* sampling-level fixes didn't close the gap; they did, so it's moot for now.
- Exposing the jitter constant, or the sampling mechanism at all, as a public parameter — it stays exactly as internal as `CLUSTER_SAMPLE_SPREAD` was.

## Decisions

**Resample real encoded points + small jitter, over both parametric alternatives.** Steps 1 and 2 both failed empirically (see Context); step 3 won by a wide, seed-robust margin. The mechanism this points to: for classes with strong nuisance-parameter-driven variation (rotation), no Gaussian centered on the class mean is a good generator, regardless of its covariance structure, because the mean itself isn't a point the decoder renders well. Sampling actual real points sidesteps that by construction — every candidate before jitter is a point the encoder/decoder pair is already known to handle (it's literally one of the training images' own latent means).

**Jitter magnitude fixed at 0.1, not tuned per class or exposed as a parameter.** The sweep (0.0/0.05/0.1/0.15/0.25, relative to each class's own per-dimension std) showed a real, above-noise jump from 0.0→0.05 (mainly `smiley`, +0.088) and a real decline by 0.25, but 0.05-0.15 in between is a flat plateau — every difference in that range is inside one training-seed's noise once averaged over multiple seeds. One shared constant across all 7 classes is simpler to implement, test, and explain than a per-class value, and the data gives no reason to prefer a per-class tune.

**`_class_latent_clusters` returns raw per-class `mu` tensors, not a new parallel helper.** The proposal considered adding a second helper alongside the existing one instead of changing `_class_latent_clusters`'s return type. Rejected: `_class_latent_clusters` has no other callers (`grep` confirms only `classify_generated()` uses it via `_sample_near_clusters`), so there's no reason to keep the old `(mean, std)` shape alive next to a new one — that would just be dead code the moment this change lands.

**Sampling with replacement, not without.** For `n` draws from a class with far more than `n` real points (2,000 train images/class in the shipped config), sampling with replacement is simpler and the difference is immaterial at typical `n` (10-50); sampling without replacement would need extra bookkeeping (and would fail outright for `n` larger than the class's image count) for no measured benefit.

**Methodological note carried forward from Phase 6, load-bearing for this change too:** `picoface/linkage.py` does `from picoface._internals.linkage_internals import (_class_latent_clusters, _sample_near_clusters, ...)` — a direct name import, so `linkage.py` holds its own binding independent of the source module's attribute. Phase 6's diagnostics.md already hit this once (patching `linkage_internals`'s attribute silently did nothing; the fix was patching `picoface.linkage`'s own name). This session's measurements patched `picoface.linkage._class_latent_clusters`/`_sample_near_clusters` for exactly that reason. It has no bearing on the actual code change (editing the source functions directly obviously works regardless of import style), but it's the reason every number in this document came from monkeypatching at the `picoface.linkage` name, not the internals module's.

## Risks / Trade-offs

- **[Risk] This narrows what "generation" means for the capstone** — sampling a real point plus small jitter is closer to "reconstruct a random real example with light noise" than "sample a fitted density," which is a real design-stance shift from Phase 4's original framing ("class-targeted generation... uses the supervised latent's empirical clusters"). → Accepted: the empirical result (every class better, more consistent, no regressions) outweighs the philosophical distinction for this project's purpose, and `classify_generated()`'s existing spec language ("drawn from the region... that class's labeled images actually occupy") already permits it — see proposal.md's Capabilities section. Worth a one-line acknowledgment in whatever Phase 7 notebook content eventually narrates this exercise, but that's Phase 7's notebook-content decision, not this change's.
- **[Risk] Jitter=0.1 was tuned against one real dataset export (`default-2x-seed0`) and 5 training seeds, all on the same underlying data.** A different real export (different rendering noise/jitter ranges) could shift the plateau. → Low severity: the plateau (0.05-0.15) is wide, not a knife-edge, so a modest shift in the "right" value from a different export is unlikely to fall outside it. Not re-validated against `default-2x-seed1` in this session; worth a quick check if this regresses unexpectedly after the fix ships.
- **[Risk] Sampling from training-set points means `classify_generated()`'s generated images are now literally training images (plus light jitter), which a careful student could notice.** → Accepted as a pedagogical non-issue: the exercise's purpose is checking whether the independently trained CNN agrees with the VAE's notion of each class, not testing out-of-distribution novelty; Phase 4's design already scoped this exercise to not require a label-conditioned decoder or true novel-sample guarantees.

## Migration Plan

No data migration. This is a code-only change to two internal functions and one module constant in `src/picoface/_internals/linkage_internals.py`. Rollback is a straight revert (no persisted state, no serialization format per the project's "linkage operates on in-memory models" design decision). Any existing test that asserts on `CLUSTER_SAMPLE_SPREAD`'s name or value needs updating alongside the implementation change (see `tasks.md`).
