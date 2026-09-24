## Real-code validation

Everything in `design.md`'s Context section was measured by monkeypatching
`picoface.linkage`'s name-bindings in memory, before any source file was
touched. This records the same measurement run against the actual changed
code in `src/picoface/_internals/linkage_internals.py` (post-implementation),
via `python -m dataset_forge.smoke dataset_forge/output/default-2x-seed0
--seed <N> --n 50 --figures <dir>`, three fresh training seeds (0, 1, 2).

### Per-seed results

| class | seed 0 | seed 1 | seed 2 | 3-seed mean | design.md fix mean±std | design.md baseline mean |
|---|---|---|---|---|---|---|
| square | 0.98 | 0.94 | 0.90 | 0.940 | 0.928 ± 0.028 | 0.915 |
| ring | 1.00 | 1.00 | 0.98 | 0.993 | 0.995 ± 0.007 | 0.799 |
| circle | 1.00 | 1.00 | 1.00 | 1.000 | 0.999 ± 0.003 | 0.727 |
| triangle | 0.96 | 0.92 | 0.76 | 0.880 | 0.844 ± 0.090 | 0.565 |
| star | 0.68 | 0.62 | 0.46 | 0.587 | 0.571 ± 0.070 | 0.081 |
| smiley | 0.40 | 0.50 | 0.78 | 0.560 | 0.584 ± 0.093 | 0.365 |
| negative_smiley | 0.88 | 0.50 | 0.86 | 0.747 | 0.771 ± 0.103 | 0.656 |
| **overall** | 0.84 | 0.78 | 0.82 | **0.813** | **0.814 ± 0.019** | 0.587 |

The 3-seed overall mean (0.813) lands almost exactly on design.md's
5-training-seed monkeypatched estimate (0.814 ± 0.019) — the real
implementation reproduces the measured effect, not just the mocked one.

### Checking task 3.1's per-seed criteria (seed 0)

`star` = 0.68 (design.md's predicted ~0.5–0.7 band: yes) and `smiley` = 0.40
(predicted ~0.5–0.6 band: slightly under, see note below). No class at seed 0
fell below its `design.md` baseline mean — every one of the 7 classes at
seed 0 is above its baseline row.

### One observation, not a regression

`negative_smiley` at seed 1 (0.50) landed below its `design.md` baseline
mean (0.656), and `smiley` at seed 0 (0.40) landed under the predicted
~0.5–0.6 band. Both are single real-training-seed draws, not repeats of the
mocked experiment's exact seeds, and `design.md` already documented
`negative_smiley`'s baseline itself as high-variance seed-to-seed
(std 0.140) — this is consistent with that same real training-seed variance
showing up here, not a new problem introduced by this change. The 3-seed
average for both classes is comfortably above their baseline means
(`negative_smiley` 0.747 vs. 0.656; `smiley` 0.560 vs. 0.365), and no class's
3-seed average regresses below its baseline mean.

### Visual check (task 3.3)

`generated.png` from the seed-0 run (`/tmp/picoface-star-sampling/apply-figures-seed0/generated.png`,
not part of this change's committed artifacts) shows crisp, individually
recognizable stars across all 12 sampled images — no smearing or merged
points, a clear visible change from the baseline's malformed stars. Smiley
and negative_smiley rows are also clearly face-shaped. Confirms the score
improvement corresponds to a real qualitative improvement, not an artifact
of the metric.

### Held-out accuracy sanity check

All three seeds' CNN and VAE held-out classification accuracy stayed at
0.98–0.99, matching Phase 6's shipped baseline — this change does not affect
training or classification accuracy, only `classify_generated()`'s sampling.
