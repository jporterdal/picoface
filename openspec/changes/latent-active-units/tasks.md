# Tasks

## 1. Active-units report in the smoke check

- [x] 1.1 Add `active_units_report(vae, data)` to `dataset_forge/smoke.py`. Over the whole of `data`, in eval mode with no gradients, it computes per latent dimension: Var(μ) across images, the mean posterior variance exp(logvar), and the mean KL against N(0, I). Dimensions are sorted by Var(μ), and active counts are given at thresholds 0.001, 0.01 and 0.1 (design.md, Decisions 1–2). Verify: a unit test in `dataset_forge/tests/test_smoke.py` uses a stand-in model with known μ/logvar (some dimensions constant, some varying) and checks the counts, the sort order, and a known KL value.
- [x] 1.2 Add masked reconstruction agreement: decode μ with the dimensions below 0.01 set to 0, and separately with the active ones set to 0, and measure CNN agreement the same way `reconstruction_report` does (design.md, Decision 3). Verify: a unit test checks that masking no dimensions gives exactly the unmasked reconstructions, and that the two masks are complementary.
- [x] 1.3 Wire both into `smoke()`'s results and `format_results()`'s Markdown: a summary table (nominal size, active counts per threshold, unmasked / inactive-masked / active-masked agreement) plus the sorted per-dimension table. Update the module docstring's list of reported measurements. Verify: `test_smoke_check_measures_both_models_on_an_export` is extended to assert the new keys and table headings on the tiny export, and `python -m dataset_forge.smoke` runs end to end on a default export.

## 2. Measurements

- [x] 2.1 Run the smoke check on the shipped model (default 2,000-per-class export, `train()` defaults, seeds 0–2, `--n 50`) and record the summary tables in `openspec/changes/latent-active-units/diagnostics.md`, following Phase 6's format (setup, baselines, results, interpretation). Verify: diagnostics.md has the 3-seed table, and the unmasked agreement numbers match the existing reconstruction report within seed noise.
- [x] 2.2 Run the nominal-size sweep (8, 16, 32, 64, 128, 192) with a throwaway script, not committed, that overrides `generator_internals.LATENT_DIM` and resets `torch.manual_seed(seed)` immediately before each model is built, over 3 seeds (design.md, Decision 4). Verify: diagnostics.md has an active-count-versus-nominal-size table for every seed, and states the exact override and seeds so the sweep can be reproduced.
- [x] 2.3 Re-run the shipped 128-dimension model at 3× the default epochs on one seed to check that ten epochs is enough for the count to settle. Verify: diagnostics.md records both counts and says whether they differ materially.

## 3. Interpretation and roadmap

- [x] 3.1 In diagnostics.md, classify the results against design.md Decision 5's outcomes (A, B or C), with the evidence for the call. Record the recommendation for a follow-up change, or record that no change is warranted. Verify: diagnostics.md has a "Decision" section naming exactly one outcome and one recommendation, citing the tables from group 2.
- [x] 3.2 Add an entry to `openspec/ROADMAP.md`'s Open Questions on effective versus nominal latent dimension, summarizing the finding and pointing at this change's diagnostics.md. Also record design.md's deferred question about measuring `generate()`'s unconditional samples, if the outcome makes it relevant. Verify: the entry exists and its link resolves.
