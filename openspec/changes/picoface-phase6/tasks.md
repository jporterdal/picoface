## 1. Real-dataset test fixture

- [ ] 1.1 Add a way for tests to obtain a real Dataset Forge export on demand (e.g. a pytest fixture that runs the Forge's export with a fixed seed into a temp directory), without committing a generated dataset to the repo. Verify: a new test using the fixture loads a bundle via `load_dataset()` with the expected shape and 7 classes.
- [ ] 1.2 Identify and list the existing stub-based tests that validate model *quality* rather than plumbing (accuracy bounds, CPU time budget, `classify_generated()` agreement) and are in scope to gain a real-dataset counterpart per design.md's "stub stays, real dataset is added" decision. Verify: the list is recorded in this change's tasks/PR notes and matches the requirements touched in `specs/shape-classifier/spec.md` and `specs/shape-generator/spec.md`.

## 2. Loss-balance tuning (log-variance floor, multiplier, KL schedule)

- [ ] 2.1 Reproduce Phase 5's baseline numbers (CNN/VAE held-out accuracy, `classify_generated()` agreement, reconstruction agreement) on a fresh real-dataset export at current defaults, to confirm the starting point before changing anything. Verify: numbers fall within Phase 5's recorded ranges in `openspec/changes/archive/2026-09-23-picoface-phase5/diagnostics.md`.
- [ ] 2.2 Sweep the log-variance floor and the log-variance learning-rate multiplier (`LOG_VAR_FLOOR`, `LOG_VAR_LR_MULTIPLIER` in `generator_internals.py`) against the real dataset, recording reconstruction agreement, classification accuracy, and the learned log-variances' trajectory per epoch for each setting tried. Verify: results recorded in this change's diagnostics file (task 8.1) with at least 3 distinct settings compared.
- [ ] 2.3 Sweep the KL annealing schedule's length/shape (end point stays fixed at weight 1, per the unchanged `shape-generator` requirement). Verify: results recorded alongside 2.2, same metrics.
- [ ] 2.4 Pick the best-performing loss-balance setting found in 2.2–2.3 and set it as the new default in `generator_internals.py`. Verify: the full existing generator test suite (stub-based) still passes, and the real-dataset quality tests from task 1 run against the new default.

## 3. `latent_dim` retuning

- [ ] 3.1 Sweep `latent_dim` around the current provisional value of 8, using the loss-balance setting chosen in 2.4, against the real dataset. Verify: results recorded in the diagnostics file with classification accuracy and reconstruction/`classify_generated()` agreement per value tried.
- [ ] 3.2 Set the best-performing `latent_dim` as the new default if it improves on 2.4's result; otherwise keep 8. Verify: chosen value and rationale recorded in the diagnostics file.

## 4. Decision checkpoint: did tuning resolve the blob problem?

- [ ] 4.1 Compare the best result from tasks 2–3 against Phase 5's baseline (`classify_generated()` 0.32–0.35, reconstruction agreement 0.26–0.31, only circle/ring recognizable) and record whether non-round classes (square, triangle, star, smiley, negative_smiley) are now visually/quantitatively recognizable in generated and reconstructed samples. Verify: a figure comparable to Phase 5's `generated.png`/`reconstructed.png` is produced (not committed, per the project's existing "generated artifacts aren't committed" convention) and described in the diagnostics file.
- [ ] 4.2 Per design.md's Decisions (tuning first, architecture only if tuning doesn't get there): if 4.1 shows the blobs persist, write an explicit "Open question for a follow-up change" section in the diagnostics file naming decoder capacity and label-conditioning as the untried next step, with the evidence for why tuning alone wasn't sufficient. Do not implement any decoder architecture change in this task. Verify: the section exists in the diagnostics file, or is explicitly absent because 4.1 found tuning sufficient.

## 5. Secondary, independent knobs

- [ ] 5.1 Try decoupled weight decay (`AdamW`, excluding the log-variance parameters) at a small value and compare classification/reconstruction against the task 2–3 default. Verify: recorded in the diagnostics file; adopted only if it doesn't regress either metric.
- [ ] 5.2 Try a small change to the classification head's architecture (e.g. hidden-layer width) and compare against the current default. Verify: recorded in the diagnostics file; adopted only if it improves classification without regressing reconstruction.
- [ ] 5.3 Compare Dataset Forge's default of 1,000 images/class against 2,000/class using the tuned model from tasks 2–4, confirming Phase 5's finding (extra data helps classification, not the decoder) still holds post-tuning. Verify: recorded in the diagnostics file; if adopted, update `dataset_forge/configs/default.json` and its default-count documentation.

## 6. `train()` defaults and CPU time-budget verification

- [ ] 6.1 Run `train()` with its actual default `epochs=10, batch_size=16, learning_rate=1e-3` end-to-end against a real Dataset Forge export (not a separate script) for both `build_classifier()` and `build_vae()`, and record held-out classification accuracy. Verify: accuracy is high enough to be a believable capstone judge (compare against Phase 5's 0.94–0.97 range); if not, note candidate alternate defaults with their measured effect.
- [ ] 6.2 Measure wall-clock training time for 6.1 on this session's machine and confirm it stays within "seconds to minutes," consistent with Phase 5's smoke-check numbers. Verify: time recorded in the diagnostics file.
- [ ] 6.3 If 6.1 concludes a different default is warranted, update `train()`'s default in the shared model-interface internals, then update the `model-interface` and `shape-classifier` specs to match via `opsx:update` before this change archives (proposal.md, Modified Capabilities). If no change is warranted, record that conclusion in the diagnostics file and leave the specs as-is. Verify: spec text and shipped default agree.

## 7. Capstone linkage retuning

- [ ] 7.1 Re-run `classify_generated()`'s existing sampling approach (mean plus per-dimension spread of `mu`) against the tuned model from tasks 2–4 and record whether agreement improved. Verify: recorded in the diagnostics file, compared against Phase 5's 0.32–0.35 baseline.
- [ ] 7.2 If 7.1 shows room to improve, try a small adjustment to the sampling spread (e.g. a scale factor on the per-dimension spread) and record the effect. Verify: recorded in the diagnostics file; adopted only if it improves agreement without a named downside.
- [ ] 7.3 Re-run `activation_maximize()` against the tuned model and assess whether its output is recognizable or still adversarial noise; if noise, try adding a simple regularization term (e.g. L2 or blur penalty on the ascended image) and/or adjusting step count/learning rate. Verify: before/after images described in the diagnostics file (not committed); adopted only if the change measurably improves visual recognizability without breaking the existing `capstone-linkage` tests.

## 8. Diagnostics record and project documentation

- [ ] 8.1 Write `diagnostics.md` in this change's directory, following Phase 5's format (setup, baselines, per-step results, decision, Phase-7-facing inputs), consolidating the results from tasks 2–7. Verify: file exists and every numeric claim in it traces to a task above.
- [ ] 8.2 Update `openspec/ROADMAP.md`: mark Phase 6 done with a link to this change's archive location (once archived), and resolve or re-scope each of its currently-listed Phase 6 open items (log-variance floor/multiplier, KL schedule, `latent_dim`, weight decay, classification head architecture, label-conditioned decoder feasibility, `activation_maximize()` regularization/steps, `classify_generated()` sampling, concrete accuracy/quality bar) to point at what this phase resolved versus what it explicitly deferred (task 4.2's open question, if any; the quality bar deferred to Phase 7 per this change's scoping). Verify: no Phase 6 line in the ROADMAP's Open Questions section still reads as unaddressed by this change.
- [ ] 8.3 Run the full existing test suite (stub-based and any real-dataset tests added in task 1) and confirm it passes. Verify: `pytest` exits 0 across `src/picoface` and `dataset_forge`.

## 9. Change hygiene

- [ ] 9.1 Confirm no generated dataset exports, figures, or diagnostics-run artifacts are committed (per the project's existing "generated datasets/figures are never committed" convention) — only code, config defaults, and the written-up `diagnostics.md`. Verify: `git status`/diff for this change contains no files under a Dataset Forge output directory.
- [ ] 9.2 Run `openspec validate --change "picoface-phase6" --strict` and resolve any reported issues before implementation is considered apply-ready. Verify: command exits clean.
