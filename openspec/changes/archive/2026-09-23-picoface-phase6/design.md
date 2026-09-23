## Context

See `proposal.md` (Why) for the motivating diagnosis. Relevant current state:

- `src/picoface/_internals/generator_internals.py` holds every VAE constant this phase touches: `LATENT_DIM` (8), `LOG_VAR_FLOOR`, `LOG_VAR_LR_MULTIPLIER` (10×), the KL annealing schedule, and the small transposed-conv decoder (`_build_decoder`, two conv layers). None of these are named in spec text — they're internals by the project's own public-API/`_internals` boundary design decision, so retuning them is implementation work, not a spec change.
- The diagnostics baseline this phase tunes against is `openspec/changes/archive/2026-09-23-picoface-phase5/diagnostics.md`: CNN 0.95–0.97, VAE 0.94–0.95 held-out classification; `classify_generated()` 0.32–0.35; reconstruction agreement 0.26–0.31; only circle/ring ever recognizable; neither 2× data nor 2× epochs moved the blobs.
- `dataset_forge/configs/default.json` currently exports 1,000 images/class at 28×28; Phase 5 found 2,000/class is a cheap classification-only win (CNN →0.99, VAE →0.98–0.99) that does not move the decoder problem.
- Existing tests validate classifier/generator/linkage behavior against the stub dataset (via `data-contract`'s internal stub generator). This phase adds a real Dataset Forge export as a second fixture for the checks that need real content (accuracy bounds, time budget, capstone agreement); the stub-based plumbing tests are untouched.

## Goals / Non-Goals

**Goals:**
- Diagnose, with the same rigor as Phase 5's diagnostics record, whether the blob problem is fixable by rebalancing the existing VAE objective (log-variance floor/multiplier, KL schedule, `latent_dim`) rather than by growing decoder capacity.
- Leave a documented, reproducible record (constants tried, what moved, by how much) that a later change can build on, whichever way the diagnosis lands.
- Verify the project's two binding constraints (CPU time budget, `train()`'s default epoch count giving good classification) hold against real data, not just the stub or Phase 5's ad hoc smoke script.

**Non-Goals:**
- ~~Changing decoder architecture~~ **Revised after task 8**: loss-balance/`latent_dim` tuning left `star`/`smiley` behind, and the user explicitly asked to push on decoder capacity before archiving rather than defer it (task 10). What was actually changed: the existing transpose-conv layers' channel width, nothing else — no new layers, and label-conditioning remains out of scope for this change (it would touch `generate()`'s public, unconditional contract, a bigger design question than a decoder-only constant).
- Committing to a final accuracy/quality bar for `classify_generated()` or reconstruction agreement — per this session's scoping decision, that stays a Phase 7 framing question.
- Changing the taxonomy, resolution, or noise policy Dataset Forge already decided in Phase 5.

## Decisions

**Investigation order: loss-balance → `latent_dim` → secondary knobs, stopping early on success.** The log-variance floor/multiplier is tried first because Phase 3c's own risk note already names it as the most likely cause, and because the diagnostics record shows classification (which reads the same latent) is unaffected while reconstruction alone fails — consistent with reconstruction's task weight being pinned too low relative to classification's. `latent_dim` comes second because it's cheap to sweep but changes what capacity is available for both tasks, so tuning the balance first avoids re-tuning it twice. If an earlier step already gets `classify_generated()`/reconstruction agreement off the "only round classes work" floor, later steps in the list are still worth a quick check but are not required to declare the phase's tuning investigation complete.

**Alternative considered and initially rejected: go straight to decoder capacity.** The diagnostics record's own framing ("It is a Phase 6 tuning question... It is not a Forge or data-contract issue") and this session's explicit scoping decision (tuning first, architecture only if tuning doesn't get there) both pointed at trying the cheaper, already-anticipated fix before adding model capacity. Trying loss-balance and `latent_dim` first, in that order, did in fact resolve most of the blob problem (`square`/`ring`/`circle`/`triangle`/`negative_smiley`) and made it unambiguous that only `star`/`smiley` needed something more — exactly the clarity this ordering was meant to produce, vindicating not skipping straight to architecture.

**Decoder capacity was pulled into this same change, at explicit user direction (task 10).** Once task 4's diagnostics showed `star`/`smiley` still unresolved, the original plan was to write that up as a named open question for a follow-up change (this section's prior text, and task 4.2). The user reviewed those diagnostics and asked to push further before archiving. Widening the decoder's existing transpose-conv channel count (a small, contained, decoder-only change — no new layers, no interface change) was tried the same way every other knob in this phase was: swept, confirmed across 3 seeds (after a single-seed pass proved misleadingly noisy — different channel widths consume different amounts of RNG state during initialization, confounding single-seed comparisons more than for scalar hyperparameters), and adopted only once the improvement was robust. It resolved `smiley` and substantially improved `star`'s reconstruction, leaving only `star`'s `classify_generated()` (cluster-sampling) specifically as a narrower open question than task 4.2 originally posed (diagnostics.md).

**Stub dataset stays; real dataset is added, not swapped in wholesale.** Tests whose purpose is fast plumbing validation (shape/class-count checks, error-path tests, capability-gating tests) keep using the stub — it's fast, deterministic, and `data-contract`'s stub-generator requirement isn't touched by this change. Tests whose purpose is validating actual model quality (accuracy bounds, time budget, `classify_generated()` agreement) move to a real Dataset Forge export, since that's the whole point of this phase. This mirrors how Phase 5's own smoke check was a separate script rather than a rewrite of the existing stub-based test suite.

**`train()` default-epoch and time-budget investigations run against `train()` itself, not a new script.** Phase 5's smoke check was deliberately an ad hoc, uncommitted script (`dataset_forge.smoke`) reporting one-time feasibility numbers. This phase's equivalent investigation should exercise the actual public `train()` defaults end-to-end (the same path a student would hit), so its conclusions are directly about the shipped defaults rather than about a parallel harness that might drift from them.

**Diagnostics record follows Phase 5's format and location convention.** A new `diagnostics.md` in this change's directory, structured like Phase 5's (setup, baselines, per-step results, decision, Phase-7-facing inputs), so the two records read as one continuing story for whoever picks this up in Phase 7.

## Risks / Trade-offs

- **Tuning may not resolve the blobs at all**, leaving Phase 6 with a documented negative result rather than a fix. → Accepted per this session's scoping decision; the negative result plus its diagnosis (which constants were tried, what each did to reconstruction vs. classification) is exactly what makes a follow-up decoder-capacity change well-scoped instead of a guess.
- **Retuning the log-variance floor/multiplier could shift classification accuracy down while fixing reconstruction**, since both are read from the same latent and the current 0.94–0.95 classification numbers partly reflect the current (low) reconstruction weight. → Track both metrics together at every step, not reconstruction alone; a regression in classification is itself a diagnostic data point, not just a cost.
- **Real-dataset test runs are slower than stub-based ones** (Phase 5: CNN ~6s, VAE ~10s per run at defaults, hundreds of steps/epoch vs. the stub's one), which could slow the test suite if used indiscriminately. → Reserve real-dataset fixtures for the specific tests that need real content (per the Decisions section above); keep everything else on the stub.
- **A real dataset export is not committed to the repository** (Phase 5 decision, unchanged), so tests needing one must generate it via Dataset Forge as a fixture/setup step rather than loading a checked-in file. → Follow the same pattern Phase 5's own tests and smoke check already use for producing a real export on demand.

## Open Questions

None. The one genuine unknown at proposal time — whether `train()`'s default `epochs`/`batch_size`/`learning_rate` need to change for the real dataset — is investigated as tasks in this same change; if it concludes a change is warranted, `model-interface` and `shape-classifier` get a spec delta via `opsx:update` before archiving (proposal.md, Modified Capabilities), rather than being left open past this change.
