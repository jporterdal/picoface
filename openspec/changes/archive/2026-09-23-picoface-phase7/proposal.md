## Why

`openspec/ROADMAP.md`'s Phase 7 ("Student docs & MVP packaging") is the last unstarted phase and has been unstarted since the roadmap was written. An `/opsx:explore` session broke it into five pieces and got explicit user scoping decisions on each: two are out of this change's scope entirely (real-dataset distribution to students, handled by the user outside this project; template notebooks, downgraded to a post-MVP nice-to-have), and three are real, doable now — choosing/preparing a distribution channel for the `picoface` package itself, writing the student-facing docs that now carry the full teaching load since notebooks are deferred, and documenting (not gating) expected result ranges. This change does that remaining real work and updates the ROADMAP to reflect the scoping decisions, closing out Phase 7 as scoped.

## What Changes

- **README.md rewrite**: corrects the stale "Status: early scaffolding (Phase 0)" line; adds an explicit constraints section (CPU-only always, no GPU required or used, seconds-to-minutes training); adds a quickstart code snippet covering the walked-through student API (`build_classifier`/`build_vae`/`train`/`evaluate`/`predict`/`generate`/`classify_generated`/`activation_maximize`), noting `build_autoencoder()`/`build_classifier_from_shape()` exist but aren't part of that walkthrough; notes `load_dataset()`'s contract (`.npz` + `classes.json`) without picoface itself providing a real dataset; adds a documented, non-gating "what to expect" results table; notes template notebooks are a planned nice-to-have, not shipped; updates the install section to note a future PyPI release as the primary long-term channel, pointing to `RELEASING.md`.
- **New root-level `RELEASING.md`**: the user's own future manual steps for a real PyPI publish (name-availability check, account + API token, TestPyPI dry run, `python -m build`, `twine upload`, versioning/tagging for future releases), plus a note that the CPU-torch-wheel install order still applies regardless of channel.
- **`pyproject.toml` publish-readiness**: fills in PyPI-facing metadata (classifiers, keywords) that's currently thin; adds `build` and `twine` to the `dev` extras. Verified locally by running `python -m build` and confirming a clean sdist + wheel — no actual publish happens in this change.
- **`openspec/ROADMAP.md` updates**: revises the Non-Goals line that currently rules out any PyPI-publishing work, to distinguish *preparing* for a release (now in scope) from *performing* the actual upload (still the user's own future manual action, not this repo's automation); resolves the "final distribution channel" Open Question as PyPI (target), zip/git-clone remaining accurate as what works today; resolves the accuracy/quality-bar Open Question as documented-ranges-only, no automated gate; downgrades (not resolves) the two notebook-content Open Questions to explicitly deferred/non-blocking; updates the Phase 7 phase-table row to reflect the real scope.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
None. `packaging`'s existing spec (`openspec/specs/packaging`, 3 requirements) is structural — naming consistency, installable-via-multiple-channels, Dataset Forge excluded from the installable package — and was already fully satisfied by Phase 0. Choosing PyPI as the target channel and preparing the repo for a future publish doesn't change any of those three requirements' text; it's the non-structural "choosing and documenting" work the capability's own description already called out as Phase 7's remaining item. No behavior any spec describes is changing, so this change sets `skip_specs: true`, matching the precedent already established by `capstone-linkage-sampling-fix`.

## Impact

- `README.md`, new `RELEASING.md`, `pyproject.toml`, `openspec/ROADMAP.md`.
- No changes to `src/picoface/` or `dataset_forge/`, and no runtime behavior changes — no existing test should be affected. The one new verification this change introduces is that `python -m build` succeeds locally.
- Closes out Phase 7 as scoped (real-dataset distribution and template notebooks are intentionally left open/deferred by the user's own decision, not this change's job to resolve).
