## Context

See `proposal.md` for the why and the full scoping breakdown. This covers the design-level choices behind the three real pieces (distribution channel, student docs, expected-results reporting) that came out of an `/opsx:explore` session's back-and-forth with the user.

**Current state this change starts from:**
- `README.md` is still a Phase-0-era stub: an accurate but minimal "Install (development)" section and a `Layout` list. No constraints, no quickstart, no expected results — nothing a student reading it today could actually learn the API from.
- `pyproject.toml` has the bare minimum PyPI-facing metadata (name, version, description, readme, license file, `requires-python`, dependencies, one project URL). No classifiers, no keywords, no `build`/`twine` in `dev` extras.
- `openspec/ROADMAP.md`'s Non-Goals section currently states flatly that "Publishing to PyPI or finalizing the distribution channel" is out of scope project-wide. Its Open Questions section currently marks the two notebook-content questions (`build_autoencoder()` walkthrough; `viz` report-plotting helpers) as "resolved in Phase 7" — which was aspirational language from the original roadmap draft, not an actual resolution; nothing about their substance has been decided, and per the user's scoping call in this session they shouldn't be marked resolved at all right now.
- No template notebooks exist anywhere in the repo (`find . -iname "*.ipynb"` is empty) and, per the user's decision, none are being added by this change.

## Goals / Non-Goals

**Goals:**
- Give students one accurate, self-contained doc (README) that states real constraints, shows a runnable quickstart against the walked-through API surface, and reports non-gating expected-result ranges — since no notebook exists to carry any of that.
- Leave the repo one command away from a real PyPI publish (clean `python -m build` output, complete metadata) without performing that publish or requiring the user's PyPI credentials in this session.
- Make `ROADMAP.md` accurately reflect all five Phase-7 pieces' actual disposition, so a future reader doesn't have to reconstruct this session's scoping conversation to understand why some Open Questions are resolved, one is deferred, and one is explicitly out of scope.

**Non-Goals:**
- Actually publishing to PyPI — account creation and `twine upload` are the user's own future manual action (`RELEASING.md` documents the steps; this change doesn't run them).
- Building any mechanism for getting the real dataset to students (piece 2) — explicitly the user's responsibility outside this project.
- Building template notebooks (piece 3) — explicitly downgraded to deferred/nice-to-have, not attempted here.
- Any automated accuracy/quality check — the user chose documented ranges only after being shown what an automated gate would involve.
- CI/release-automation (e.g. a GitHub Actions publish-on-tag workflow) — that's *running* the release process, not *preparing* for it; stays manual per the user's own request.

## Decisions

**PyPI as the target/primary distribution channel**, with the current git-clone-or-zip + `pip install -e .` path staying documented as what actually works today. Alternatives considered: a Colab git-clone workflow was the initial recommendation (zero local setup for the non-CS target audience), but the user has an actual PyPI preference and only needed the mechanics explained before committing to it — this follows that stated preference rather than the audience-fit argument. A separately-documented "zip download" path was considered and dropped: it's functionally identical to a git clone once you're at `pip install -e .`, so it doesn't need its own instructions.

**`RELEASING.md` as a separate root-level file, not folded into README.** README is the student's onboarding doc; `RELEASING.md` is a maintainer-only runbook the user follows once, later, with their own PyPI account. Keeping them separate means a student never has to read past irrelevant publish mechanics, and the release runbook doesn't get lost inside a change/design doc under `openspec/` (which won't survive the same way once this change archives — the ROADMAP.md precedent of "individual phases get archived, but the standing doc survives" argues for putting operational content in a file that isn't a change artifact).

**`pyproject.toml` gets only the metadata a clean PyPI listing needs (classifiers, keywords) plus `build`/`twine` in `dev` extras** — not a release-automation workflow. This is the line between "preparing the repo" (in scope) and "running the release process" (still the user's manual action, per their own request in this session).

**Accuracy/quality bar is documented ranges only, sourced from already-recorded diagnostics — no automated check, and no new measurement run.** This change makes no code changes, so there is nothing new to measure; the numbers already sitting in `capstone-linkage-sampling-fix/diagnostics.md` and `picoface-phase6/diagnostics.md` are the accurate, current ones to publish. This also matches the project's existing house style: Phase 5's smoke check and Phase 6's tuning sweeps were both explicitly framed as measurement, never pass/fail, and the user confirmed that's the intent here too after seeing what a hard gate would actually require (an automated script/test failing loudly below a threshold).

**The two notebook-content Open Questions get marked deferred, not resolved-with-a-placeholder or deleted.** Piece 3 (notebooks) is a real, valid future capability that the user downgraded for scope reasons, not an idea being rejected — the ROADMAP should keep the question visibly open for whoever picks it up later, rather than implying a decision was made against it.

## Risks / Trade-offs

- **[Risk] Naming PyPI as "the chosen target channel" in the ROADMAP before actually verifying `picoface` is free on PyPI.** This project already renamed itself once (`tinyface` → `picoface`) over exactly this kind of collision, so it's a real possibility, not a hypothetical. → Mitigation: `RELEASING.md`'s very first step is verifying name availability before anything else; the ROADMAP language frames PyPI as the target, not an already-secured fact, and the git-clone/zip path stays fully documented and functional regardless of how that check turns out — nothing breaks if the name has to change or the plan reverts to zip/Colab.
- **[Risk] Documented expected-result ranges can go stale** the next time a change touches sampling, tuning, or the dataset, and nobody remembers to update README. → Accepted: this is the same risk profile every numbers-in-docs situation in this project already carries (e.g. ROADMAP's own historical measurement callouts), and building tooling to keep README's numbers in sync automatically would be new scope disproportionate to what is otherwise a docs-only change.
- **[Risk] The new quickstart still leaves a student with no real dataset to run it against** (piece 2 is explicitly out of scope). → Accepted per the user's own scoping decision; the doc should be honest about this rather than paper over it — the quickstart demonstrates the API against a caller-supplied `Dataset`, and states `load_dataset()`'s contract without implying picoface provides real data.

## Migration Plan

No migration. This is a docs- and packaging-metadata-only change: no persisted state, no schema, no runtime behavior change. Rollback is a straight revert. The new `dev` extras (`build`, `twine`) are additive and don't affect the installed package's runtime dependencies.
