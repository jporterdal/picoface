# picoface Roadmap

This is the single, standing document for the whole `picoface` project plan. It exists outside any one OpenSpec change so it survives archiving: individual phases become their own changes (proposal/design/specs/tasks) under `openspec/changes/`, get implemented, and get archived, while this file keeps the full picture intact across all of them.

Originally captured in full inside the `picoface-phase0` change (`proposal.md` + `design.md` + `tasks.md`) during the initial `/opsx:explore` session on 2026-08-27. Split out here on 2026-08-27 so `picoface-phase0` could be narrowed to just Phase 0 (scaffolding) and later phases could become independent changes without losing the project-wide narrative.

## Overview

First-year non-CS students currently have no lightweight way to build and train a working neural network end to end without either drowning in framework/architecture detail or using a toy so abstracted it teaches nothing real. `picoface` closes that gap: a small, pip-installable, PyTorch-backed Python library where students assemble a real classifier and/or a real image generator from well-named function calls, train it in seconds to minutes on an old laptop CPU, and see it work — while an instructor-only, unrestricted tool produces the training data offline before term starts.

(Project renamed from `tinyface` to `picoface` on 2026-08-27 — `tinyface` collides with an existing, actively-maintained PyPI package. Repo, import name, and PyPI distribution name are all `picoface`.)

## Student-Visible Scope

Students only ever receive the finished Phase 7 deliverable: the packaged `picoface` library, its student-facing docs, and the per-arm template notebooks. They never see the phase-by-phase development history, the stub dataset (a pre-Phase-5 plumbing-validation aid, not real content), work-in-progress hyperparameter choices, or public functions kept only for API completeness rather than pedagogical use.

A function existing in the public API does not imply it appears in student-facing docs or notebooks — Phase 7 decides, per arm, what's actually walked through. Code that exists for developer/OpenSpec sequencing reasons rather than pedagogical ones only needs to work correctly, not be polished for a student audience, unless a later phase decides to surface it to students.

## Three-Arm Architecture

- **Arm 1 (student-facing): classifier** — recognize basic shapes/smiley face in tiny images. `build_classifier()`, `train()`, `evaluate()`, `predict()`, plus viz helpers.
- **Arm 2 (student-facing): generator** — produce images of the same shape classes via an autoencoder → VAE progression. `build_autoencoder()`, `build_vae()`, `train()`, `generate()`, plus latent-space viz.
- **Arm 3 (instructor-only): Dataset Forge** — unrestricted, modern-hardware, offline tool that procedurally renders the real training/testing image sets and ships them to students in a fixed data-contract format. Runs once before term starts.

A capstone linkage module ties Arms 1 and 2 together: `classify_generated()` (score VAE output with the trained classifier) and `activation_maximize()` (visualize what the classifier "imagines" for a class).

## Binding Constraint

Training must complete in seconds to minutes on an older CPU-only laptop — the worst case among students' machines. This is designed against directly (not GPU/Colab, which then becomes strictly easier, not the baseline).

## Capabilities

- **`data-contract`** — the shared image-dataset interchange format (`.npz` schema + `classes.json`), the `load_dataset()` entry point, and the synthetic stub dataset used to validate downstream plumbing before real content exists.
- **`shape-classifier`** — the student-assembled classification arm: building, training, evaluating, and running inference with a small CNN classifier via simple function calls, within an old-CPU time budget.
- **`shape-generator`** — the student-assembled generative arm: an autoencoder-to-VAE progression for producing images of the same shape classes, via simple function calls, within an old-CPU time budget.
- **`capstone-linkage`** — the functions that tie the classifier and generator arms together into a closing exercise (classifying generated images; activation-maximization visualization from the classifier).
- **`dataset-forge`** — the instructor-only, unrestricted offline tool that generates the real training/testing dataset and exports it in the `data-contract` format.
- **`packaging`** — the installable-package structure and naming (`picoface` repo, import name, and PyPI distribution name aligned) that supports pip/zip/Colab-git-clone distribution without committing to one channel. Its requirements (naming consistency, `pyproject.toml` + src layout, Dataset Forge excluded from the installable package) are purely structural and fully satisfied by Phase 0 — Phase 7's remaining work is *choosing and documenting* which channel to actually publish through, not adding new structural requirements.

No capability is a modification of an existing one — this is a new project with no prior specs.

## Goals

- Let a student assemble a working classifier and/or generator using only function calls and simple Python — no layer definitions, no hand-written training loops.
- **Guarantee the generative arm is reliable to train, not merely impressive when it happens to converge** — training must succeed unsupervised on a first-year student's laptop, not just in a demo run.
- Establish one explicit, versioned data contract between the unrestricted Dataset Forge and the constrained student library, so each side can be built and tested independently.
- Sequence implementation so real content decisions are deferred as late as possible, validated instead against a disposable synthetic stub dataset.
- Tie the classifier and generator arms together with a capstone exercise, without introducing a model-serialization format for MVP.
- Keep the distribution channel (PyPI / zip / Colab git-clone) an open, low-cost decision by committing only to standard packaging conventions now.

## Dependencies

PyTorch (CPU), numpy, matplotlib. Packaging via `pyproject.toml` with a src layout. Dataset Forge maintains its own separate dependency manifest, decoupled from the installable package.

## Key Design Decisions

- **PyTorch as the backend**, over Keras/TensorFlow — thinner CPU-only install, more predictable CPU performance on old hardware.
- **Autoencoder → VAE progression, GAN deferred.** GANs are unstable to train unsupervised on a student laptop; AE→VAE shares plumbing and adds one new idea (reparameterization + KL term) at a time. Diffusion and autoregressive (PixelCNN) approaches were considered and rejected as too slow/complex for this audience and hardware budget.
- **Public API / `_internals` boundary.** All `nn.Module` subclasses, loss functions, and training-loop code live in a non-public internals module; public modules (`datasets`, `classifier`, `generator`, `linkage`, `viz`) expose only named entry-point functions. This is the mechanism that makes "students never need to understand the details underneath" actually true.
- **Generic data contract:** `.npz` (`images`: uint8 N×H×W×C, `labels`: int array) + `classes.json`, parameterized by H/W/C/num_classes rather than hardcoded, so the loader and both arms' plumbing can be built and proven before real content decisions exist.
- **Disposable synthetic stub dataset** — small, arbitrary-dimension, 2–3 fake classes, shipped inside the package purely to exercise `load_dataset()` and all three arms before Dataset Forge exists. This is the mechanism (not just the intent) behind deferring taxonomy/resolution/noise decisions to Phase 5.
- **Linkage operates on in-memory model objects, not files.** No save/load or serialization format for MVP; capstone exercise runs within a single notebook session.
- **Packaging:** standard `pyproject.toml` + src layout; Dataset Forge lives in its own top-level directory with its own dependency manifest, entirely decoupled from the installable `picoface` package. Supports pip-from-PyPI, pip-from-zip, and Colab-git-clone without committing to one channel now.
- **Phase sequencing preserves deferral over dependency-minimality.** Dataset Forge (Phase 5) has no actual code dependency on Phases 2–4 — it's sequenced after them anyway so shape taxonomy, resolution, color depth, and noise policy stay undecided as long as possible. This is a project requirement, not a technical constraint, and should not be "optimized away."

## Non-Goals (project-wide)

- GAN implementation — documented future extension; requires its own future change.
- Diffusion or autoregressive (PixelCNN-style) generative approaches.
- Expression/emotion recognition — "smiley face" is one shape class among others, not a sub-taxonomy.
- GPU-specific optimization, multi-GPU, or distributed training.
- Publishing to PyPI or finalizing the distribution channel — packaging structure is in scope, the release process is not.
- Finalizing the real shape taxonomy, resolution, color depth, or noise/augmentation policy — deliberately deferred to Phase 5 (Dataset Forge).

## Phase Plan

Each phase assumes all prior phases are complete. A disposable stub dataset is used in Phases 2–4 so classifier/generator/linkage plumbing can be built and proven before the real shape taxonomy, resolution, or noise policy are decided in Phase 5.

| Phase | Name | Capability spec | Status |
|---|---|---|---|
| 0 | Scaffolding | `packaging` (partial) | **Done** — archived at `openspec/changes/archive/2026-08-28-picoface-phase0` |
| 1 | Data contract + stub dataset | `data-contract` | **Done** — archived at `openspec/changes/archive/2026-08-31-picoface-phase1` |
| 2 | Classifier arm (Arm 1) plumbing | `shape-classifier` | **In progress** — `openspec/changes/picoface-phase2` |
| 3 | Generator arm (Arm 2) plumbing | `shape-generator` | Not started |
| 4 | Arms linkage | `capstone-linkage` | Not started |
| 5 | Dataset Forge (Arm 3) + real content decisions | `dataset-forge` | Not started |
| 6 | End-to-end integration & tuning | — (no new capability; revisits 1–4) | Not started |
| 7 | Student docs & MVP packaging | `packaging` (remainder) | Not started |

Phase 5 has no functional dependency on Phases 2–4 — it's placed late deliberately to keep content decisions open as long as possible, not because of a technical blocker. Could be parallelized with 2–4 if resourcing allows.

Draft capability specs for Phases 1–5 (`data-contract`, `shape-classifier`, `shape-generator`, `capstone-linkage`, `dataset-forge`) were written during initial planning and now live in `openspec/draft-specs/` — a local-only, gitignored staging folder, not tracked in the repo. As each phase starts, copy that capability's draft spec into the new change (verifying it still holds) rather than rewriting from scratch. `packaging`, the one capability Phase 0 actually implements, stayed in `openspec/changes/picoface-phase0/specs/`.

### Phase 0 — Scaffolding
Installable package layout, module skeleton (`datasets.py`, `classifier.py`, `generator.py`, `linkage.py`, `viz.py`, `_internals/`), `dataset_forge/` as a separate top-level tool. No open questions block this.

### Phase 1 — Data contract + stub dataset
Generic `.npz` schema (images/labels/`classes.json`, parameterized by H/W/C/num_classes), `load_dataset()`, and a throwaway stub dataset for plumbing tests. Defers real taxonomy/resolution/noise decisions.

### Phase 2 — Classifier arm (Arm 1) plumbing
`build_classifier()`, `train()`, `evaluate()`, `predict()`, viz helpers; proven against the stub dataset. Resolves the student-facing function signatures for this arm.

### Phase 3 — Generator arm (Arm 2) plumbing
`build_autoencoder()` → `build_vae()`, `train()`, `generate()`, latent-space viz; proven against the stub dataset. Resolves VAE hyperparameter defaults.

- `latent_dim` is fixed at 2 (not student-configurable for MVP) — keeps `show_latent_space()` a direct scatter plot with no projection step (PCA/t-SNE), avoiding a new dependency beyond `torch`/`numpy`/`matplotlib`.
- The reconstruction/KL-divergence loss weight (β) is a fixed internal constant, not a student-facing parameter — placeholder value tuned against the stub dataset, flagged for experimental validation against real data in Phase 6 (see "Student-Visible Scope" above — students never see this constant or its tuning history either way).
- `generate()` is VAE-only. Calling it with a model from `build_autoencoder()` raises an explicit error naming the AE/VAE mismatch, rather than failing on a missing sampling method deep in `_internals` — same pattern as Phase 2's `ShapeError`.
- `build_autoencoder()` is kept at function/signature parity with `build_vae()` (same call shape, same arm) for API consistency, but per Student-Visible Scope only needs to work correctly — it does not need notebook-ready polish. Phase 7 decides whether the AE step is actually walked through in the arm's template notebook or the notebook goes straight to `build_vae()`.

### Phase 4 — Arms linkage
`classify_generated()`, `activation_maximize()`, consuming trained models from Phases 2–3. Resolves what the capstone tie-in exercise looks like.

### Phase 5 — Dataset Forge (Arm 3) + real content decisions
THIS is where shape taxonomy, resolution/color depth, and noise/augmentation policy actually get decided and built, unrestricted hardware/libs, exports the real dataset in the Phase 1 contract format.

### Phase 6 — End-to-end integration & tuning
Swap real dataset in for the stub across Arms 1/2/linkage, tune for "seconds-to-minutes on old CPU," set/verify an accuracy/quality bar. Also revisits Phase 3's placeholder VAE hyperparameters (β KL-divergence weight, `latent_dim=2`) against real data — the stub-dataset values were never meant to be final.

### Phase 7 — Student docs & MVP packaging
Docs stating explicit constraints, template notebooks per arm plus the capstone, finalize distribution mechanism (pip/zip/Colab clone).

## Risks / Trade-offs

- **VAE output may look too blurry to feel motivating.** → Keep resolution small enough that blur reads as expected, not a bug; document explicitly; GAN remains a documented stretch extension.
- **Plumbing validated only against a synthetic stub dataset (Phases 1–4) may hit unexpected issues once real, more visually complex content arrives in Phase 6.** → Phase 6 is explicitly scoped as an integration/tuning phase with room to revisit Phases 2–4; the stub dataset should include at least one deliberately non-trivial synthetic class.
- **The "seconds-to-minutes on old CPU" requirement is only fully validated at Phase 6.** → Measure and log wall-clock training time against the stub dataset from Phase 2 onward, so speed regressions surface early.
- **Dataset Forge's unrestricted dependencies could leak into or drift against the student package's environment.** → Dataset Forge maintains its own dependency manifest, entirely decoupled from `picoface`'s install.
- **"GAN as documented future work" could create scope-creep pressure mid-project.** → Explicitly a non-goal; any GAN work requires a new change proposal.

## Open Questions

- Exact shape taxonomy / class list — resolved in Phase 5.
- Exact resolution and color depth — resolved in Phase 5, guided by "as small as possible while still recognizable to both a human and the classifier."
- Noise/augmentation policy for Dataset Forge — resolved in Phase 5.
- Concrete accuracy/quality bar for the reference implementation — resolved in Phase 6, measured against the real dataset.
- VAE hyperparameter placeholders (β KL-divergence weight, `latent_dim=2`) set in Phase 3 against the stub dataset — revisited in Phase 6 against real data.
- Whether Phase 3's `build_autoencoder()` step is walked through in the generator arm's template notebook, or the notebook goes straight to `build_vae()` — resolved in Phase 7.
- Final distribution channel (PyPI vs. zip vs. Colab git-clone) — resolved in Phase 7; the packaging structure already supports all three, so this is a low-stakes choice deferred on purpose.
