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

- **Arm 1: classifier (CNN)** — recognize basic shapes/smiley face in tiny images. `build_classifier()`, `train()`, `evaluate()`, `predict()`, plus viz helpers. Since Phase 3c it is **no longer the student's main path**: it is the capstone's *independent judge*, a model trained separately from the generator so that scoring generated images with it means something.
- **Arm 2: the student's model (supervised VAE)** — since Phase 3c, one model that both classifies and generates from a single `train()` call on labeled data. `build_vae()`, `train()`, `evaluate()`, `predict()`, `generate()`. Its classifier reads the same probabilistic latent space that `generate()` samples from. `build_autoencoder()` remains as an optional, reconstruction-only model; it is not a step on the way to the VAE, and it does not classify.
- **Arm 3 (instructor-only): Dataset Forge** — unrestricted, modern-hardware, offline tool that procedurally renders the real training/testing image sets and ships them to students in a fixed data-contract format. Runs once before term starts.

A capstone linkage module ties Arms 1 and 2 together: `classify_generated()` (generate images for each class from the VAE and check whether the independently trained CNN agrees) and `activation_maximize()` (visualize what a classifier "imagines" for a class).

## Binding Constraint

Training must complete in seconds to minutes on an older CPU-only laptop — the worst case among students' machines. This is designed against directly (not GPU/Colab, which then becomes strictly easier, not the baseline).

## Capabilities

- **`data-contract`** — the shared image-dataset interchange format (`.npz` schema + `classes.json`), the `load_dataset()` entry point, and the synthetic stub dataset used to validate downstream plumbing before real content exists.
- **`shape-classifier`** — the student-assembled classification arm: building, training, evaluating, and running inference with a small CNN classifier via simple function calls, within an old-CPU time budget.
- **`shape-generator`** — the student's model: a supervised VAE that classifies and generates images of the shape classes from one `train()` call, plus an optional reconstruction-only autoencoder, via simple function calls, within an old-CPU time budget.
- **`model-interface`** — the model-agnostic public verbs (`train`, `evaluate`, `predict`, `generate`) and the abstract model contract behind them, so a call's shape never depends on which kind of model was built. Added in Phase 3b.
- **`capstone-linkage`** — the functions that tie the classifier and generator arms together into a closing exercise (classifying class-targeted generated images with the independent CNN; activation-maximization visualization from any classifying model). Added in Phase 4.
- **`dataset-forge`** — the instructor-only, unrestricted offline tool that generates the real training/testing dataset and exports it in the `data-contract` format.
- **`packaging`** — the installable-package structure and naming (`picoface` repo, import name, and PyPI distribution name aligned) that supports pip/zip/Colab-git-clone distribution without committing to one channel. Its requirements (naming consistency, `pyproject.toml` + src layout, Dataset Forge excluded from the installable package) are purely structural and fully satisfied by Phase 0 — Phase 7's remaining work is *choosing and documenting* which channel to actually publish through, not adding new structural requirements.

Phases 0–3 each introduced new capabilities. Phase 3b adds `model-interface` and is the first change to *modify* existing capabilities (`shape-classifier`, `shape-generator`). Phase 3c modifies `shape-generator`, `shape-classifier`, `model-interface`, and `data-contract`. Phase 4 adds `capstone-linkage` and modifies `model-interface`.

## Goals

- Let a student assemble a working classifier and/or generator using only function calls and simple Python — no layer definitions, no hand-written training loops.
- **Guarantee the generative arm is reliable to train, not merely impressive when it happens to converge** — training must succeed unattended on a first-year student's laptop, not just in a demo run.
- Establish one explicit, versioned data contract between the unrestricted Dataset Forge and the constrained student library, so each side can be built and tested independently.
- Sequence implementation so real content decisions are deferred as late as possible, validated instead against a disposable synthetic stub dataset.
- Tie the classifier and generator arms together with a capstone exercise, without introducing a model-serialization format for MVP.
- Keep the distribution channel (PyPI / zip / Colab git-clone) an open, low-cost decision by committing only to standard packaging conventions now.

## Dependencies

PyTorch (CPU), numpy, matplotlib. Packaging via `pyproject.toml` with a src layout. Dataset Forge maintains its own separate dependency manifest, decoupled from the installable package.

## Key Design Decisions

- **PyTorch as the backend**, over Keras/TensorFlow — thinner CPU-only install, more predictable CPU performance on old hardware.
- **VAE as the generator, GAN deferred.** GANs are unstable to train on a student laptop without close supervision; a VAE trains reliably and shares plumbing with the plain autoencoder. Diffusion and autoregressive (PixelCNN) approaches were considered and rejected as too slow/complex for this audience and hardware budget. (Phase 3 framed the autoencoder as a stepping stone to the VAE; Phase 3c drops that framing.)
- **Public API / `_internals` boundary.** All `nn.Module` subclasses, loss functions, and training-loop code live in a non-public internals module; public modules (`datasets`, `classifier`, `generator`, `linkage`, `viz`) expose only named entry-point functions. This is the mechanism that makes "students never need to understand the details underneath" actually true.
- **Model-agnostic verbs over an abstract model (Phase 3b).** `train`, `evaluate`, `predict`, and `generate` have one implementation in an arm-neutral internal module; each model type defines its own loss (`training_step`) and declares optional capabilities (`classify`, `sample`). The arm modules re-export the verbs (`classifier.train is generator.train`). Phase 3c adds no-op hooks through which a model schedules its own loss terms (`on_epoch_start`), sets per-parameter learning rates (`optimizer_param_groups`), and constrains its parameters after each step (`on_step_end`), so `train()` never branches on model type. This replaces the earlier "arms share no code" stance, which protected decoupling that the neutral module now provides.
- **~~Joint classifier branch sits on the shared conv-trunk features, not the latent.~~** *Superseded by Phase 3c.* Phase 3b chose the trunk from a synthetic-shapes spike (best classifier, reconstruction untouched). But a trunk head puts no class structure into the latent that `generate()` samples from.
- **The VAE's classifier reads the probabilistic latent (Phase 3c).** It reads the reparameterized sample `z` during training and the mean `mu` for inference, so supervision shapes the distribution `generate()` samples from. The loss is a per-pixel ELBO: reconstruction is the per-pixel Gaussian negative log-likelihood, and KL is divided by H·W·C and annealed from 0 to 1. Reconstruction and classification are balanced by learned uncertainty weighting (Kendall et al., 2018), with no hand-tuned loss weights. See `openspec/changes/archive/2026-09-22-picoface-phase3c/design.md`.
- **Class-targeted generation for the capstone uses the supervised latent's empirical clusters (Phase 4).** `classify_generated()` encodes the labeled training data, takes each class's mean and spread of `mu`, samples near it, and decodes. This gives each generated image an intended class without a label-conditioned decoder, retraining, or a public `generate(cls=...)`. The mechanism stays internal to `classify_generated()`. It reaches the VAE through a new `latent_access` capability (encode to `mu`, decode any `z`) on the abstract model, gated like `classify`/`sample`, so `linkage` never touches a concrete model class. See `openspec/changes/archive/2026-09-22-picoface-phase4/design.md`.
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
| 2 | Classifier arm (Arm 1) plumbing | `shape-classifier` | **Done** — archived at `openspec/changes/archive/2026-08-31-picoface-phase2` |
| 3 | Generator arm (Arm 2) plumbing | `shape-generator` | **Done** — archived at `openspec/changes/archive/2026-08-31-picoface-phase3` |
| 3b | Joint classifier/generator model + model-agnostic verbs | `model-interface` (new); `shape-classifier`, `shape-generator` (modified) | **Done** — archived at `openspec/changes/archive/2026-09-21-picoface-phase3b-joint-model`; model decisions superseded by 3c |
| 3c | Supervised VAE: one model that classifies and generates | `shape-generator`, `shape-classifier`, `model-interface`, `data-contract` (modified) | **Done** — archived at `openspec/changes/archive/2026-09-22-picoface-phase3c` |
| 4 | Arms linkage | `capstone-linkage` (new); `model-interface` (modified) | **Done** — archived at `openspec/changes/archive/2026-09-22-picoface-phase4` |
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

- ~~`latent_dim` is fixed at 2~~ *(superseded by Phase 3c: unpinned, and `show_latent_space()` removed)* — was fixed so `show_latent_space()` could be a direct scatter plot with no projection step (PCA/t-SNE).
- ~~The reconstruction/KL-divergence loss weight (β) is a fixed internal constant~~ *(superseded by Phase 3c: the KL term is on a per-pixel ELBO scale and annealed to the ELBO weight of 1; there is no β)* — was a placeholder tuned against the stub dataset. Students never see any of these internal constants either way (see "Student-Visible Scope").
- `generate()` is VAE-only. Calling it with a model from `build_autoencoder()` raises an explicit error naming the AE/VAE mismatch, rather than failing on a missing sampling method deep in `_internals` — same pattern as Phase 2's `ShapeError`.
- `build_autoencoder()` is kept at function/signature parity with `build_vae()` (same call shape, same arm) for API consistency, but per Student-Visible Scope only needs to work correctly — it does not need notebook-ready polish. Since Phase 3c it is documented as optional and not a prerequisite for `build_vae()`.

### Phase 3b — Joint model + model-agnostic verbs
Adds a classification branch (a small MLP head on the shared conv-trunk features) to both `build_autoencoder()` and `build_vae()` models, trained jointly with reconstruction (+ KL for the VAE) via one `train()` call; loss = `MSE + β·KL + λ·CE` with fixed internal `λ = 1.0`. Behind it, a behavior-preserving refactor (a hard prerequisite, done first) puts every model on an abstract `_Model` so `train`/`evaluate`/`predict`/`generate` are model-agnostic: `classifier.evaluate`/`predict` work on the joint models, and `generate` raises a clear capability error for any model that can't sample. `TrainingHistory` is unified and gains `classification_loss`/`accuracy`; `plot_training_history` plots accuracy when present.

Deliberate non-goals: the joint branch does **not** make `show_latent_space()` show class clusters, and there is no class-conditional `generate(cls=...)`. Whether Arm 1 remains a separate student stage is left to Phase 7.

*Superseded by Phase 3c:* the head's placement on the trunk, the fixed `λ = 1.0`, and the autoencoder's classification branch. 3b's model-agnostic refactor stands and is what 3c builds on.

### Phase 3c — Supervised VAE
Makes the VAE the student's model: labeled data in, and one model out that classifies (`evaluate()`/`predict()`) and generates (`generate()`) after one `train()` call. The CNN stays as the capstone's independent judge. The plain autoencoder returns to reconstruction-only. See `openspec/changes/archive/2026-09-22-picoface-phase3c/` (proposal, design, specs, tasks).

- The classification head reads the latent: the sampled `z` during training, `mu` for inference.
- The loss is a per-pixel ELBO: Gaussian reconstruction likelihood with a learned per-pixel noise scale, plus KL divided by H·W·C and annealed from 0 to 1 over the first half of training. Classification is added with learned uncertainty weighting (Kendall et al., 2018). The KL term is not uncertainty-weighted. No hand-tuned loss weight remains.
- The learned log-variances train at 10× the model's learning rate, because under Adam they otherwise barely move in a default-length run. They have a floor, enforced on the stored parameters after each step.
- `latent_dim` is unpinned (provisionally 8), and `show_latent_space()` is removed.
- The stub dataset gains a `kind="shapes"` mode whose classes are separable only by spatial arrangement: every figure covers the same number of pixels, so mean brightness is uninformative.
- `TrainingHistory` records per-epoch KL weight and learned log-variances. The Phase 6 baseline tables (accuracy, losses, learned weights, latent-dim trend, wall-clock) are in `openspec/changes/archive/2026-09-22-picoface-phase3c/diagnostics.md`. They are stub-dataset numbers, not predictions about real content.

### Phase 4 — Arms linkage
`classify_generated()`, `activation_maximize()`, consuming trained models from Phases 2–3c. Resolves what the capstone tie-in exercise looks like. Since Phase 3c, classification ships with the student's model rather than after it, so Phase 4 is purely the linkage exercise. See `openspec/changes/archive/2026-09-22-picoface-phase4/` (proposal, design, specs, tasks).

- `classify_generated(classifier_model, generator_model, data, n=10)` makes `n` images per class from the VAE and scores them with the **independently trained CNN** (Phase 3c). A model grading its own samples with its own head would be circular. It returns a `GeneratedImagesReport`: per-class and overall agreement between intended and predicted class, plus the images themselves.
- The draft spec's "intended class" did not fit an unconditional `generate()`. Resolved by sampling near each class's empirical latent cluster, found by encoding the labeled `data` (see Key Design Decisions). `generate()` itself stays unconditional.
- `classify_generated()` rejects a classifier, generator, and `data` that disagree on class names or image shape, with a named error. Classes are compared by name, so their order need not match.
- `activation_maximize(model, target_class)` works on any model with a `classify` capability, CNN or VAE. It runs gradient ascent on the target class's raw logit from uniform noise (Adam, 200 steps at 0.05, pixels clamped to `[0, 1]`), without touching the model's parameters or their gradients. It is deliberately unregularized for now (see Phase 6).
- On the three-class stub, agreement was 0.33–0.75 over 5 seeds, typically with one or two classes near 1.0 and the rest near 0. The class clusters were well separated, so the weak link is decoder sample quality, not the targeting. Tests assert the report's shape, never a value.

### Phase 5 — Dataset Forge (Arm 3) + real content decisions
THIS is where shape taxonomy, resolution/color depth, and noise/augmentation policy actually get decided and built, unrestricted hardware/libs, exports the real dataset in the Phase 1 contract format.

### Phase 6 — End-to-end integration & tuning
Swap real dataset in for the stub across Arms 1/2/linkage, tune for "seconds-to-minutes on old CPU," set/verify an accuracy/quality bar. Also revisits the placeholder model constants against real data — the stub-dataset values were never meant to be final. From Phase 3c: `latent_dim` (provisionally 8), the KL annealing schedule's length and shape (its end point is fixed at 1), the log-variance floor, the log-variances' learning-rate multiplier (conservatively 10×; retune once the real dataset's steps per epoch are known, and check the learned weights reach equilibrium within the default epochs), whether to add weight decay (decoupled `AdamW`, excluding the log-variances), the classification head's architecture, and whether a label-conditioned decoder becomes workable. Start from the Phase 3c diagnostics record (`openspec/changes/archive/2026-09-22-picoface-phase3c/diagnostics.md`). Also verify that `train()`'s default epochs give high classification accuracy on the real dataset — the 16-image stub is one optimizer step per epoch and too small for the defaults to show it. From Phase 4: whether `activation_maximize()` needs regularization (e.g. an L2 or blur penalty) to produce recognizable images rather than adversarial noise, its ascent step count and learning rate, how `classify_generated()` samples around a class's latent cluster (currently mean plus per-dimension spread of `mu`), and whether `classify_generated()`'s agreement becomes informative once the generator is tuned.

### Phase 7 — Student docs & MVP packaging
Docs stating explicit constraints, template notebooks per arm plus the capstone, finalize distribution mechanism (pip/zip/Colab clone).

## Risks / Trade-offs

- **VAE output may look too blurry to feel motivating.** → Keep resolution small enough that blur reads as expected, not a bug; document explicitly; GAN remains a documented stretch extension.
- **Plumbing validated only against a synthetic stub dataset (Phases 1–4) may hit unexpected issues once real, more visually complex content arrives in Phase 6.** → Phase 6 is explicitly scoped as an integration/tuning phase with room to revisit Phases 2–4. The non-trivial synthetic classes are implemented as of Phase 3c: `kind="shapes"`, separable only by spatial arrangement, with a test that mean brightness alone does no better than chance.
- **The "seconds-to-minutes on old CPU" requirement is only fully validated at Phase 6.** → Measure and log wall-clock training time against the stub dataset from Phase 2 onward, so speed regressions surface early.
- **Uncertainty weighting can starve reconstruction** (Phase 3c design, Decision 4). Lower loss earns higher weight, so on easily classified data the weighting can pour capacity into the already-solved classification task. → Bounded by a floor on the learned log-variances. Per-epoch learned weights are recorded so the dynamic is visible. Not observed on the stub (reconstruction kept improving with classification at the floor); Phase 6 checks it on real data.
- **The learned loss weights may lag their equilibrium in short runs** (Phase 3c design, Decision 4). Adam moves them by about their learning rate per step. → They train at a 10× learning-rate multiplier. On the stub, default-length runs still lag; Phase 6 retunes the multiplier once real steps per epoch are known.
- **Classifying through the probabilistic bottleneck caps the VAE's accuracy below the standalone CNN's.** → Accepted (Phase 3c, Decision 1): it is the only placement where supervision reaches what `generate()` samples. The CNN stays as the independent judge, and `latent_dim` is the main relief valve. On the stub, both reach 1.00 held-out at 30 epochs.
- **The capstone may underwhelm on first contact** (Phase 4). `classify_generated()` agreement is only as good as the VAE's samples (0.33–0.75 on the stub), and unregularized `activation_maximize()` may produce noise a human can't read as the class. → Both are recorded as Phase 6 questions against real data, and neither affects the function contracts. Phase 7 decides how the notebook frames the numbers.
- **Dataset Forge's unrestricted dependencies could leak into or drift against the student package's environment.** → Dataset Forge maintains its own dependency manifest, entirely decoupled from `picoface`'s install.
- **"GAN as documented future work" could create scope-creep pressure mid-project.** → Explicitly a non-goal; any GAN work requires a new change proposal.

## Open Questions

- Exact shape taxonomy / class list — resolved in Phase 5.
- Exact resolution and color depth — resolved in Phase 5, guided by "as small as possible while still recognizable to both a human and the classifier."
- Noise/augmentation policy for Dataset Forge — resolved in Phase 5.
- Concrete accuracy/quality bar for the reference implementation — resolved in Phase 6, measured against the real dataset.
- ~~VAE hyperparameter placeholders (β, `latent_dim=2`)~~ — resolved in Phase 3c: `latent_dim` is unpinned, and there is no β (the KL weight is the ELBO weight of 1 on a per-pixel scale, reached by annealing). What remains open for Phase 6: the annealing schedule's length and shape, the log-variance floor, the log-variances' learning-rate multiplier, and whether to add weight decay.
- Whether Phase 3's `build_autoencoder()` step is walked through in the generator arm's template notebook, or the notebook goes straight to `build_vae()` — resolved in Phase 7.
- ~~Whether Arm 1 remains a separate student stage~~ — resolved in Phase 3c: the supervised VAE is the student's model, and the CNN is the capstone's independent judge.
- ~~Phase 3b's `λ` and head placement~~ — resolved in Phase 3c (learned uncertainty weighting; head on the latent). The head's architecture and a label-conditioned decoder are revisited in Phase 6.
- ~~What the capstone exercise looks like, and how `classify_generated()` gets an intended class from an unconditional generator~~ — resolved in Phase 4: class-targeted sampling from the supervised latent's empirical clusters, internal to `classify_generated()`.
- Whether `viz` gains helpers to show a `GeneratedImagesReport`'s images and an `activation_maximize()` image, or the capstone notebook plots them directly — resolved in Phase 7.
- Final distribution channel (PyPI vs. zip vs. Colab git-clone) — resolved in Phase 7; the packaging structure already supports all three, so this is a low-stakes choice deferred on purpose.
