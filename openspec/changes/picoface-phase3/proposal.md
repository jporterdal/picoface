## Why

Phase 2 delivered the classifier arm (Arm 1); the roadmap's Phase 3 is next: the generator arm (Arm 2), which lets a student build and train an autoencoder-to-VAE progression and generate new shape images, using only named function calls — no `nn.Module` authoring, no hand-written training loop, matching the pattern already established by the classifier arm.

## What Changes

- Add `build_autoencoder()`: constructs a plain (non-variational) encoder/decoder model from a `Dataset`, as a pedagogical stepping stone toward the VAE. Kept at function/signature parity with `build_vae()`, but only needs to work correctly — it is not required to be notebook-polished (Phase 7 decides whether it's walked through in the student-facing notebook or the notebook goes straight to `build_vae()`).
- Add `build_vae()`: constructs a VAE (probabilistic latent space, reparameterization trick, combined reconstruction + KL-divergence loss) from a `Dataset`, with the same call shape as `build_autoencoder()` so a student can swap one for the other and re-run `train()` unchanged.
- Extend `train()` (already public in `classifier.py`'s style, but generator arm gets its own internals) to work against autoencoder/VAE models built by the two functions above, sharing the existing wall-clock-time-logging pattern from Phase 2.
- Add `generate()`: samples `n` new images from a trained VAE's latent space. VAE-only — calling it with a `build_autoencoder()` model raises an explicit error naming the AE/VAE mismatch, rather than failing on a missing sampling method deep in `_internals` (mirrors Phase 2's `ShapeError` pattern).
- Add `show_latent_space()` viz helper: scatter-plots a trained VAE's 2D latent encoding of a `Dataset`, colored by class. `latent_dim` is fixed at 2 (not student-configurable) so no projection step (PCA/t-SNE) or new dependency is needed.
- Internally, the reconstruction/KL-divergence loss weight (β) is a fixed constant tuned against the stub dataset — not a student-facing parameter. It and `latent_dim=2` are explicitly flagged as placeholders to revisit in Phase 6 against real data.
- All new model/training code lives under `_internals/generator_internals.py`, following the Phase 2 `_internals` boundary convention. Shares no model code with the classifier arm.

## Capabilities

### New Capabilities
- `shape-generator`: the student-assembled generative arm — building, training, and generating from an autoencoder-to-VAE progression, plus latent-space visualization, via simple function calls, within an old-CPU time budget. Draft spec already exists at `openspec/draft-specs/shape-generator/spec.md`; this change formalizes and resolves its open points (`latent_dim`, β, AE/VAE `generate()` scope, notebook-visibility deferral).

### Modified Capabilities
(none — `shape-generator` is new; no existing spec's requirements change)

## Impact

- New module code: `src/picoface/generator.py` (currently an empty Phase 0 stub — fills it in), `src/picoface/_internals/generator_internals.py` (new).
- `viz.py` gains `show_latent_space()` alongside Phase 2's `plot_training_history()`.
- No new dependencies (`torch`/`numpy`/`matplotlib` only, per the `latent_dim=2` decision avoiding a PCA/t-SNE library).
- Tests proven against the existing stub dataset (`_internals/stub_data.py`), consistent with Phase 2's approach; no interaction with real dataset content (deferred to Phase 5).
- `openspec/ROADMAP.md` already updated (in the preceding explore session) with the `latent_dim`/β placeholder-vs-Phase-6 note and the new "Student-Visible Scope" section this proposal's AE-parity decision depends on.
