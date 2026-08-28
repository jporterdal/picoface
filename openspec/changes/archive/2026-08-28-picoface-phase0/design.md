## Context

This change is **Phase 0 (Scaffolding)** of the `picoface` project — see [`openspec/ROADMAP.md`](../../ROADMAP.md) for the full project context, architecture, and 8-phase plan. `picoface` is a greenfield project: there is no existing code, no existing specs, and no commits in the repository yet.

Phase 0's job is narrow: produce a real, installable package skeleton that later phases build capability logic into, and structurally separate Dataset Forge from the installable package from the very start. No classifier, generator, linkage, or dataset-forge logic is implemented here — only the shell.

## Goals / Non-Goals

**Goals:**
- Produce a working, installable `picoface` package skeleton that `pip install -e .` and `import picoface` succeed against, even with empty modules.
- Get the naming right immediately: repo, `import` name, and PyPI distribution name must all be `picoface` — this is much cheaper to fix now than after later phases add real code.
- Physically separate Dataset Forge (its own top-level directory, own dependency manifest) so its unrestricted dependencies can never leak into the installable package, from day one.
- Choose a packaging structure (`pyproject.toml` + src layout) that keeps pip/zip/Colab-git-clone all viable without committing to one now.

**Non-Goals:**
- Any classifier, generator, linkage, or Dataset Forge logic — those are Phases 1–5, each their own future change.
- Finalizing which distribution channel is actually used (PyPI vs. zip vs. Colab clone) — the structure supports all three; the choice itself is Phase 7.
- Any real dataset content decisions (taxonomy, resolution, noise policy) — Phase 5.

## Decisions

**Packaging: standard `pyproject.toml` + src layout, Dataset Forge kept outside the installable package.** Repo name, `import` name, and PyPI distribution name are all `picoface`. Dataset Forge lives in its own top-level directory with its own dependency manifest, so installing `picoface` never pulls in Dataset Forge's unrestricted (and likely heavier) dependency set. This one structural decision supports pip-from-PyPI, pip-from-local-zip, and Colab-git-clone distribution without picking one now.

**Dependencies: PyTorch (CPU), numpy, matplotlib.** Chosen project-wide (see ROADMAP.md for the PyTorch-vs-Keras rationale); Phase 0 just wires them into `pyproject.toml`.

**Module skeleton follows the public/`_internals` boundary from day one.** `src/picoface/{datasets,classifier,generator,linkage,viz}.py` are the public surface; `_internals/` is where later phases will put `nn.Module` subclasses and training loops. Creating this split now, even with empty files, means later phases never have to retrofit it.

## Risks / Trade-offs

- **[Risk]** Dataset Forge's unrestricted dependencies could leak into or drift against the student package's environment if not kept strictly separate. → **Mitigation:** this change is what establishes that separation — a distinct top-level `dataset_forge/` directory with its own dependency manifest, decoupled from `picoface`'s install, before any Dataset Forge code exists to tempt otherwise.

(Other project-wide risks — VAE output quality, stub-dataset-only validation, late speed validation, GAN scope creep — apply to later phases and are tracked in [`openspec/ROADMAP.md`](../../ROADMAP.md), not here.)

## Migration Plan

Not applicable — greenfield project, no existing users, data, or deployed version to migrate from. Rollout is simply implementing Phase 0's tasks; nothing is deployed or released as part of this change beyond repository content.

## Open Questions

None specific to Phase 0. See [`openspec/ROADMAP.md`](../../ROADMAP.md)'s Open Questions for project-wide items (all resolved in later phases).
