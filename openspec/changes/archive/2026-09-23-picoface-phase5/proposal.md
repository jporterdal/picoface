## Why

Every arm of `picoface` has so far been proven only against the internal stub dataset, a disposable plumbing aid with placeholder figures. The course still has no real training data, and the content decisions the project has deferred since Phase 0 (shape taxonomy, resolution, color depth, noise and variation) are still open. Phases 2–4 are done, so nothing downstream is waiting on plumbing anymore. What Phase 6 (integration and tuning) needs is a real dataset to swap in. Phase 5 builds Dataset Forge, the instructor-only tool that renders it, and makes the content decisions that tool encodes.

## What Changes

- Implement Dataset Forge in `dataset_forge/`, outside the installable package. It is a config-driven, seeded tool that procedurally renders grayscale shape images with Pillow and exports class-balanced train and test splits in the existing `data-contract` format (`train.npz`, `test.npz`, and one shared `classes.json`), plus a manifest recording how the export was made.
- Decide the initial course taxonomy: seven classes, each defined without regard to rotation.
  - `square`, `triangle`, `star`, `circle`: filled.
  - `ring`: an outlined circle.
  - `smiley`: a face outline with eyes and mouth, painted in the foreground shade on the background.
  - `negative_smiley`: a filled disc with eyes and mouth cut out to the background.

  Classes live in a registry keyed by name, so adding one later is one draw function plus one config entry.
- Decide the initial format: 28×28 pixels (MNIST's size), one grayscale channel. Resolution is a config value, so changing it is a config change, not a code change. The plan started at 24×24. The smoke check (below) showed the time headroom and a clear gain in the student's model's accuracy at 28×28, so the default was moved there before this change was finished (`diagnostics.md`).
- Decide the variation and noise policy:
  - full 0–360° rotation;
  - slight size jitter;
  - position jitter that keeps every figure fully in frame;
  - randomized background and foreground gray shades, with the foreground always lighter and a minimum contrast between them;
  - mild Gaussian pixel noise;
  - anti-aliasing by supersampling.
- Validate every export before it is accepted:
  - class balance;
  - no image appearing in both splits;
  - a round trip through `picoface.load_dataset()`;
  - a gate that a mean-brightness-only classifier scores near chance, carried over from the stub's test.

  The accuracy of an ink-amount-only classifier is recorded as a baseline. It is not a gate: real shapes differ in area, and that signal is accepted rather than engineered away.
- Run a one-time feasibility smoke check. Train the existing CNN and VAE at default settings on a default export, and record wall-clock time, held-out accuracy, per-class confusion, and `classify_generated()` agreement in `diagnostics.md`. This informs Phase 6 and informed the 24→28 decision. It is not tuning.
- Generated datasets are never committed. Only the Forge and its default config are tracked; exports go to the already-gitignored `dataset_forge/output/`.

## Capabilities

### New Capabilities
- `dataset-forge`: the instructor-only, unrestricted, offline tool that renders the real training and testing images from an extensible shape taxonomy with controlled variation, exports them reproducibly in the `data-contract` format, and validates each export.

### Modified Capabilities
None. The export fits the existing `data-contract` as-is: `load_dataset()` already reads `classes.json` from the `.npz` file's folder, so the two splits can share one. `packaging` already requires the Forge to stay out of the installable package. The student-facing package, the stub dataset, and every existing test are untouched. Swapping the real dataset in for the stub is Phase 6.

## Impact

- `dataset_forge/`: new Python package (renderer, shape registry, config, export, validation, command-line entry point, feasibility smoke-check script), a checked-in default config, and its own tests under `dataset_forge/tests/`.
- `dataset_forge/requirements.txt`: pins Pillow and numpy for byte-for-byte reproducible exports. The Forge also imports `picoface`, the installed dev package, for the `load_dataset()` round trip and the smoke check. The dependency runs one way only: `picoface` never imports the Forge, and its declared dependencies do not change.
- `dataset_forge/README.md`: how to install the Forge's requirements, run an export, and run its tests.
- Test configuration: the Forge's tests skip cleanly when Pillow is not installed, so a plain `pytest` in a student-package-only environment still passes.
- No change to `src/picoface/` or to any existing spec.
