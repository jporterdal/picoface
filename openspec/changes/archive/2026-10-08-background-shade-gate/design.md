## Context

See proposal.md, "Why". `validate_export()` in `dataset_forge/validate.py` computes two scalar features per image, `mean_brightness()` and `ink_fraction()`. It scores each with `nearest_class_mean_accuracy()` (fit on train, scored on test) and gates mean brightness at `chance + BRIGHTNESS_MARGIN` (0.1). Rendering already keeps every figure's circumscribed circle at least `ForgeConfig.clearance` (`EDGE_MARGIN` + one supersampled pixel, 1.25 px by default) from each edge, so the outermost rows and columns are pure `background + noise`. Over 3,500 default renders, no border pixel had any ink coverage.

Measured on the default config (seed 0, 2,000 train / 800 test images per class):

| Feature | Accuracy |
|---|---|
| chance | 0.143 |
| border mean | 0.149 |
| mean brightness | 0.244 |
| ink fraction | 0.410 |

At the test suite's reduced counts (120 / 60 per class) mean brightness happens to land under the limit, so the existing tests pass. The failure shows only in a full-size export.

## Goals / Non-Goals

**Goals:**
- A gate that measures class information in the shades, independent of figure area and of the configured background range.
- No change to rendering, the config, or the export format. The default config passes as it stands.

**Non-Goals:**
- A gate on the foreground (ink) shade. See Risks.
- Changing the margin, or making it configurable.
- picoface's stub-dataset brightness test and its data-contract requirement.

## Decisions

### 1. The background shade is the mean of the 1-pixel border

A new feature function, `background_shade(images)`, averages each image's outermost rows and columns (108 pixels at 28×28). The border is ink-free by construction, so this is the image's sampled background plus averaged noise. With σ = 6 noise, its standard deviation around the true background is about 0.6 gray levels. Near 255, clipped noise pulls it down by up to about 2 levels more. Neither depends on the class.

**Alternatives considered:**
- *A high percentile, such as the 95th pixel value.* This only measures background by assumption: a figure covering most of the image pulls it down, which would leak area back in. It scored about the same here (0.151), but it isn't clean by construction.
- *A 2-pixel border.* Clearance is 1.25 px, so the second ring can get anti-aliased ink. Only one ring is guaranteed clean.

### 2. Same classifier, same margin, renamed check

The gate keeps `nearest_class_mean_accuracy()` and the 0.1 margin. Only the feature changes, so every threshold stays comparable with today's. `BRIGHTNESS_MARGIN` becomes `SHADE_MARGIN`. Its comment still points to the stub test's matching margin. The check's key changes from `"mean_brightness"` to `"background"`, and `ValidationReport` gains `background_accuracy`.

### 3. Mean brightness stays as a reported baseline

`mean_brightness_accuracy` keeps its name and its key in `to_dict()`, so manifests stay readable side by side. In `__str__` its label changes from "(gated)" to "(baseline)", and a "background-only (gated)" line comes first. The module docstring's account of what's gated and what's reported changes to match.

### 4. Leak fixtures edit an export after it's written

No config setting can make the background depend on the class, because shades are drawn independently of class by design. The failing-export tests therefore export normally, then load one class's images in both splits, add a fixed offset (clipped to 0–255), and write them back. This mimics the real bug: shades drawn differently per class. `test_validate.py` already has `_load()` and `_rewrite()` helpers for this. They move into a shared test module so `test_cli.py` can use them too.

The old fixture (fixed shades, no noise, circle vs ring) stays as the test for the "Brightness that follows figure area is reported, not gated" scenario: it must now pass, with a high mean-brightness accuracy. Its border means are all identical, so every class centroid is the same, the classifier always predicts class 0, and it scores exactly chance.

## Risks / Trade-offs

- [A foreground shade that depends on the class goes uncaught] → Measuring ink shade on its own is contaminated by area: thin or small figures have few fully covered pixels, so the 5th-percentile pixel scored 0.175 here. Rendering draws both shades in the same `sample_params()` call, so a class-dependent bug would most likely affect the background too. Left for a later change if it's ever needed.
- [The gate now relies on figures never reaching the border] → `ForgeConfig` already enforces this through `clearance`. The check fails if a config or a new class breaks it: border ink is area information, so it would push the background accuracy up.
- [Small exports make the background accuracy noisy] → It's at chance, so a 0.1 margin is several standard deviations even at the test suite's counts. Tiny two-class fixtures are only used where the expected result is a clear failure or exactly chance.
- [Anything that reads the `mean_brightness` check by name breaks] → Nothing in this repo does. Old manifests keep their old record and aren't revalidated automatically.
