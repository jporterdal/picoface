## Why

Export validation fails the default config: mean-brightness-only accuracy is 0.246, against a limit of chance + 0.1 = 0.243. The cause is the intentional narrower `background_range` (200–255, was 96–255). With less background variation, mean brightness mostly tracks how much of the image the figure covers, and real shapes differ in area. The shades themselves carry no class information: a classifier on the background shade alone scores 0.149, against chance at 0.143. So the gate is failing a config the instructor accepts, for a reason the export's ink-fraction baseline already reports and accepts. What the gate should catch is a shade that depends on the class, and that's what it should measure.

## What Changes

- **The gated statistic becomes the background shade.** Each image's background shade is measured as the mean of its outermost rows and columns, which never contain ink. A classifier using only that shade must score below chance + 0.1, the same margin as today.
- **Mean brightness is reported, not gated.** It joins ink fraction as a baseline. With a narrow background range it mostly reflects figure area, so it says little on its own.
- **A failure names the new check.** The validation report, the CLI output, and the manifest's `validation` record call the check `background` instead of `mean_brightness`, and record the background-only accuracy. **BREAKING** only for anything that reads a check by name from a manifest or the CLI output. Nothing in this repo does.
- **The default config passes again**, with its current background range and image counts unchanged.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `dataset-forge`: the "Export validation" requirement changes. Its fourth gated check moves from mean brightness to the background shade. Mean brightness moves to the reported-but-not-gated baselines, next to ink fraction. The scenario "A brightness-separable export is rejected" now covers only brightness that comes from a class-dependent background. A new scenario says that brightness which follows figure area passes. The "A valid export passes" scenario reports all three accuracies.

## Impact

- `dataset_forge/validate.py`: a background-shade feature, the renamed gated check and its report field, and mean brightness moved to the reported baselines.
- Tests: `dataset_forge/tests/test_validate.py` and `dataset_forge/tests/test_cli.py`. Their failing-export fixtures leak only through area, so they no longer fail and need a class-dependent background instead.
- Two tests that hard-code the old default config and fail on the current one, unrelated to the gate: `dataset_forge/tests/test_render.py` (full-circle rotation, background spread) and `tests/test_real_dataset.py` (image counts). Fixed here so the full suite passes.
- `dataset_forge/README.md`: the "Validation" section's gated and reported lists.
- Unchanged: rendering, the config and `configs/default.json`, the export format, and picoface's stub-dataset brightness test (`tests/test_stub_data.py`) with its data-contract requirement, which are about the stub generator, not Forge exports.
