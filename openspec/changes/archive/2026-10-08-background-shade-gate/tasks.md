## 1. Validation

- [x] 1.1 In `dataset_forge/validate.py`, add `background_shade(images)`, the mean of each image's outermost rows and columns, next to `mean_brightness()`. Its docstring says why the border is ink-free (`ForgeConfig.clearance`). Verify in a Python session that on 50 default renders it matches each image's sampled `background` to within noise (about 1.5 gray levels; a little lower near 255, where noise is clipped), using `sample_params()` and `shade()` as `test_validate.py` already does.
- [x] 1.2 Rename `BRIGHTNESS_MARGIN` to `SHADE_MARGIN`, keeping its comment's pointer to the stub test. Gate the new check `"background"` on `background_accuracy < chance + SHADE_MARGIN`, with a detail message worded like today's. Add `background_accuracy` to `ValidationReport`, `to_dict()`, and `__str__`. Keep `mean_brightness_accuracy` computed and reported, relabelled "(baseline)". Update the module docstring's account of what's gated and what's reported. Verify by running `python -m dataset_forge.validate` on a scratch copy of `dataset_forge/output/default-seed0` (it rewrites the manifest, so leave the original alone): it shows `[ok] background` and all three accuracies.

## 2. Tests

- [x] 2.1 Move `_load()` and `_rewrite()` from `dataset_forge/tests/test_validate.py` into a shared test helper module, and add a helper that offsets every image of one class in both splits by a fixed amount, clipped to 0–255. Verify `pytest dataset_forge/tests/test_validate.py` still passes before changing any assertions.
- [x] 2.2 In `test_validate.py`:
  - the course-export test asserts `background_accuracy < chance + 0.1` and keeps the ink-fraction assertion;
  - the manifest test expects the check names `{"loads", "balanced", "disjoint", "background"}` and a recorded `background_accuracy`;
  - a new test makes the background depend on the class with the offset helper and expects `failed_checks == ["background"]`;
  - the old fixed-shade circle-vs-ring test is renamed for the "Brightness that follows figure area is reported, not gated" scenario: it expects the export to pass, with `mean_brightness_accuracy` well above chance.

  Verify with `pytest dataset_forge/tests/test_validate.py`.
- [x] 2.3 In `dataset_forge/tests/test_cli.py`, the failed-check test uses a class-dependent background fixture and asserts `[FAIL] background` with exit code 1. Update the comment above the export test, which mentions the mean-brightness check. Verify with `pytest dataset_forge/tests/test_cli.py`.

## 3. Documentation

- [x] 3.1 In `dataset_forge/README.md`'s "Validation" section, replace the mean-brightness gate with the background-shade gate, and list mean brightness with ink fraction under "Also reported". Say briefly why: with a narrow background range, mean brightness mostly tracks figure area. Verify with `grep -n -i "brightness" dataset_forge/README.md` that nothing still calls mean brightness gated.

## 4. Full check

- [x] 4.1 Fix the two tests that hard-code the old default config, which fail on the committed `configs/default.json` (`rotation_range` 30, 8000 / 800 per class) independently of this change: `test_variation_covers_rotation_size_position_and_shades` in `dataset_forge/tests/test_render.py` draws from a full-circle (`rotation_range` 180) copy of the default config and expects a background spread relative to `background_range` rather than over 100 gray levels, and `test_real_dataset_fixture_loads_expected_shape_and_classes` in `tests/test_real_dataset.py` takes its expected image counts from the default config. Verify with `pytest dataset_forge/tests/test_render.py tests/test_real_dataset.py::test_real_dataset_fixture_loads_expected_shape_and_classes`.
- [x] 4.2 Export the default config at full size into a scratch folder (`python -m dataset_forge --out <scratch>`) and confirm "Validation PASSED", with background accuracy near chance and mean brightness around 0.24. Then run `pytest dataset_forge/tests`, `pytest`, and `openspec validate background-shade-gate --strict`. Verify everything passes.
