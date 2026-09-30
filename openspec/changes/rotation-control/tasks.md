## 1. Rotation distributions

- [ ] 1.1 Create `dataset_forge/rotation.py` with a `uniform` and a `normal` function. Each takes one uniform draw `u` in [0, 1) and the range in radians, and returns an angle in radians. `normal` returns the inverse CDF of a normal distribution with σ = range / 2, cut off at ±2σ, using `statistics.NormalDist`, and returns exactly 0 when the range is 0. Register both in `ROTATION_DISTRIBUTIONS`. Verify with a quick check in a Python session that `u` = 0, 0.5, and just below 1 give angles within ±range, with 0.5 giving 0.
- [ ] 1.2 Add unit tests in `dataset_forge/tests/test_rotation.py`:
  - both distributions stay within ±range at the ends of `u`;
  - both return 0 at a range of 0;
  - `uniform` is linear in `u`;
  - over an even grid of `u`, about 72% of `normal`'s angles lie within range / 2.

  Verify with `pytest dataset_forge/tests/test_rotation.py`.

## 2. Config settings

- [ ] 2.1 Add `rotation_range: float = 180.0` (degrees) and `rotation_distribution: str = "uniform"` to `ForgeConfig`, with comments in the style of the existing fields, and a `rotation_range_radians` property. Verify that `ForgeConfig.from_dict({"name": "x", "class_names": ["circle"]})` has both defaults.
- [ ] 2.2 In `__post_init__`, reject a `rotation_range` outside [0, 180], and a `rotation_distribution` that isn't in `ROTATION_DISTRIBUTIONS`, with an error listing the available names. Verify by adding cases to `test_invalid_configs_are_rejected_with_a_clear_message`'s parameters and running `pytest dataset_forge/tests/test_config.py`.
- [ ] 2.3 Add both keys, with their default values, to `dataset_forge/configs/default.json`, without touching the user's other uncommitted edits in that file. Verify that `git diff dataset_forge/configs/default.json` shows only the two new keys on top of the existing edits, and that `test_config_round_trips_through_json` passes.

## 3. Rendering

- [ ] 3.1 In `render.py`, replace `rng.uniform(0, 2 * math.pi)` with one `rng.uniform()` draw passed through the configured distribution, keeping the angle at the same place in the draw order. Update the module docstring if its wording no longer matches. Verify that the existing `pytest dataset_forge/tests/test_render.py` still passes, including `test_variation_covers_rotation_size_position_and_shades` at the default range.
- [ ] 3.2 Add render tests for the "Configurable rotation" scenarios:
  - a range of 0 gives every angle 0, with either distribution;
  - over many draws at a range of R = 30, no angle passes ±R, with either distribution;
  - `uniform` reaches close to both ends;
  - about 72% of `normal`'s angles lie within R / 2.

  Verify with `pytest dataset_forge/tests/test_render.py`.
- [ ] 3.3 Add a test that the same seed gives the same shades (`background` and `foreground`) under `uniform` and `normal`, which shows each angle uses exactly one random number. Verify with `pytest dataset_forge/tests/test_render.py`.

## 4. Documentation and check

- [ ] 4.1 In `dataset_forge/README.md`'s Settings section, document `rotation_range` and `rotation_distribution`: their units and defaults, that angle 0 is upright for every class, that normal uses σ = range / 2 and draws again past the edge, and that one range covers each class's symmetry unevenly. Also say that this change renders different images for the same seed than earlier versions did, and that an old export can be recreated from the commit in its manifest. Also update the "Add a class" step so it says draw functions must treat angle 0 as upright. Verify by reading the section back.
- [ ] 4.2 Update the package docstring in `dataset_forge/__init__.py` if it still says "randomized rotation" without qualification. Verify with `grep -rn -i "full circle\|randomized rotation" dataset_forge`.
- [ ] 4.3 Look at the result: render contact sheets from a scratch config with `rotation_range` 30 and each distribution, as well as with the defaults. Confirm the figures look tilted as configured, and upright at a range of 0. Verify by opening each PNG.
- [ ] 4.4 Run the full `pytest dataset_forge/tests` and `pytest`, and run `openspec validate rotation-control --strict`. Verify that everything passes.
