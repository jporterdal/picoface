## Why

Dataset Forge rotates every figure by a uniformly random angle over the full circle, and no setting changes that. The angle is hard-coded in `sample_params()`. An instructor who wants upright figures, figures tilted only slightly, or tilts that cluster around upright has to edit code. Rotation is also a known source of difficulty for the VAE: the `star` and `smiley` classes decode poorly because of their rotation-driven variation within each class. So how much rotation the data has is worth being able to vary and measure.

## What Changes

- **Two new config settings control rotation:**
  - `rotation_range`: in degrees, from 0 to 180. Each figure is rotated within ±`rotation_range` of upright (0°). At 0, every figure is upright. At 180, rotation covers the full circle.
  - `rotation_distribution`: `"uniform"` or `"normal"`.
    - `"uniform"` draws angles evenly across the range.
    - `"normal"` draws them around upright, with a standard deviation of half the range. Any angle outside the range is drawn again, so no figures pile up at the range's edges.
- **Invalid settings fail before rendering.** A `rotation_range` outside 0–180 is rejected. An unknown distribution name fails with an error listing the available ones, the same way unknown class names fail today.
- **The defaults keep today's design:** `rotation_range` 180, `rotation_distribution` `"uniform"`. Configs that omit the settings get the defaults, and `configs/default.json` lists both explicitly.
- **BREAKING (reproducibility): the same config and seed now renders different images.** Angles are now drawn from −180° to +180° instead of 0° to 360°. The distribution is unchanged, but each image's angle moves by 180°. Earlier exports stay valid data. To recreate one exactly, check out the git commit recorded in its manifest.
- **One range applies to every class.** A class's symmetry isn't taken into account. A square repeats every 90° and a smiley every 360°, so the same range covers a larger share of a square's distinct orientations than of a smiley's. This limit is documented, not addressed.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `dataset-forge`: the "Controlled per-image variation" requirement changes. Rotation is no longer always over the full circle: it falls within a configured range around upright, following a configured distribution. The scenario that renders varied images of one class expects different rotations only when the range is above 0. A new "Configurable rotation" requirement defines the two settings, their defaults, what "upright" means, and how invalid values are rejected.

## Impact

- `dataset_forge/config.py`: the two new fields and their validation.
- `dataset_forge/rotation.py` (new): the two distributions, each turning one uniform draw into an angle.
- `dataset_forge/render.py`: `sample_params()` draws the angle from the configured range and distribution.
- `dataset_forge/configs/default.json`: lists both settings with their default values.
- `dataset_forge/README.md`: documents both settings, the upright convention, the class-symmetry limit, and the reproducibility break.
- Tests: `dataset_forge/tests/test_config.py` (defaults, validation) and `dataset_forge/tests/test_render.py` (range, both distributions, a range of 0).
- Unchanged: the data-contract format, the draw functions in `shapes.py`, the random stream for each split and class, and picoface itself. No new dependencies: the normal distribution uses Python's standard `statistics.NormalDist`.
