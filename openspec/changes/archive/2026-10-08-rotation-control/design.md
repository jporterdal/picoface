## Context

See proposal.md, "Why". Today `sample_params()` in `dataset_forge/render.py` draws `angle = rng.uniform(0, 2 * math.pi)`, and `Placement.angle` (radians, clockwise on screen) rotates every point of the figure. Every draw function in `shapes.py` already treats angle 0 as upright. The random stream for each split and class comes from `export._class_rng()`, and `sample_params()` draws from it in a fixed order: radius, stroke, cx, cy, angle, background, foreground. A figure's circumscribed circle doesn't change as it rotates, so placement and clipping don't depend on the angle.

## Goals / Non-Goals

**Goals:**
- The two settings from the "Configurable rotation" requirement, validated when the config is built, so a bad value fails before any rendering.
- Exactly one random number used per angle, whatever the distribution, so the draws after it (the shades) line up with the same stream positions for either distribution.
- No new dependencies.

**Non-Goals:**
- Per-class or per-split rotation settings, or ranges that account for a class's symmetry.
- Centres other than upright, or ranges that aren't symmetric around it.
- Keeping old seeds byte-identical (see Decision 4).

## Decisions

### 1. Settings are degrees in the config, radians in the code

`ForgeConfig` gains `rotation_range: float = 180.0` and `rotation_distribution: str = "uniform"`, with comments in the style of the existing fields. A property, `rotation_range_radians`, does the one conversion, the same way `nominal_radius` derives pixels from `radius_fraction`. Degrees are what a person editing JSON thinks in. Radians stay inside the render code, as they are today.

### 2. Validation lives in `ForgeConfig.__post_init__`

Next to the existing checks:
- `rotation_range` must satisfy `0 <= rotation_range <= 180`. Anything past 180 would rotate some figures more than half a turn, which is the same as a smaller rotation the other way, so it's refused rather than quietly folded back.
- `rotation_distribution` must be a key of the `ROTATION_DISTRIBUTIONS` mapping (Decision 3). The error lists the available names, worded like `check_class_names()`.

Loading a config with `from_dict()` already builds a `ForgeConfig`, so a bad value fails before anything renders. That covers the CLI, the contact sheet, and export.

### 3. Each distribution maps one uniform draw to an angle

`sample_params()` draws `u = rng.uniform()` once and passes it to the configured distribution's function, which returns an angle in radians:
- `uniform`: `(2u − 1) · range`.
- `normal`: an inverse-CDF draw from a normal distribution cut off at ±range. With σ = range / 2, the cutoff is at ±2σ, so `u` is rescaled to lie between Φ(−2) and Φ(2), and then `statistics.NormalDist(0, σ).inv_cdf()` turns it into an angle. This gives exactly the distribution you'd get by drawing from the normal distribution and drawing again whenever the result falls outside the range (the spec's "drawn again"), but it uses exactly one random number.
- With a range of 0, both distributions return exactly 0. The `normal` function returns 0 early, because `NormalDist` rejects σ = 0.

`ROTATION_DISTRIBUTIONS` maps names to these functions. It lives in a new module, `dataset_forge/rotation.py`, which both `config.py` (for the names to validate against) and `render.py` (to draw) import. It can't live in `render.py`, because `render.py` already imports `ForgeConfig`, and `config.py` importing back from it would be circular. Adding a distribution later means one function and one entry.

**Alternatives considered:**
- *Literal redrawing with `rng.normal()` in a loop.* Each image would use a different amount of the stream, so an image's shades would depend on how many redraws its angle needed. Rejected because the stream would stop lining up.
- *Clamping to ±range.* It piles images up at exactly ±range. The user ruled it out.
- *`scipy.stats.truncnorm`.* A new dependency for one line that the standard library already covers.

### 4. Draw from −range to +range and accept the 180° shift

At the defaults, the new formula is `(2u − 1)·π = 2πu − π`. The old one was `rng.uniform(0, 2π) = 2πu`. Both give the same distribution, but each image's angle moves by exactly 180°. Keeping old seeds byte-identical would take a special case written just for the default settings. The user accepted the break: exports aren't committed, and manifests record the git commit. The config's new fields appear in every manifest automatically, through `to_dict()`.

## Risks / Trade-offs

- [One range for every class treats symmetric shapes unevenly. ±30° covers 60 of a square's 90 distinct degrees but only 60 of a smiley's 360] → The README says so. Per-class ranges are a possible later change.
- [Floating-point edges: `inv_cdf` can't take exactly 0 or 1, and `u` could land on Φ(±2) after rescaling] → Φ(−2) ≈ 0.023 and Φ(2) ≈ 0.977, so `inv_cdf` never gets 0 or 1. A test checks that angles never pass ±range at `u` = 0 and at `u` just below 1.
- [The existing test `test_variation_covers_rotation_size_position_and_shades` expects a spread of angles over 1.9π] → It still holds at the default range. The new tests cover the other settings.
- [Existing exports and figures on the user's machine no longer match a fresh render of the same seed] → Regenerate them as needed. The proposal marks this BREAKING.

## Migration Plan

Old configs without the new keys load with the defaults, which keep full-circle, even rotation. No other migration is needed. `dataset_forge/configs/default.json` gains both keys. Its other uncommitted edits (8000/800 images per class, background range 200–255) belong to the user and stay as they are.
