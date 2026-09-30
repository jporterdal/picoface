## MODIFIED Requirements

### Requirement: Controlled per-image variation
Dataset Forge SHALL vary every rendered image independently, within ranges set by the configuration:
- a rotation within a configured range either side of upright, drawn from a configured distribution (see "Configurable rotation");
- a slight size change around a nominal figure size;
- a position offset that always keeps the whole figure inside the image;
- the background gray shade and the foreground gray shade, with the foreground always darker than the background by at least a configured minimum contrast, so that images resemble dark drawings on light paper, as in mediaComp's default drawing colors;
- mild random pixel noise.

Figure edges SHALL be anti-aliased. Every class, including `negative_smiley`, SHALL share this polarity, so that no class can be told apart by which of its shades is lighter.

#### Scenario: Rendering produces varied images per class
- **WHEN** Dataset Forge renders many images of the same class
- **THEN** the images SHALL differ from one another in size, position, and shades, SHALL also differ in rotation whenever the configured rotation range is above 0, and SHALL NOT be pixel-identical copies

#### Scenario: Figures are never clipped
- **WHEN** any image is rendered at any rotation, size, and position the configuration allows
- **THEN** the entire figure SHALL lie inside the image

#### Scenario: Foreground polarity and contrast are fixed
- **WHEN** any image is rendered
- **THEN** its foreground shade SHALL be darker than its background shade by at least the configured minimum contrast

#### Scenario: The negative smiley shares the polarity
- **WHEN** a `negative_smiley` is rendered
- **THEN** its disc SHALL be in the darker foreground shade, and its background and cut-out eyes and mouth SHALL be in the lighter background shade

## ADDED Requirements

### Requirement: Configurable rotation
The configuration SHALL set each figure's rotation with two settings:
- a rotation range, in degrees from 0 to 180: every figure's rotation SHALL lie within that many degrees either side of upright;
- a rotation distribution, naming how rotations are drawn within that range:
  - `uniform`: evenly across the whole range;
  - `normal`: centred on upright, with a standard deviation of half the range. Any rotation outside the range SHALL be drawn again, so rotations never pile up at the range's edges.

Upright SHALL mean the same pose for every class: square sides aligned with the image edges, the triangle's and star's top point straight up, and the smiley's eyes above its mouth. The same range and distribution SHALL apply to every class.

When the configuration omits them, the rotation range SHALL default to 180 and the distribution to `uniform`, so that rotations cover the full circle evenly. The default configuration SHALL list both settings explicitly with these values.

#### Scenario: A range of zero renders every figure upright
- **WHEN** the rotation range is 0, with either distribution
- **THEN** every figure SHALL be rendered upright

#### Scenario: Rotations stay within the range
- **WHEN** many images are rendered with a rotation range of R degrees, with either distribution
- **THEN** no figure SHALL be rotated more than R degrees either side of upright

#### Scenario: A uniform distribution spreads rotations evenly
- **WHEN** many images are rendered with a rotation range of R degrees and the `uniform` distribution
- **THEN** their rotations SHALL spread evenly across −R to +R degrees, reaching close to both ends

#### Scenario: A normal distribution clusters rotations around upright
- **WHEN** many images are rendered with a rotation range of R degrees and the `normal` distribution
- **THEN** about 72% of the rotations SHALL lie within R/2 degrees of upright (68.3% of a normal distribution lies within one standard deviation, out of the 95.4% kept within two), and none SHALL lie outside the range

#### Scenario: Omitted rotation settings default to the full circle, evenly
- **WHEN** a configuration omits both rotation settings
- **THEN** its rotation range SHALL be 180 and its distribution `uniform`

#### Scenario: Invalid rotation settings are rejected before rendering
- **WHEN** the configuration's rotation range is below 0 or above 180, or its rotation distribution isn't one of the available ones
- **THEN** Dataset Forge SHALL fail with an error naming the problem, and for an unknown distribution listing the available ones, before rendering any images
