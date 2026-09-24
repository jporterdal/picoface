## MODIFIED Requirements

### Requirement: Unrestricted, isolated execution environment
Dataset Forge SHALL be a separate, instructor-only tool with no constraint on hardware, libraries, or run time. It SHALL declare its own dependencies apart from the student-facing package. Dataset Forge MAY depend on the student-facing `picoface` package; `picoface` SHALL NOT depend on Dataset Forge. A library both use, such as numpy or Pillow, MAY appear in both dependency lists. `picoface` SHALL declare only a compatible version range for it, and the Forge's exact version pins SHALL NOT apply to `picoface`.

#### Scenario: Dataset Forge dependencies stay isolated
- **WHEN** the student-facing `picoface` package is installed
- **THEN** Dataset Forge's exact version pins SHALL NOT be required by that installation, and no library the Forge uses that `picoface` does not use itself SHALL be installed by it

#### Scenario: The student-facing package does not reach into the Forge
- **WHEN** any module of the installed `picoface` package is imported
- **THEN** no Dataset Forge code SHALL be imported as a result

#### Scenario: The project's test suite runs without the Forge's dependencies
- **WHEN** the project's tests are run in an environment that has `picoface` and its development dependencies but not Dataset Forge's dependencies
- **THEN** the Forge's tests SHALL be skipped rather than fail, and all other tests SHALL run as before

### Requirement: Controlled per-image variation
Dataset Forge SHALL vary every rendered image independently, within ranges set by the configuration:
- rotation over the full circle;
- a slight size change around a nominal figure size;
- a position offset that always keeps the whole figure inside the image;
- the background gray shade and the foreground gray shade, with the foreground always darker than the background by at least a configured minimum contrast, so that images resemble dark drawings on light paper, as in mediaComp's default drawing colors;
- mild random pixel noise.

Figure edges SHALL be anti-aliased. Every class, including `negative_smiley`, SHALL share this polarity, so that no class can be told apart by which of its shades is lighter.

#### Scenario: Rendering produces varied images per class
- **WHEN** Dataset Forge renders many images of the same class
- **THEN** the images SHALL differ from one another in rotation, size, position, and shades, and SHALL NOT be pixel-identical copies

#### Scenario: Figures are never clipped
- **WHEN** any image is rendered at any rotation, size, and position the configuration allows
- **THEN** the entire figure SHALL lie inside the image

#### Scenario: Foreground polarity and contrast are fixed
- **WHEN** any image is rendered
- **THEN** its foreground shade SHALL be darker than its background shade by at least the configured minimum contrast

#### Scenario: The negative smiley shares the polarity
- **WHEN** a `negative_smiley` is rendered
- **THEN** its disc SHALL be in the darker foreground shade, and its background and cut-out eyes and mouth SHALL be in the lighter background shade
