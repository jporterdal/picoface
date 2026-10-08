## Purpose

The instructor-only, offline tool that procedurally renders the course's real training and testing images from an extensible shape taxonomy with controlled variation, exports them reproducibly in the `data-contract` format, and validates each export before it reaches students.

## Requirements

### Requirement: Unrestricted, isolated execution environment
Dataset Forge SHALL be a separate, instructor-only tool with no constraint on hardware, libraries, or run time. It SHALL declare its own dependencies apart from the student-facing package. Dataset Forge MAY depend on the student-facing `picoface` package; `picoface` SHALL NOT depend on Dataset Forge. A library both use, such as numpy or Pillow, MAY appear in both dependency lists. `picoface` SHALL declare only a compatible version range for it, and the Forge's exact version pins SHALL NOT apply to `picoface`.

#### Scenario: Dataset Forge dependencies stay isolated
- **WHEN** the student-facing `picoface` package is installed
- **THEN** Dataset Forge's exact version pins SHALL NOT be required by that installation, and no library the Forge uses that `picoface` does not use itself SHALL be installed by it

#### Scenario: The project's test suite runs without the Forge's dependencies
- **WHEN** the project's tests are run in an environment that has `picoface` and its development dependencies but not Dataset Forge's dependencies
- **THEN** the Forge's tests SHALL be skipped rather than fail, and all other tests SHALL run as before

#### Scenario: The student-facing package does not reach into the Forge
- **WHEN** any module of the installed `picoface` package is imported
- **THEN** no Dataset Forge code SHALL be imported as a result

#### Scenario: The project's test suite runs without the Forge's dependencies
- **WHEN** the project's tests are run in an environment that has `picoface` and its development dependencies but not Dataset Forge's dependencies
- **THEN** the Forge's tests SHALL be skipped rather than fail, and all other tests SHALL run as before

### Requirement: Extensible class taxonomy
Dataset Forge SHALL render each class through its own named renderer, and SHALL take the list of classes to export, in label order, from its configuration. Adding a new class SHALL require only a new renderer and a configuration entry, with no change to how images are varied, shaded, split, exported, or validated.

#### Scenario: Exported classes follow the configuration
- **WHEN** Dataset Forge exports with a configuration listing a set of class names in some order
- **THEN** the export's class names SHALL be exactly those, and label `i` SHALL correspond to the `i`-th listed class

#### Scenario: A subset of classes can be exported
- **WHEN** the configuration lists only some of the available classes
- **THEN** the export SHALL contain only those classes, with labels numbered from 0 in the listed order

#### Scenario: Unknown class names are rejected before rendering
- **WHEN** the configuration names a class that has no renderer
- **THEN** Dataset Forge SHALL fail with an error naming the unknown class and the available ones, before rendering any images

### Requirement: Initial course taxonomy
The default configuration SHALL define seven classes. Each is defined without regard to rotation, and each is drawn in the image's foreground shade on its background shade:
- `square`, `triangle` (equilateral), `star` (five-pointed), and `circle`: filled figures.
- `ring`: an outlined circle.
- `smiley`: an outlined face with two eyes and a mouth, all painted in the foreground shade.
- `negative_smiley`: a filled disc whose two eyes and mouth are cut out to the background shade.

A smiley's eyes and mouth SHALL make it look different from its own rotations. No class SHALL be a rotation of another class.

#### Scenario: Default export has the seven course classes
- **WHEN** Dataset Forge exports with its default configuration
- **THEN** the export's class names SHALL be `square`, `ring`, `circle`, `triangle`, `star`, `smiley`, and `negative_smiley`

#### Scenario: Filled and cut-out round classes differ from their plain counterparts
- **WHEN** a `circle` and a `negative_smiley`, or a `ring` and a `smiley`, are rendered with the same position, size, rotation, and shades
- **THEN** the two images SHALL differ, and the difference SHALL lie within the figure's outline

### Requirement: Configurable resolution and color depth
The exported image height, width, and channel count SHALL come from the configuration. The default configuration SHALL produce 28×28 single-channel (grayscale) images.

#### Scenario: Default images are 28×28 grayscale
- **WHEN** Dataset Forge exports with its default configuration
- **THEN** every exported image SHALL have shape 28×28×1 and dtype `uint8`

#### Scenario: Changing resolution is a configuration change
- **WHEN** the configuration's resolution is changed (e.g. to 24×24 or 32×32)
- **THEN** the export SHALL have the new resolution, with no code change

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

### Requirement: Configurable rotation
The configuration SHALL set each figure's rotation with two settings:
- a rotation range, in degrees from 0 to 180: every figure's rotation SHALL lie within that many degrees either side of upright;
- a rotation distribution, naming how rotations are drawn within that range:
  - `uniform`: evenly across the whole range;
  - `normal`: centred on upright, with a standard deviation of half the range. Any rotation outside the range SHALL be drawn again, so rotations never pile up at the range's edges.

Upright SHALL mean the same pose for every class: square sides aligned with the image edges, the triangle's and star's top point straight up, and the smiley's eyes above its mouth. The same range and distribution SHALL apply to every class.

When the configuration omits them, the rotation range SHALL default to 180 and the distribution to `uniform`, so that rotations cover the full circle evenly. The default configuration SHALL list both settings explicitly.

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

### Requirement: Export conforms to the data contract
Dataset Forge SHALL write each export as one folder holding a training bundle, a testing bundle, and one `classes.json` that both bundles share, in the exact format defined by the `data-contract` capability. Each bundle SHALL load with `load_dataset()` with no conversion step.

#### Scenario: Exported bundles load via the student-facing loader
- **WHEN** either the training or the testing bundle of an export is passed to `load_dataset()`
- **THEN** it SHALL load successfully, with the configured image shape and the configured class names

### Requirement: Train/test split
Each export SHALL include separate training and testing bundles. Each bundle SHALL have the configured number of images for every class. No image SHALL appear in both bundles.

#### Scenario: Both bundles are class-balanced
- **WHEN** Dataset Forge completes an export
- **THEN** the training bundle SHALL contain exactly the configured training count of images for each class, and the testing bundle exactly the configured testing count

#### Scenario: Splits are disjoint
- **WHEN** the training and testing bundles of an export are compared
- **THEN** no image in one SHALL be pixel-identical to any image in the other

### Requirement: Reproducible exports
An export SHALL be fully determined by its configuration and seed. Each export folder SHALL include a manifest recording the complete configuration, the seed, and the versions of the tools used to render it. Generated datasets SHALL NOT be committed to the repository; only Dataset Forge and its configuration are tracked.

#### Scenario: Same configuration and seed give identical data
- **WHEN** Dataset Forge exports twice with the same configuration and seed, in the same pinned environment
- **THEN** the two exports' images, labels, and class names SHALL be identical

#### Scenario: Different seeds give different data
- **WHEN** Dataset Forge exports twice with the same configuration and different seeds
- **THEN** the two exports' images SHALL differ

#### Scenario: The manifest is enough to reproduce an export
- **WHEN** an export's manifest is read
- **THEN** it SHALL state the configuration, seed, and tool versions needed to produce that export again

#### Scenario: Exports stay out of version control
- **WHEN** Dataset Forge writes an export to its default output location
- **THEN** that location SHALL be ignored by version control

### Requirement: Export validation
Dataset Forge SHALL validate every export after writing it, and SHALL report the export as failed, naming the check, if any of these fail:
- each bundle loads with `load_dataset()` and matches the configured shape and classes;
- each bundle is class-balanced;
- the two bundles are disjoint;
- a classifier using only each image's mean brightness, fit on the training bundle and scored on the testing bundle, scores less than chance (1 / number of classes) plus 0.1.

Dataset Forge SHALL also measure and report, without failing the export, the accuracy of a classifier using only each image's estimated ink fraction (the fraction of pixels belonging to the figure). This is a baseline that shape-aware models are expected to beat, not a gate.

#### Scenario: A valid export passes
- **WHEN** Dataset Forge exports with its default configuration
- **THEN** every validation check SHALL pass, and the mean-brightness and ink-fraction baseline accuracies SHALL be reported

#### Scenario: A brightness-separable export is rejected
- **WHEN** a configuration makes the classes separable by mean brightness (for example, fixed shades with very different figure areas per class)
- **THEN** validation SHALL report the export as failed, naming the mean-brightness check

#### Scenario: Ink fraction is reported, not gated
- **WHEN** the ink-fraction-only classifier scores well above chance on an export that passes every other check
- **THEN** the export SHALL still pass, and the ink-fraction accuracy SHALL appear in the validation report
