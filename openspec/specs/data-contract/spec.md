## Purpose

The shared image-dataset interchange format that every downstream arm
(classifier, generator, linkage) loads data through: an `.npz` + `classes.json`
on-disk schema, the `load_dataset()` entry point that hides all file-format
and array-handling details, a single immutable dataset return type shared by
all consumers, and an internal synthetic stub-dataset generator that lets
downstream phases build and test plumbing before the real course dataset
(Phase 5) exists.

## Requirements

### Requirement: Dataset interchange format
The system SHALL define a dataset interchange format consisting of an `.npz` file containing an `images` array (uint8, shape N×H×W×C) and a `labels` array (integer, shape N), accompanied by a `classes.json` file mapping integer label indices to human-readable class names.

#### Scenario: Loading a well-formed dataset bundle
- **WHEN** a dataset bundle conforming to the schema is loaded
- **THEN** `load_dataset()` SHALL return images, labels, and class names without requiring the caller to specify image dimensions or class count in advance

### Requirement: load_dataset() entry point
The system SHALL provide a `load_dataset()` function as the sole public entry point for reading a dataset bundle, hiding all file-format and array-handling details behind that function.

#### Scenario: Student loads a dataset with one function call
- **WHEN** a student calls `load_dataset(path)` with a valid bundle path
- **THEN** they SHALL receive a ready-to-use dataset object without writing any file I/O or parsing code themselves

### Requirement: Format is size- and class-agnostic
The dataset interchange format SHALL NOT hardcode image resolution, color depth, or number of classes; these SHALL be inferred from the bundle's contents at load time.

#### Scenario: Two datasets with different shapes both load correctly
- **WHEN** two dataset bundles with different image resolutions or class counts are each loaded with `load_dataset()`
- **THEN** both SHALL load successfully without any code changes to `load_dataset()` or to the calling code

### Requirement: Shared dataset return type
The system SHALL define a single, immutable dataset value type (numpy-only fields: `images`, `labels`, `class_names`; no framework-specific types such as `torch` tensors) that `load_dataset()` returns, so that all consumers of dataset data share one contract regardless of framework choices made in later phases.

#### Scenario: Inspecting a loaded dataset does not dump raw pixel data
- **WHEN** a loaded dataset value is printed or repr'd (e.g. at a REPL or in a notebook)
- **THEN** the output SHALL show array shapes and dtypes (e.g. image count, dimensions, class list) rather than raw array contents

### Requirement: Internal synthetic stub dataset
The system SHALL provide a non-public, internal synthetic stub-dataset generator (small arbitrary image dimensions, at least two fake classes) that returns the same dataset value type as `load_dataset()`, usable by this project's own development and test code to validate classifier, generator, and linkage plumbing before a real course dataset exists. This generator SHALL NOT be part of the package's public/documented API.

The generator SHALL be able to produce classes that are distinguishable only by the spatial arrangement of pixels within an image, at comparable mean brightness across classes — not solely by per-image brightness or another single global statistic. Classes separable by a global statistic alone can be solved through a one-dimensional bottleneck, so a model's accuracy on them carries no information about its ability to distinguish shapes, which is what the real dataset (Phase 5) will require.

The generator SHALL retain a mode producing trivially-separable classes, for plumbing and shape-agnosticism tests that do not need difficulty.

This requirement does not commit the project to any particular shape taxonomy, resolution, or color depth — those remain Phase 5 decisions. It constrains only the *kind* of discriminative difficulty the stub is able to present.

#### Scenario: Validating plumbing without real content
- **WHEN** downstream test code exercises classifier, generator, or linkage plumbing against the stub dataset generator's output
- **THEN** it SHALL be able to treat that output identically to a value returned by `load_dataset()`, using the same interface that will later serve the real dataset

#### Scenario: Stub dataset generator proves the real file-format parser
- **WHEN** stub-generated data is written to a `.npz` file and companion `classes.json`, then read back with `load_dataset()`
- **THEN** the values read back SHALL match the originally generated data, demonstrating that `load_dataset()` correctly parses the on-disk interchange format

#### Scenario: Spatially-distinguished classes resist a brightness-only solution
- **WHEN** the stub generator is asked for spatially-distinguished classes
- **THEN** classifying those images by per-image mean brightness alone SHALL perform no better than chance, while their class identities remain recoverable from spatial arrangement

#### Scenario: Trivially-separable classes remain available
- **WHEN** test code requires a stub dataset whose classes are separable without spatial reasoning
- **THEN** the generator SHALL still provide one, so plumbing tests are not made slower or more fragile by unnecessary difficulty
