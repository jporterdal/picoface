## MODIFIED Requirements

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
