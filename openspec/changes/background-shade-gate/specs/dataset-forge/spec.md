## MODIFIED Requirements

### Requirement: Export validation
Dataset Forge SHALL validate every export after writing it, and SHALL report the export as failed, naming the check, if any of these fail:
- each bundle loads with `load_dataset()` and matches the configured shape and classes;
- each bundle is class-balanced;
- the two bundles are disjoint;
- a classifier using only each image's background shade, fit on the training bundle and scored on the testing bundle, scores less than chance (1 / number of classes) plus 0.1. An image's background shade is the mean of its outermost rows and columns.

Dataset Forge SHALL also measure and report, without failing the export, the accuracy of two single-statistic baselines: a classifier using only each image's mean brightness, and one using only each image's estimated ink fraction (the fraction of pixels belonging to the figure). Real shapes differ in area, so both can score above chance. They are baselines that shape-aware models are expected to beat, not gates.

#### Scenario: A valid export passes
- **WHEN** Dataset Forge exports with its default configuration
- **THEN** every validation check SHALL pass, and the background-shade, mean-brightness, and ink-fraction accuracies SHALL be reported

#### Scenario: A brightness-separable export is rejected
- **WHEN** an export's classes are separable by brightness because its background shade differs by class (for example, every image of one class has a lighter background than every image of another)
- **THEN** validation SHALL report the export as failed, naming the background check

#### Scenario: Brightness that follows figure area is reported, not gated
- **WHEN** an export's background and foreground shades are drawn independently of class, but mean brightness separates the classes because their figures differ in area (for example, fixed shades with very different figure areas per class)
- **THEN** the export SHALL still pass, and its mean-brightness accuracy SHALL appear in the validation report

#### Scenario: Ink fraction is reported, not gated
- **WHEN** the ink-fraction-only classifier scores well above chance on an export that passes every other check
- **THEN** the export SHALL still pass, and the ink-fraction accuracy SHALL appear in the validation report
