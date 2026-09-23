## MODIFIED Requirements

### Requirement: CPU training time budget
Training a classifier built with `build_classifier()` via `train()` with default parameters SHALL complete in no more than a few minutes on a CPU-only machine with no GPU, using either the built-in stub dataset or a real Dataset Forge export.

#### Scenario: Training completes within budget on CPU
- **WHEN** `train()` is run on a CPU-only machine using the built-in stub dataset with default parameters
- **THEN** training SHALL complete in under 5 minutes

#### Scenario: Training completes within budget on CPU with a real dataset
- **WHEN** `train()` is run on a CPU-only machine using a Dataset Forge export with default parameters
- **THEN** training SHALL complete in under 5 minutes
