## MODIFIED Requirements

### Requirement: CPU training time budget
Training a VAE built with `build_vae()` via `train()` SHALL complete in no more than a few minutes on a CPU-only machine with no GPU, using either the built-in stub dataset or a real Dataset Forge export. This budget covers the full combined objective — reconstruction, KL-divergence, and classification trained together in one call — not reconstruction alone. This budget applies only to `build_vae()` models; `build_autoencoder()` has no enforced budget for MVP (Requirement: build_autoencoder() is a nice-to-have).

#### Scenario: VAE training completes within budget on CPU
- **WHEN** `train()` is run on a CPU-only machine with a `build_vae()` model using the built-in stub dataset with default parameters
- **THEN** training SHALL complete in under 5 minutes

#### Scenario: VAE training completes within budget on CPU with a real dataset
- **WHEN** `train()` is run on a CPU-only machine with a `build_vae()` model using a Dataset Forge export with default parameters
- **THEN** training SHALL complete in under 5 minutes

#### Scenario: Wall-clock duration is recorded for comparison
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the returned history SHALL record the total wall-clock duration, so that a regression against this budget is visible without separate instrumentation
