## ADDED Requirements

### Requirement: build_autoencoder()
The system SHALL provide a `build_autoencoder()` function constructing a basic (non-variational) encoder/decoder model from a `Dataset`, as a pedagogical stepping stone toward the VAE. It SHALL accept the same call shape as `build_vae()` (Requirement: build_vae()), so the two are interchangeable at the call site.

#### Scenario: Student builds and trains a plain autoencoder
- **WHEN** a student calls `build_autoencoder(data)` then `train(model, data)`
- **THEN** the model SHALL learn to reconstruct input images without the student implementing encoder/decoder layers

### Requirement: build_vae()
The system SHALL provide a `build_vae()` function constructing a variational autoencoder (probabilistic latent space, reparameterization, combined reconstruction + KL-divergence loss) from a `Dataset`, exposed as a comparable API surface to `build_autoencoder()`.

#### Scenario: Student upgrades from autoencoder to VAE
- **WHEN** a student replaces `build_autoencoder(data)` with `build_vae(data)` in their assembly code and re-runs `train()`
- **THEN** training SHALL succeed using the same `train()` entry point, without the student writing the reparameterization trick or KL-divergence loss themselves

### Requirement: Fixed 2-dimensional latent space
Models built by `build_vae()` SHALL use a fixed, non-configurable 2-dimensional latent space. No student-facing parameter SHALL exist to change this for MVP.

#### Scenario: Latent space is directly plottable
- **WHEN** a student calls `show_latent_space()` (Requirement: show_latent_space()) on a model built by `build_vae()`
- **THEN** the encoded points SHALL be plotted directly as (x, y) coordinates, with no dimensionality-reduction step applied

### Requirement: generate()
The system SHALL provide a `generate()` function that samples new images from a trained VAE's latent space, given only the trained model and a requested count. `generate()` SHALL only accept models built by `build_vae()`.

#### Scenario: Student generates new images
- **WHEN** a student calls `generate(vae_model, n=5)` after training a `build_vae()` model
- **THEN** they SHALL receive 5 newly sampled images without manually sampling the latent distribution

#### Scenario: Student calls generate() on an autoencoder model
- **WHEN** a student calls `generate(model, n=5)` where `model` was built by `build_autoencoder()` rather than `build_vae()`
- **THEN** the system SHALL raise an explicit error naming the AE/VAE mismatch, rather than failing inside `_internals` on a missing sampling method

### Requirement: show_latent_space()
The system SHALL provide a visualization helper that displays a trained VAE's 2-dimensional latent space so students can inspect how shape classes are organized without analyzing the model's internals directly.

#### Scenario: Student visualizes the latent space
- **WHEN** a student calls `show_latent_space(vae_model, data)`
- **THEN** a plot SHALL be displayed showing the distribution of encoded data points in latent space, colored by class

### Requirement: CPU training time budget
Training a VAE built with `build_vae()` on the stub dataset via `train()` SHALL complete in no more than a few minutes on a CPU-only machine with no GPU. This budget applies only to `build_vae()` models; `build_autoencoder()` has no enforced budget for MVP (Requirement: build_autoencoder() is a nice-to-have).

#### Scenario: VAE training completes within budget on CPU
- **WHEN** `train()` is run on a CPU-only machine with a `build_vae()` model using the built-in stub dataset with default parameters
- **THEN** training SHALL complete in under 5 minutes

### Requirement: build_autoencoder() is a nice-to-have
`build_autoencoder()` SHALL function correctly (Requirement: build_autoencoder()) but is not required to meet the CPU training time budget (Requirement: CPU training time budget) or any polish/documentation bar beyond correctness — it exists for API completeness and as an optional pedagogical stepping stone, not as a guaranteed student-facing workflow.

#### Scenario: Autoencoder has no enforced time budget
- **WHEN** `train()` is run with a `build_autoencoder()` model using the built-in stub dataset
- **THEN** the system SHALL NOT enforce any wall-clock training time ceiling on this path

### Requirement: GAN excluded from MVP
The system SHALL NOT include a GAN implementation in this change; GAN support SHALL be documented as a future, optional extension rather than built now.

#### Scenario: Documentation states GAN is out of scope
- **WHEN** a student or instructor reads the generator arm's documentation
- **THEN** it SHALL explicitly state that GAN-based generation is a possible future extension, not part of the current library
