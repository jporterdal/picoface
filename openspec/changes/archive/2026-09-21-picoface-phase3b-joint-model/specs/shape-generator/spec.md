## MODIFIED Requirements

### Requirement: build_autoencoder()
The system SHALL provide a `build_autoencoder()` function constructing a basic (non-variational) encoder/decoder model from a `Dataset`, as a pedagogical stepping stone toward the VAE. The model SHALL also carry a classification branch (Requirement: Joint classification branch), so that it both reconstructs and classifies. It SHALL accept the same call shape as `build_vae()` (Requirement: build_vae()), so the two are interchangeable at the call site.

#### Scenario: Student builds and trains a plain autoencoder
- **WHEN** a student calls `build_autoencoder(data)` then `train(model, data)`
- **THEN** the model SHALL learn to reconstruct input images and to classify them, without the student implementing encoder/decoder layers

### Requirement: build_vae()
The system SHALL provide a `build_vae()` function constructing a variational autoencoder (probabilistic latent space, reparameterization, combined reconstruction + KL-divergence loss) from a `Dataset`, exposed as a comparable API surface to `build_autoencoder()`. The model SHALL also carry a classification branch trained jointly with the generative objective (Requirement: Joint classification branch).

#### Scenario: Student upgrades from autoencoder to VAE
- **WHEN** a student replaces `build_autoencoder(data)` with `build_vae(data)` in their assembly code and re-runs `train()`
- **THEN** training SHALL succeed using the same `train()` entry point, without the student writing the reparameterization trick or KL-divergence loss themselves

#### Scenario: The swap adds only the probabilistic latent
- **WHEN** a student swaps `build_autoencoder(data)` for `build_vae(data)`
- **THEN** both models SHALL expose the same classification behavior (`evaluate`/`predict`) and differ only in having a probabilistic latent space that can be sampled

### Requirement: generate()
The system SHALL provide a `generate()` function that samples new images from a trained VAE's latent space, given only the trained model and a requested count. `generate()` SHALL only accept models with a sampling capability, which are models built by `build_vae()`. Generation SHALL remain unconditional: it SHALL NOT take a class argument.

#### Scenario: Student generates new images
- **WHEN** a student calls `generate(vae_model, n=5)` after training a `build_vae()` model
- **THEN** they SHALL receive 5 newly sampled images without manually sampling the latent distribution

#### Scenario: Student calls generate() on an autoencoder model
- **WHEN** a student calls `generate(model, n=5)` where `model` was built by `build_autoencoder()` rather than `build_vae()`
- **THEN** the system SHALL raise an explicit error naming the AE/VAE mismatch, rather than failing inside `_internals` on a missing sampling method

## ADDED Requirements

### Requirement: Joint classification branch
Models built by `build_autoencoder()` and `build_vae()` SHALL include a classification branch producing one score per class of the `Dataset` they were built from, in parallel with the generative branch and reading from the encoder's shared feature representation (not from the sampled or mean latent vector). `train()` SHALL optimize the generative objective and a cross-entropy classification loss together in a single training run, with the classification loss weight fixed internally and not student-configurable.

#### Scenario: Model classifies after a single training call
- **WHEN** a student builds a model with `build_vae(data)` and trains it with a single `train(model, data, epochs=50)` call on the stub dataset (at least two classes)
- **THEN** `evaluate(model, data)` on the same data SHALL return an accuracy above chance level (1 divided by the number of classes)

Note: how many epochs are needed depends on dataset size, because each epoch performs one optimizer step per batch. The tiny stub dataset (16 images, one step per epoch at the default batch size) needs more epochs than `train()`'s default of 10; this scenario therefore fixes the epoch count rather than promising it at the default.

#### Scenario: Model output covers each class
- **WHEN** a model is built from a `Dataset` with N class names
- **THEN** its classification branch SHALL produce exactly N class scores per image, and `predict()` SHALL return one of those N class names

#### Scenario: Generative training is not degraded by the classification branch
- **WHEN** a VAE is trained with the classification branch on the stub dataset for the default number of epochs
- **THEN** its per-epoch reconstruction loss SHALL decrease over training, as it does without the branch

#### Scenario: Classification loss weight is not exposed
- **WHEN** a student inspects the signatures of `build_autoencoder()`, `build_vae()`, and `train()`
- **THEN** none SHALL accept a parameter controlling the classification loss weight

### Requirement: Classification metrics in training history
For models built by `build_autoencoder()` or `build_vae()`, the history returned by `train()` SHALL record per-epoch classification loss and classification accuracy (measured on the training batches seen during that epoch), in addition to the total loss and — for a VAE — reconstruction and KL-divergence loss.

#### Scenario: History exposes accuracy rising over training
- **WHEN** a student trains a `build_vae()` model on the stub dataset and reads the returned history
- **THEN** it SHALL contain one classification-accuracy value per epoch, each between 0 and 1

### Requirement: Latent-space visualization scope unchanged
The classification branch SHALL NOT change what `show_latent_space()` plots: it SHALL continue to display the VAE's 2D latent mean per data point, colored by class, and the system SHALL make no guarantee that classes form separated clusters in that plot.

#### Scenario: Latent plot still one point per image
- **WHEN** a student calls `show_latent_space(vae_model, data)` on a trained joint model
- **THEN** one point per input image SHALL be plotted in 2D latent coordinates, colored by class, with no dimensionality-reduction step
