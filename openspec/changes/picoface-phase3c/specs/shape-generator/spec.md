## ADDED Requirements

### Requirement: VAE training is supervised and requires labeled data
Training a model built by `build_vae()` SHALL use the `Dataset`'s labels as a supervisory signal, not only its images. The system SHALL reject an attempt to train a `build_vae()` model against a `Dataset` whose class count does not match the model's, with an explicit error rather than a failure inside the training loop.

#### Scenario: Labels influence the trained model
- **WHEN** a `build_vae()` model is trained via `train(model, data)` on a labeled `Dataset`
- **THEN** the resulting model SHALL be able to report a class for an input image, which a model trained on the same images with shuffled labels SHALL NOT reproduce

#### Scenario: Class count mismatch is rejected
- **WHEN** a student calls `train(model, data)` where `data`'s class count differs from the class count the model was built for
- **THEN** the system SHALL raise a clear error naming the mismatch, rather than failing inside the training loop

### Requirement: Classification is routed through the probabilistic latent space
A model built by `build_vae()` SHALL derive its class predictions from the same probabilistic latent representation its decoder consumes, rather than from an independent path that bypasses the latent bottleneck. This is what makes the supervisory signal shape the distribution that `generate()` samples from.

#### Scenario: Supervision reaches the sampled distribution
- **WHEN** a `build_vae()` model is trained on labeled data and images are then sampled via `generate()`
- **THEN** the sampled images SHALL reflect the class structure learned during training, rather than being drawn from a latent space that received no class information

#### Scenario: Inference is deterministic
- **WHEN** `predict(model, image)` is called twice on the same image with the same trained model
- **THEN** both calls SHALL return the same class name, despite the latent space being probabilistic during training

### Requirement: evaluate() and predict() for generative models
The system SHALL provide `evaluate(model, data)` reporting classification accuracy on a `Dataset`, and `predict(model, image)` returning a single image's class name, for models built by `build_vae()` — so that one trained model serves both the classification and the generation workflow without the caller managing tensors or label indices.

#### Scenario: Student evaluates the trained model's accuracy
- **WHEN** a student calls `evaluate(model, data)` on a trained `build_vae()` model
- **THEN** the function SHALL return an accuracy value between 0 and 1 via a single function call

#### Scenario: Student classifies a single image
- **WHEN** a student calls `predict(model, image)` with a single image array
- **THEN** the function SHALL return the predicted class name, not a raw integer index

#### Scenario: Student calls evaluate() on an autoencoder model
- **WHEN** a student calls `evaluate(model, data)` or `predict(model, image)` where `model` was built by `build_autoencoder()`
- **THEN** the system SHALL raise an explicit error naming the AE/VAE mismatch, rather than failing inside internals

### Requirement: KL-divergence weight is annealed, not fixed
The weight applied to the KL-divergence term during VAE training SHALL increase from zero over the course of training according to an internal schedule, rather than being held at a fixed value for every epoch. No student-facing parameter SHALL control this schedule.

#### Scenario: Early training is not dominated by the prior
- **WHEN** a `build_vae()` model is trained and its per-epoch diagnostics are inspected
- **THEN** the KL weight recorded for the first epoch SHALL be lower than the weight recorded for the final epoch

### Requirement: Reconstruction and classification losses are balanced by learned uncertainty
The relative weighting between the reconstruction objective and the classification objective SHALL be learned during training rather than set by a fixed constant, so that no per-dataset weight needs to be chosen by hand. This learned weighting SHALL apply only to those two objectives and SHALL NOT be applied to the KL-divergence term, whose weight is governed by the annealing schedule.

#### Scenario: No hand-tuned task weight is required
- **WHEN** a `build_vae()` model is trained on datasets whose reconstruction and classification difficulty differ substantially
- **THEN** training SHALL proceed without any weighting constant being changed between the two runs

#### Scenario: Learned weights are observable
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the returned training history SHALL expose the learned weighting values per epoch, so that a run which collapses onto one objective can be diagnosed

### Requirement: Regularization strength is independent of image dimensions
The balance between the reconstruction and KL-divergence terms SHALL NOT vary with the image height, width, or channel count of the dataset being trained on. Loss terms SHALL be reduced to a common scale before any weighting is applied.

#### Scenario: Two resolutions regularize equivalently
- **WHEN** the same model architecture is trained on two datasets that differ only in image resolution
- **THEN** the effective weighting between reconstruction and KL-divergence SHALL be equivalent across the two runs, rather than scaling with pixel count

### Requirement: Training history records per-objective diagnostics
For models built by `build_vae()`, the object returned by `train()` SHALL record, per epoch: the reconstruction loss, the KL divergence, the classification loss, the KL weight actually applied, and the learned task weighting values — in addition to the total loss and wall-clock duration it already records.

#### Scenario: A run that collapses onto one objective is diagnosable
- **WHEN** a `build_vae()` model finishes training
- **THEN** the returned history SHALL contain enough per-epoch detail to determine which objective the model optimized and how the weighting evolved, without re-running training or inspecting internals

## MODIFIED Requirements

### Requirement: build_vae()
The system SHALL provide a `build_vae()` function constructing a supervised variational autoencoder from a `Dataset`: a probabilistic latent space with reparameterization, a decoder, and a classification head sized to the `Dataset`'s class count, trained against a combined reconstruction + KL-divergence + classification objective. It SHALL be exposed as a comparable API surface to `build_autoencoder()`, taking the same `build_vae(data)` call shape.

A model built this way SHALL serve both workflows from a single `train()` call: classifying novel images (Requirement: evaluate() and predict() for generative models) and generating new ones (Requirement: generate()).

#### Scenario: Student trains once and uses the model two ways
- **WHEN** a student calls `build_vae(data)` then `train(model, data)` on labeled data
- **THEN** the resulting model SHALL both classify novel images and generate new ones, without a second training call and without the student writing the reparameterization trick, the KL-divergence loss, or the classification loss themselves

#### Scenario: Model output shape matches the dataset's class count
- **WHEN** a model is built from a `Dataset` with N class names
- **THEN** the model SHALL predict among exactly those N classes

#### Scenario: Student upgrades from autoencoder to VAE
- **WHEN** a student replaces `build_autoencoder(data)` with `build_vae(data)` in their assembly code and re-runs `train()`
- **THEN** training SHALL succeed using the same `train()` entry point, without the student writing the reparameterization trick, the KL-divergence loss, or the classification loss themselves — even though the VAE consumes the `Dataset`'s labels and the autoencoder ignores them

### Requirement: CPU training time budget
Training a VAE built with `build_vae()` on the stub dataset via `train()` SHALL complete in no more than a few minutes on a CPU-only machine with no GPU. This budget covers the full combined objective — reconstruction, KL-divergence, and classification trained together in one call — not reconstruction alone. This budget applies only to `build_vae()` models; `build_autoencoder()` has no enforced budget for MVP (Requirement: build_autoencoder() is a nice-to-have).

#### Scenario: VAE training completes within budget on CPU
- **WHEN** `train()` is run on a CPU-only machine with a `build_vae()` model using the built-in stub dataset with default parameters
- **THEN** training SHALL complete in under 5 minutes

#### Scenario: Wall-clock duration is recorded for comparison
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the returned history SHALL record the total wall-clock duration, so that a regression against this budget is visible without separate instrumentation

### Requirement: build_autoencoder() is a nice-to-have
`build_autoencoder()` SHALL function correctly (Requirement: build_autoencoder()) but is not required to meet the CPU training time budget (Requirement: CPU training time budget) or any polish/documentation bar beyond correctness — it exists for API completeness, not as a guaranteed student-facing workflow. It remains unsupervised: `train()` SHALL ignore a `Dataset`'s labels for these models.

The system's documentation SHALL state explicitly that `build_autoencoder()` is not a required step on the way to `build_vae()`. Students never see the underlying architecture of either model, so no autoencoder-to-VAE progression is exposed to them to walk through.

#### Scenario: Autoencoder has no enforced time budget
- **WHEN** `train()` is run with a `build_autoencoder()` model using the built-in stub dataset
- **THEN** the system SHALL NOT enforce any wall-clock training time ceiling on this path

#### Scenario: Documentation does not present the autoencoder as a prerequisite
- **WHEN** a developer or instructor reads the generator arm's documentation
- **THEN** it SHALL state that `build_autoencoder()` is optional and not a prerequisite for using `build_vae()`

## REMOVED Requirements

### Requirement: Fixed 2-dimensional latent space
**Reason**: This requirement existed solely to keep `show_latent_space()` a projection-free (x, y) scatter plot, avoiding a PCA/t-SNE dependency. With that visualization removed (see Requirement: show_latent_space() below), the constraint has no remaining justification — and it has become actively harmful, since classification is now routed through the latent bottleneck, making two dimensions a cap on both classification accuracy and sample quality.

**Migration**: None required. Latent dimensionality remains a non-public internal constant with no student-facing parameter; only the spec-mandated value of 2 is removed. No new dependency is introduced — removing the plot removes the need for the projection library the fixed value existed to avoid. The final value is a Phase 6 tuning decision.

### Requirement: show_latent_space()
**Reason**: Ranked a nice-to-have rather than a necessary capability, and its only structural cost — pinning the latent space to 2 dimensions — is no longer acceptable now that classification is routed through that bottleneck. The visualization also presupposed an unsupervised latent whose class organization was worth inspecting; it was never part of a student-facing notebook.

**Migration**: None required — no student-facing workflow depended on it. Whether some form of latent-space visualization returns, and in what form, is deferred to a later phase; nothing in this change prevents reintroducing one.
