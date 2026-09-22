## ADDED Requirements

### Requirement: VAE training is supervised and requires labeled data
Training a model built by `build_vae()` SHALL use the `Dataset`'s labels as a supervisory signal, not only its images. The system SHALL reject an attempt to train a `build_vae()` model against a `Dataset` whose class count does not match the model's, with an explicit error rather than a failure inside the training loop.

#### Scenario: Labels influence the trained model
- **WHEN** a `build_vae()` model is trained via `train(model, data)` on a labeled `Dataset`, and a second model is trained identically on the same images with shuffled labels
- **THEN** the first model's accuracy on held-out data SHALL exceed the second's

#### Scenario: Class count mismatch is rejected
- **WHEN** a student calls `train(model, data)` where `data`'s class count differs from the class count the model was built for
- **THEN** the system SHALL raise a clear error naming the mismatch, rather than failing inside the training loop

### Requirement: Classification is routed through the probabilistic latent space
A model built by `build_vae()` SHALL derive its class predictions from the same probabilistic latent representation its decoder consumes, rather than from an independent path that bypasses the latent bottleneck. During training the classifier SHALL read the sampled latent vector; for inference it SHALL read the latent mean. This is what makes the supervisory signal shape the distribution that `generate()` samples from.

#### Scenario: Supervision reaches the sampled distribution
- **WHEN** a `build_vae()` model is trained on labeled data and images are then sampled via `generate()`
- **THEN** the sampled images SHALL reflect the class structure learned during training, rather than being drawn from a latent space that received no class information

#### Scenario: Inference is deterministic
- **WHEN** `predict(model, image)` is called twice on the same image with the same trained model
- **THEN** both calls SHALL return the same class name, despite the latent space being probabilistic during training

#### Scenario: Classification remains differentiable with respect to the input
- **WHEN** the model's class scores for an image are computed through the latent mean
- **THEN** they SHALL be differentiable with respect to the input pixels, so that later input-optimization features can use them

### Requirement: evaluate() and predict() for generative models
The system SHALL provide `evaluate(model, data)` reporting classification accuracy on a `Dataset`, and `predict(model, image)` returning a single image's class name, for models built by `build_vae()` — so that one trained model serves both the classification and the generation workflow without the caller managing tensors or label indices. These SHALL be the same model-agnostic functions specified by `model-interface`, importable from `picoface.generator`.

#### Scenario: Student evaluates the trained model's accuracy
- **WHEN** a student calls `evaluate(model, data)` on a trained `build_vae()` model
- **THEN** the function SHALL return an accuracy value between 0 and 1 via a single function call

#### Scenario: Student classifies a single image
- **WHEN** a student calls `predict(model, image)` with a single image array
- **THEN** the function SHALL return the predicted class name, not a raw integer index

#### Scenario: Student calls evaluate() on an autoencoder model
- **WHEN** a student calls `evaluate(model, data)` or `predict(model, image)` where `model` was built by `build_autoencoder()`
- **THEN** the system SHALL raise an explicit error naming the AE/VAE mismatch, rather than failing inside internals

### Requirement: KL-divergence weight is annealed to the ELBO weight
The weight applied to the KL-divergence term during VAE training SHALL increase from zero over the course of training according to an internal schedule, rather than being held at a fixed value for every epoch, and SHALL reach 1 — the weight at which the reconstruction and KL-divergence terms together form the evidence lower bound (Requirement: Loss terms form a per-pixel ELBO) — by the end of training. The schedule SHALL be defined relative to the number of epochs the caller requested. No student-facing parameter SHALL control this schedule.

#### Scenario: Early training is not dominated by the prior
- **WHEN** a `build_vae()` model is trained and its per-epoch diagnostics are inspected
- **THEN** the KL weight recorded for the first epoch SHALL be lower than the weight recorded for the final epoch

#### Scenario: The schedule ends at the ELBO weight
- **WHEN** a `build_vae()` model is trained for any number of epochs and its per-epoch diagnostics are inspected
- **THEN** the KL weight recorded for the final epoch SHALL be 1

### Requirement: Reconstruction and classification losses are balanced by learned uncertainty
The relative weighting between the reconstruction objective and the classification objective SHALL be learned during training rather than set by a fixed constant, so that no per-dataset weight needs to be chosen by hand. The reconstruction objective's learned parameter SHALL be a per-pixel noise scale of the decoder's likelihood. This learned weighting SHALL apply only to those two objectives and SHALL NOT be applied to the KL-divergence term, whose weight is governed by the annealing schedule.

#### Scenario: No hand-tuned task weight is required
- **WHEN** a `build_vae()` model is trained on datasets whose reconstruction and classification difficulty differ substantially
- **THEN** training SHALL proceed without any weighting constant being changed between the two runs

#### Scenario: Learned weights are observable
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the returned training history SHALL expose the learned weighting values per epoch, so that a run which collapses onto one objective can be diagnosed

#### Scenario: Classification loss weight is not exposed
- **WHEN** a student inspects the signatures of `build_autoencoder()`, `build_vae()`, and `train()`
- **THEN** none SHALL accept a parameter controlling the weighting between reconstruction, classification, or KL-divergence

### Requirement: Loss terms form a per-pixel ELBO
The reconstruction term SHALL be the per-pixel Gaussian negative log-likelihood of the input under the decoder, using the learned per-pixel noise scale (Requirement: Reconstruction and classification losses are balanced by learned uncertainty), and the KL-divergence term SHALL be the per-image KL divergence divided by the number of pixel values per image (height × width × channels). At a KL weight of 1 these two terms together SHALL equal the negative evidence lower bound divided by that pixel count, up to an additive constant.

As a consequence, the balance between the reconstruction and classification objectives SHALL NOT scale with image dimensions, and the learned per-pixel noise scale SHALL be comparable across image resolutions. The KL-divergence term's weight relative to both objectives SHALL fall in inverse proportion to the pixel count, as the evidence lower bound prescribes; this is intended behavior, not a defect.

#### Scenario: Task balance is independent of resolution
- **WHEN** the same model architecture is trained on two datasets that differ only in image resolution
- **THEN** the learned reconstruction noise scales SHALL be comparable across the two runs, and the balance between reconstruction and classification SHALL NOT differ in proportion to the pixel count

#### Scenario: No hand-tuned KL weight remains
- **WHEN** a developer inspects the VAE's loss computation
- **THEN** the KL-divergence weight SHALL come only from the annealing schedule, with no separate fixed multiplier applied to the KL term

### Requirement: Training history records per-objective diagnostics
For models built by `build_vae()`, the object returned by `train()` SHALL record, per epoch: the reconstruction loss, the KL divergence, the classification loss, the classification accuracy measured on the training batches seen during that epoch, the KL weight actually applied, and the learned task weighting values — in addition to the total loss and wall-clock duration it already records.

#### Scenario: A run that collapses onto one objective is diagnosable
- **WHEN** a `build_vae()` model finishes training
- **THEN** the returned history SHALL contain enough per-epoch detail to determine which objective the model optimized and how the weighting evolved, without re-running training or inspecting internals

#### Scenario: History exposes training accuracy
- **WHEN** a student trains a `build_vae()` model on the stub dataset and reads the returned history
- **THEN** it SHALL contain one classification-accuracy value per epoch, each between 0 and 1

## MODIFIED Requirements

### Requirement: build_autoencoder()
The system SHALL provide a `build_autoencoder()` function constructing a basic (non-variational) encoder/decoder model from a `Dataset`. The model SHALL be reconstruction-only: it SHALL NOT classify, and training it SHALL ignore the `Dataset`'s labels. It SHALL accept the same call shape as `build_vae()` (Requirement: build_vae()), so the two are interchangeable at the call site.

#### Scenario: Student builds and trains a plain autoencoder
- **WHEN** a student calls `build_autoencoder(data)` then `train(model, data)`
- **THEN** the model SHALL learn to reconstruct input images, without the student implementing encoder/decoder layers

#### Scenario: Labels are present but ignored
- **WHEN** a `build_autoencoder()` model is trained on a labeled `Dataset` of any class count
- **THEN** training SHALL succeed without validating or using the labels

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

#### Scenario: The swap adds only the probabilistic latent
- **WHEN** a student swaps `build_autoencoder(data)` for `build_vae(data)`
- **THEN** the VAE SHALL additionally support `evaluate`/`predict` and `generate`, neither of which the autoencoder supports

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

### Requirement: Joint classification branch
**Reason**: Phase 3b placed the classification head on the shared conv-trunk features of both the autoencoder and the VAE, with a fixed internal loss weight. This change supersedes all three choices: the head reads the probabilistic latent (Requirement: Classification is routed through the probabilistic latent space), only the VAE classifies (Requirement: build_autoencoder()), and the weight is learned (Requirement: Reconstruction and classification losses are balanced by learned uncertainty).

**Migration**: None required for students — call shapes are unchanged. `evaluate()`/`predict()` on a `build_autoencoder()` model now raise the generator arm's capability error. The "weight is not exposed" guarantee is carried forward in Requirement: Reconstruction and classification losses are balanced by learned uncertainty.

### Requirement: Classification metrics in training history
**Reason**: Superseded by Requirement: Training history records per-objective diagnostics, which records per-epoch classification loss and accuracy for `build_vae()` models alongside the new loss diagnostics. `build_autoencoder()` models no longer classify, so they no longer record classification metrics.

**Migration**: None required. The history fields keep their names; for autoencoder models the classification lists are now empty, per the shared-history contract in `model-interface`.

### Requirement: Latent-space visualization scope unchanged
**Reason**: It constrained what `show_latent_space()` plots, and that function is removed (see Requirement: show_latent_space() above).

**Migration**: None required.
