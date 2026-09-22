## ADDED Requirements

### Requirement: Optional latent-access capability
The abstract model contract SHALL support an optional `latent_access` capability, consisting of two operations: encoding a batch of images to their latent mean vectors, and decoding an arbitrary latent vector of the model's latent dimensionality back into an image in the model's image shape. Only models with a probabilistic latent space and a decoder (such as those built by `build_vae()`) SHALL declare this capability; models without a latent space (such as `build_classifier()` or `build_autoencoder()`) SHALL NOT declare it. This capability introduces no new student-facing verb and does not change the signature or behavior of `train`, `evaluate`, `predict`, or `generate` — it exists for the library's own internal composition (Requirement: classify_generated(), in `capstone-linkage`).

#### Scenario: Encoding recovers latent means without side effects
- **WHEN** the latent-access capability's encode operation is used on a batch of images with a model that declares it
- **THEN** it SHALL return one latent mean vector per image, and the model's trained parameters SHALL be unchanged

#### Scenario: Decoding an arbitrary latent vector produces an image
- **WHEN** the latent-access capability's decode operation is used with a latent vector of the model's latent dimensionality
- **THEN** it SHALL return an image in the model's image shape, using the same decoder `generate()` samples through

#### Scenario: A model without a latent space lacks this capability
- **WHEN** a function requiring the `latent_access` capability is given a model that does not declare it (e.g. a `build_classifier()` model)
- **THEN** the system SHALL raise the library's shared capability-error type (Requirement: Clear error for missing capabilities), naming the missing capability

#### Scenario: Existing verbs are unaffected
- **WHEN** a student calls `train`, `evaluate`, `predict`, or `generate` on any model, whether or not it declares the `latent_access` capability
- **THEN** each call's signature and behavior SHALL be exactly as specified before this capability existed
