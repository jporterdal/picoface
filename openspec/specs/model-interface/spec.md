## Purpose

The model-agnostic public verbs of the library (`train`, `evaluate`, `predict`, `generate`) and the contract every model object satisfies to be used with them, so that a student's calls do not change with the kind of model they built and new model kinds can be added without new public functions.

## Requirements

### Requirement: Model-agnostic train()
The system SHALL provide a single `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)` function that trains any model built by this library's `build_*` functions on a `Dataset` and returns a training-history object. The loss being optimized SHALL be determined by the model itself, not by which module `train` was imported from or by the caller.

#### Scenario: Same call trains different model kinds
- **WHEN** a student calls `train(model, data)` with, in turn, a model from `build_classifier()`, `build_autoencoder()`, and `build_vae()`
- **THEN** each call SHALL run to completion and return a training history, with no change to the call and no student-written training loop

#### Scenario: Each model is trained against its own objective
- **WHEN** `train()` is run on a classifier-only model, and separately on a VAE model
- **THEN** the classifier's training SHALL optimize classification loss only, and the VAE's SHALL optimize reconstruction, KL-divergence, and classification loss together, without the caller specifying either

#### Scenario: Training a non-model is rejected clearly
- **WHEN** a student calls `train()` with an object that was not built by this library
- **THEN** the system SHALL raise a clear error naming the problem, rather than an internal attribute or framework failure

### Requirement: Verbs are importable from each arm's public module
The functions `train`, `evaluate`, and `predict` SHALL be importable from `picoface.classifier`, and `train` and `generate` from `picoface.generator`, and each name SHALL refer to the same callable regardless of which module it is imported from. Each public module SHALL continue to expose only named entry-point functions, exceptions, and the history type — no model classes or training-loop code.

#### Scenario: Both import paths give the same function
- **WHEN** a student imports `train` from `picoface.classifier` and from `picoface.generator`
- **THEN** both names SHALL refer to the same callable and behave identically for any model

#### Scenario: Existing imports keep working
- **WHEN** existing student code imports `train` from `picoface.generator` or `evaluate`/`predict` from `picoface.classifier`
- **THEN** the imports SHALL succeed and the functions SHALL behave as before for the model kinds they previously supported

### Requirement: Model-agnostic evaluate() and predict()
`evaluate(model, data)` and `predict(model, image)` SHALL work on any model that has a classification capability — including a CNN classifier and a model built by `build_autoencoder()` or `build_vae()` — with unchanged signatures, returning accuracy (0 to 1) and a class name respectively. The same shape and class-count consistency checks SHALL apply regardless of model kind.

#### Scenario: Evaluating a joint model with the classifier's function
- **WHEN** a student trains a model from `build_vae(data)` and then calls `evaluate(model, data)`
- **THEN** the function SHALL return a classification accuracy between 0 and 1

#### Scenario: Predicting with a joint model returns a class name
- **WHEN** a student calls `predict(model, image)` on a trained `build_vae()` or `build_autoencoder()` model
- **THEN** the function SHALL return one of the dataset's class names, not a raw integer index

#### Scenario: Mismatched data is rejected for any model kind
- **WHEN** a student calls `evaluate()` or `predict()` on a joint model with data whose image shape or class count differs from the model's
- **THEN** the system SHALL raise a clear shape/class-count error rather than an internal framework failure

#### Scenario: Evaluation is deterministic and does not train
- **WHEN** `evaluate(model, data)` is called twice in a row on the same trained model
- **THEN** it SHALL return the same accuracy both times, and the model's parameters SHALL be unchanged

### Requirement: Model-agnostic generate()
`generate(model, n)` SHALL work on any model that has a sampling capability and SHALL return `n` newly sampled images as a `uint8` array in the model's image shape. It SHALL raise an explicit, named error for a model without that capability, stating the required capability rather than failing inside internals.

#### Scenario: Sampling from a VAE
- **WHEN** a student calls `generate(vae_model, n=5)` on a trained `build_vae()` model
- **THEN** they SHALL receive 5 images of the model's image shape and `uint8` dtype

#### Scenario: Model without sampling capability is rejected
- **WHEN** a student calls `generate(model, n=5)` with a `build_autoencoder()` model or a `build_classifier()` model
- **THEN** the system SHALL raise an explicit error identifying that the model cannot generate, and this error SHALL be the same exception type previously raised for an autoencoder

### Requirement: Clear error for missing capabilities
When a verb requires a capability that the given model lacks, the system SHALL raise a clear error that names the verb and states what the model cannot do. All such errors SHALL share a common exception type so a single `except` can catch them.

#### Scenario: Shared exception type for capability errors
- **WHEN** a student catches the library's capability-error type around `generate(cnn_classifier, 5)` and around `generate(autoencoder_model, 5)`
- **THEN** both failures SHALL be caught by that single handler

### Requirement: Shared training history
`train()` SHALL return one history type for all model kinds, containing per-epoch total `loss` and the total `wall_clock_seconds`, plus any additional per-epoch metrics the model reports (reconstruction loss, KL-divergence, classification loss, accuracy). Metrics a model does not report SHALL be present as empty lists, not absent.

#### Scenario: History fields for a classifier-only model
- **WHEN** `train()` completes on a `build_classifier()` model
- **THEN** the history SHALL have a per-epoch `loss` and `wall_clock_seconds`, and its reconstruction and KL metric lists SHALL be empty

#### Scenario: History fields for a joint VAE
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the history SHALL have per-epoch `loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`, and `accuracy`, each with one entry per epoch

### Requirement: Shape errors share a common base type
The classifier arm's and generator arm's shape/class-count error types SHALL share a common base type, so that a single `except` clause catches a shape error raised through either arm's functions, while each arm's own error type remains importable and catchable individually.

#### Scenario: One handler catches both arms' shape errors
- **WHEN** a student catches the shared base type around a mismatched-shape `predict()` call on a CNN classifier and around a mismatched-shape `evaluate()` call on a joint VAE
- **THEN** both errors SHALL be caught by that handler
