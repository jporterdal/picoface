## MODIFIED Requirements

### Requirement: Model-agnostic train()
The system SHALL provide a single `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)` function that trains any model built by this library's `build_*` functions on a `Dataset` and returns a training-history object. The loss being optimized SHALL be determined by the model itself, not by which module `train` was imported from or by the caller. `train()` SHALL make the model aware of its progress through training — the current epoch relative to the caller's requested `epochs` — so that a model can schedule its own loss terms without `train()` knowing what kind of model it is training.

#### Scenario: Same call trains different model kinds
- **WHEN** a student calls `train(model, data)` with, in turn, a model from `build_classifier()`, `build_autoencoder()`, and `build_vae()`
- **THEN** each call SHALL run to completion and return a training history, with no change to the call and no student-written training loop

#### Scenario: Each model is trained against its own objective
- **WHEN** `train()` is run on a classifier-only model, on an autoencoder model, and on a VAE model
- **THEN** the classifier's training SHALL optimize classification loss only, the autoencoder's reconstruction loss only, and the VAE's reconstruction, KL-divergence, and classification loss together, without the caller specifying any of them

#### Scenario: Loss schedules follow the requested epoch count
- **WHEN** the same model kind is trained once with a small `epochs` value and once with a large one
- **THEN** any loss schedule the model applies SHALL complete within each run, rather than being defined in absolute epochs

#### Scenario: Training a non-model is rejected clearly
- **WHEN** a student calls `train()` with an object that was not built by this library
- **THEN** the system SHALL raise a clear error naming the problem, rather than an internal attribute or framework failure

### Requirement: Verbs are importable from each arm's public module
The functions `train`, `evaluate`, and `predict` SHALL be importable from `picoface.classifier`, and `train`, `evaluate`, `predict`, and `generate` from `picoface.generator`, and each name SHALL refer to the same callable regardless of which module it is imported from. Each public module SHALL continue to expose only named entry-point functions, exceptions, and the history type — no model classes or training-loop code.

#### Scenario: Both import paths give the same function
- **WHEN** a student imports `train`, `evaluate`, or `predict` from `picoface.classifier` and from `picoface.generator`
- **THEN** both names SHALL refer to the same callable and behave identically for any model

#### Scenario: Existing imports keep working
- **WHEN** existing student code imports `train` from `picoface.generator` or `evaluate`/`predict` from `picoface.classifier`
- **THEN** the imports SHALL succeed and the functions SHALL behave as before for the model kinds that still support them

### Requirement: Model-agnostic evaluate() and predict()
`evaluate(model, data)` and `predict(model, image)` SHALL work on any model that has a classification capability — a CNN classifier or a model built by `build_vae()` — with unchanged signatures, returning accuracy (0 to 1) and a class name respectively. Models built by `build_autoencoder()` do not have this capability. The same shape and class-count consistency checks SHALL apply regardless of model kind.

#### Scenario: Evaluating a joint model with the classifier's function
- **WHEN** a student trains a model from `build_vae(data)` and then calls `evaluate(model, data)`
- **THEN** the function SHALL return a classification accuracy between 0 and 1

#### Scenario: Predicting with a joint model returns a class name
- **WHEN** a student calls `predict(model, image)` on a trained `build_vae()` model
- **THEN** the function SHALL return one of the dataset's class names, not a raw integer index

#### Scenario: Autoencoder is rejected with a capability error
- **WHEN** a student calls `evaluate()` or `predict()` on a `build_autoencoder()` model
- **THEN** the system SHALL raise the library's capability-error type (Requirement: Clear error for missing capabilities), stating that the model cannot classify

#### Scenario: Mismatched data is rejected for any model kind
- **WHEN** a student calls `evaluate()` or `predict()` on a classifying model with data whose image shape or class count differs from the model's
- **THEN** the system SHALL raise a clear shape/class-count error rather than an internal framework failure

#### Scenario: Evaluation is deterministic and does not train
- **WHEN** `evaluate(model, data)` is called twice in a row on the same trained model
- **THEN** it SHALL return the same accuracy both times, and the model's parameters SHALL be unchanged

### Requirement: Shared training history
`train()` SHALL return one history type for all model kinds, containing per-epoch total `loss` and the total `wall_clock_seconds`, plus any additional per-epoch metrics the model reports (reconstruction loss, KL-divergence, classification loss, accuracy, the KL weight applied, and learned task-weighting values). Metrics a model does not report SHALL be present as empty lists, not absent.

#### Scenario: History fields for a classifier-only model
- **WHEN** `train()` completes on a `build_classifier()` model
- **THEN** the history SHALL have a per-epoch `loss` and `wall_clock_seconds`, and its reconstruction and KL metric lists SHALL be empty

#### Scenario: History fields for an autoencoder
- **WHEN** `train()` completes on a `build_autoencoder()` model
- **THEN** the history SHALL have per-epoch `loss` and `reconstruction_loss`, and its KL, classification, accuracy, KL-weight, and task-weighting lists SHALL be empty

#### Scenario: History fields for a joint VAE
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the history SHALL have per-epoch `loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`, `accuracy`, KL weight, and learned task-weighting values, each with one entry per epoch
