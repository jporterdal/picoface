## MODIFIED Requirements

### Requirement: Data/model shape consistency is validated
The system SHALL reject an image (whether from a `Dataset` passed to `train()`/`evaluate()`, or a single image passed to `predict()`) whose shape does not match the model's expected input shape, and SHALL reject a `Dataset` passed to `train()`/`evaluate()` whose class count does not match the model's, both with a clear error rather than an internal framework failure. This applies to any model accepted by these functions (Requirement: Model-agnostic evaluate() and predict() in `model-interface`), not only CNN classifiers.

#### Scenario: Student trains with mismatched data
- **WHEN** a student calls `train(model, data)` where `data`'s image shape or class count differs from the model's
- **THEN** the system SHALL raise a clear error rather than failing inside the training loop

#### Scenario: Student predicts with a mismatched image
- **WHEN** a student calls `predict(model, image)` where `image`'s shape differs from the model's expected input shape
- **THEN** the system SHALL raise a clear error rather than failing inside the forward pass

#### Scenario: Mismatch is rejected for a joint model too
- **WHEN** a student calls `evaluate(model, data)` on a `build_vae()` model with a `Dataset` whose class count differs from the model's
- **THEN** the system SHALL raise a clear class-count error rather than an internal framework failure

### Requirement: train()
The system SHALL provide a `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)` function that runs the full training loop (forward pass, loss computation, backward pass, optimizer step, epoch iteration) for a model, given a model and a `Dataset`, and returns a training-history object. It is the same model-agnostic function specified in the `model-interface` capability; for a classifier model built by `build_classifier()` it optimizes classification loss.

#### Scenario: Student trains a classifier in one call
- **WHEN** a student calls `train(model, data)`
- **THEN** the function SHALL run to completion and return a training history without the student writing a training loop

#### Scenario: Training history records loss and wall-clock time
- **WHEN** `train()` completes
- **THEN** the returned history SHALL include a per-epoch loss record and the total wall-clock training duration

#### Scenario: Classifier training behavior is unchanged
- **WHEN** a student trains a `build_classifier()` model with default parameters
- **THEN** it SHALL be optimized on classification loss alone, with the same defaults as before this change

### Requirement: evaluate() and predict()
The system SHALL provide `evaluate(model, data)` to report classification accuracy on a `Dataset`, and `predict(model, image)` to classify a single new image and return its class name, both without exposing internal tensor manipulation to the caller. Both SHALL accept any model with a classification capability, including models built by `build_autoencoder()` and `build_vae()` (see `model-interface`).

#### Scenario: Student evaluates a trained classifier
- **WHEN** a student calls `evaluate(model, data)` after training
- **THEN** the function SHALL return an accuracy value between 0 and 1 via a single function call

#### Scenario: Student predicts a single image's class
- **WHEN** a student calls `predict(model, image)` with a single image array
- **THEN** the function SHALL return the predicted class name (not a raw integer index)

#### Scenario: Same functions work on a joint model
- **WHEN** a student calls `evaluate(model, data)` and `predict(model, image)` on a trained `build_vae()` model
- **THEN** the functions SHALL return accuracy and a class name respectively, with no different call shape from the CNN case

### Requirement: Training-history visualization
The system SHALL provide a viz helper function that plots a training history (loss per epoch, and classification accuracy per epoch when the history contains it) via a single function call, without the student writing plotting code.

#### Scenario: Student plots training progress
- **WHEN** a student calls the training-history plotting helper with the object returned by `train()`
- **THEN** a loss-vs-epoch chart SHALL be produced without the student manipulating matplotlib directly

#### Scenario: Accuracy is plotted when available
- **WHEN** the helper is given the history of a joint model trained with a classification branch
- **THEN** the resulting figure SHALL also show classification accuracy per epoch

#### Scenario: Classifier-only history plots as before
- **WHEN** the helper is given the history of a `build_classifier()` model
- **THEN** the figure SHALL show loss per epoch, with no error from the missing reconstruction or KL metrics
