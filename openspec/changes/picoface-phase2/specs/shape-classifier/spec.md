## ADDED Requirements

### Requirement: build_classifier()
The system SHALL provide a `build_classifier(data)` function that constructs a small CNN classifier sized for the given `Dataset`'s class count and image shape, without requiring the caller to define network layers or derive shape/class-count values themselves.

#### Scenario: Student builds a classifier from a dataset
- **WHEN** a student calls `build_classifier(data)` with a loaded or stub `Dataset`
- **THEN** they SHALL receive a trainable model object without writing any PyTorch `nn.Module` code and without deriving `num_classes` or `input_shape` themselves

#### Scenario: Model shape matches the dataset's class count
- **WHEN** a model is built from a `Dataset` with N class names
- **THEN** the model's output SHALL have exactly N class scores per input image

### Requirement: build_classifier_from_shape()
The system SHALL provide a `build_classifier_from_shape(num_classes, input_shape)` function that constructs the same kind of CNN classifier from an explicit class count and image shape, for use when no `Dataset` is yet available.

#### Scenario: Student builds a classifier without a dataset in hand
- **WHEN** a student calls `build_classifier_from_shape(num_classes=3, input_shape=(16, 16, 3))`
- **THEN** they SHALL receive a trainable model object equivalent to one that `build_classifier()` would return for a `Dataset` with matching class count and image shape

### Requirement: Minimum input size is validated at build time
The system SHALL reject an `input_shape` too small for the fixed CNN architecture to process, at the point `build_classifier()` or `build_classifier_from_shape()` is called, with an explicit error identifying the minimum viable size rather than failing later during training.

#### Scenario: Student passes a too-small input shape
- **WHEN** a student calls `build_classifier_from_shape(num_classes=2, input_shape=(2, 2, 3))`
- **THEN** the system SHALL raise a clear error naming the minimum viable input size, instead of failing inside training

### Requirement: Data/model shape consistency is validated
The system SHALL reject an image (whether from a `Dataset` passed to `train()`/`evaluate()`, or a single image passed to `predict()`) whose shape does not match the model's expected input shape, and SHALL reject a `Dataset` passed to `train()`/`evaluate()` whose class count does not match the model's, both with a clear error rather than an internal framework failure.

#### Scenario: Student trains with mismatched data
- **WHEN** a student calls `train(model, data)` where `data`'s image shape or class count differs from the model's
- **THEN** the system SHALL raise a clear error rather than failing inside the training loop

#### Scenario: Student predicts with a mismatched image
- **WHEN** a student calls `predict(model, image)` where `image`'s shape differs from the model's expected input shape
- **THEN** the system SHALL raise a clear error rather than failing inside the forward pass

### Requirement: train()
The system SHALL provide a `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3)` function that runs the full training loop (forward pass, loss computation, backward pass, optimizer step, epoch iteration) for a classifier model, given a model and a `Dataset`, and returns a training-history object.

#### Scenario: Student trains a classifier in one call
- **WHEN** a student calls `train(model, data)`
- **THEN** the function SHALL run to completion and return a training history without the student writing a training loop

#### Scenario: Training history records loss and wall-clock time
- **WHEN** `train()` completes
- **THEN** the returned history SHALL include a per-epoch loss record and the total wall-clock training duration

### Requirement: evaluate() and predict()
The system SHALL provide `evaluate(model, data)` to report classification accuracy on a `Dataset`, and `predict(model, image)` to classify a single new image and return its class name, both without exposing internal tensor manipulation to the caller.

#### Scenario: Student evaluates a trained classifier
- **WHEN** a student calls `evaluate(model, data)` after training
- **THEN** the function SHALL return an accuracy value between 0 and 1 via a single function call

#### Scenario: Student predicts a single image's class
- **WHEN** a student calls `predict(model, image)` with a single image array
- **THEN** the function SHALL return the predicted class name (not a raw integer index)

### Requirement: CPU training time budget
Training a classifier built with `build_classifier()` on the stub dataset via `train()` with default parameters SHALL complete in no more than a few minutes on a CPU-only machine with no GPU.

#### Scenario: Training completes within budget on CPU
- **WHEN** `train()` is run on a CPU-only machine using the built-in stub dataset with default parameters
- **THEN** training SHALL complete in under 5 minutes

### Requirement: Training-loop internals hidden
All model-definition and training-loop code SHALL live outside the public API surface (in a non-public internals module), such that students calling the public functions are never required to read or understand it.

#### Scenario: Public API has no exposed internals
- **WHEN** a student inspects the public `picoface.classifier` module
- **THEN** they SHALL find only the named entry-point functions (`build_classifier`, `build_classifier_from_shape`, `train`, `evaluate`, `predict`) and no `nn.Module` subclasses or raw training-loop code

### Requirement: Training-history visualization
The system SHALL provide a viz helper function that plots a classifier's training history (loss per epoch) via a single function call, without the student writing plotting code.

#### Scenario: Student plots training progress
- **WHEN** a student calls the training-history plotting helper with the object returned by `train()`
- **THEN** a loss-vs-epoch chart SHALL be produced without the student manipulating matplotlib directly
