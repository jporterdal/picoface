## ADDED Requirements

### Requirement: build_classifier()
The system SHALL provide a `build_classifier(num_classes, input_shape=None)` function that constructs a small CNN classifier sized for a given number of classes and input image shape, without requiring the caller to define network layers.

#### Scenario: Student builds a classifier
- **WHEN** a student calls `build_classifier(num_classes=3, input_shape=(16, 16, 3))`
- **THEN** they SHALL receive a trainable model object without writing any PyTorch `nn.Module` code

#### Scenario: Model shape matches requested class count
- **WHEN** a model is built with `num_classes=N`
- **THEN** the model's output SHALL have exactly N class scores per input image

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
- **THEN** they SHALL find only the named entry-point functions (`build_classifier`, `train`, `evaluate`, `predict`) and no `nn.Module` subclasses or raw training-loop code

### Requirement: Training-history visualization
The system SHALL provide a viz helper function that plots a classifier's training history (loss per epoch) via a single function call, without the student writing plotting code.

#### Scenario: Student plots training progress
- **WHEN** a student calls the training-history plotting helper with the object returned by `train()`
- **THEN** a loss-vs-epoch chart SHALL be produced without the student manipulating matplotlib directly
