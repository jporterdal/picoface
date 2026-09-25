## MODIFIED Requirements

### Requirement: train()
The system SHALL provide a `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3, verbose=True)` function that runs the full training loop (forward pass, loss computation, backward pass, optimizer step, epoch iteration) for a model, given a model and a `Dataset`, and returns a training-history object. It is the same model-agnostic function specified in the `model-interface` capability; for a classifier model built by `build_classifier()` it optimizes classification loss.

#### Scenario: Student trains a classifier in one call
- **WHEN** a student calls `train(model, data)`
- **THEN** the function SHALL run to completion and return a training history without the student writing a training loop

#### Scenario: Training history records loss and wall-clock time
- **WHEN** `train()` completes
- **THEN** the returned history SHALL include a per-epoch loss record and the total wall-clock training duration

#### Scenario: Classifier training behavior is unchanged
- **WHEN** a student trains a `build_classifier()` model with default parameters
- **THEN** it SHALL be optimized on classification loss alone, with the same defaults as before this change
