## MODIFIED Requirements

### Requirement: Model-agnostic evaluate() and predict()
`evaluate(model, data)` and `predict(model, image)` SHALL work on any model that has a classification capability — a CNN classifier or a model built by `build_vae()` — with unchanged signatures, returning accuracy (0 to 1) and a class name respectively. Models built by `build_autoencoder()` do not have this capability. The same shape and class-count consistency checks SHALL apply regardless of model kind.

`predict()` SHALL accept, as `image`, either an image array or a picture as defined by the `mediacomp-bridge` capability. A picture SHALL be converted exactly as `picture_to_array()` converts it, and the same grayscale requirement SHALL apply. `predict()` SHALL NOT crop or resize a picture: one whose size differs from the model's SHALL be rejected with the usual shape error.

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

#### Scenario: Predicting from a prepared picture
- **WHEN** a student calls `predict(model, pic)` with a grayscale picture of the model's image size
- **THEN** the function SHALL return the same class name as `predict(model, picture_to_array(pic))`

#### Scenario: A picture of the wrong size is rejected, not resized
- **WHEN** a student calls `predict(model, pic)` with a grayscale 200×200 picture on a model trained on 28×28 images
- **THEN** the system SHALL raise the model's shape error, and SHALL NOT crop or resize the picture

#### Scenario: A color picture is rejected with guidance
- **WHEN** a student calls `predict(model, pic)` with a picture that is not grayscale
- **THEN** the system SHALL raise the same error `picture_to_array()` raises for it
