## ADDED Requirements

### Requirement: The CNN classifier is the project's independent external validator
The CNN classifier built by `build_classifier()` SHALL remain a separately-constructed, separately-trained model that shares no learned parameters with the generative model built by `build_vae()`. Its role is to provide an independent judgement of generated image quality: a measurement in which the generative model grades its own output shares an encoder with the thing being graded and is therefore not evidence of quality.

This capability SHALL remain interoperable with the rest of the project — consuming the same `Dataset` contract and exposing the same public call shapes (`train()`, `evaluate()`, `predict()`) as the generative arm — even though it is no longer the primary student-facing classification path.

#### Scenario: Generated images are scored by an independent model
- **WHEN** images produced by a trained `build_vae()` model are classified by a separately-trained `build_classifier()` model
- **THEN** the resulting accuracy SHALL reflect a judgement made by a model that shares no learned parameters with the generator

#### Scenario: The classifier keeps pace with the shared dataset contract
- **WHEN** a `Dataset` valid for the generative arm is passed to `build_classifier()` and `train()`
- **THEN** it SHALL be accepted without conversion or adaptation, using the same call shapes as the generative arm

### Requirement: Classification accuracy is regression-guarded
The project's tests SHALL assert classifier accuracy against a bound that a broken model would fail, measured on data held out from training. An assertion that merely confirms accuracy lies within the range 0 to 1 SHALL NOT be treated as satisfying this requirement, as it holds for any value and detects no regression.

#### Scenario: Accuracy is measured on unseen data
- **WHEN** classifier accuracy is asserted in the project's tests
- **THEN** it SHALL be computed on a dataset disjoint from the one used for training, rather than on the training data itself

#### Scenario: A broken model fails the assertion
- **WHEN** a classifier that predicts without regard to its input is evaluated
- **THEN** the accuracy assertion SHALL fail

## MODIFIED Requirements

### Requirement: evaluate() and predict()
The system SHALL provide `evaluate(model, data)` to report classification accuracy on a `Dataset`, and `predict(model, image)` to classify a single new image and return its class name, both without exposing internal tensor manipulation to the caller. Both SHALL accept any model with a classification capability — a model built by `build_classifier()` or by `build_vae()` (see `model-interface`). Models built by `build_autoencoder()` do not classify.

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
- **WHEN** the helper is given the history of a `build_vae()` model, which records classification accuracy
- **THEN** the resulting figure SHALL also show classification accuracy per epoch

#### Scenario: Classifier-only history plots as before
- **WHEN** the helper is given the history of a `build_classifier()` model
- **THEN** the figure SHALL show loss per epoch, with no error from the missing reconstruction or KL metrics
