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
