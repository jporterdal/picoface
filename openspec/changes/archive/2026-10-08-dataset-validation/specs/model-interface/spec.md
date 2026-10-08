## ADDED Requirements

### Requirement: Wrong kind of input is named in the error
Every public function that takes a dataset argument (`build_classifier`, `build_autoencoder`, `build_vae`, `train`, `evaluate`, and `classify_generated`) SHALL raise `TypeError` when that argument is not a dataset value. The message SHALL name the function, the kind of object received (for example a picture, an image array, or a file path), and how to get a dataset (`load_dataset()`). When the object received is a single picture or image array, the message SHALL also point to `predict()` for classifying one image.

`predict()` SHALL check that an image array it is given is a single uint8 H×W×C image before using it. A violation SHALL raise the model's shape error type (Requirement: Shape errors share a common base type), with a message that states what was expected, what was found, and how to fix it:
- a two-dimensional H×W array SHALL be rejected with a message showing how to add the channel axis;
- a floating-point array SHALL be rejected with a message stating that pixel values must be whole numbers from 0 to 255;
- a batch of images (N×H×W×C) SHALL be rejected with a message pointing to `evaluate()` for many labeled images, or to indexing a single image out of the batch.

#### Scenario: A picture passed where a dataset is expected
- **WHEN** a student calls `evaluate(model, pic)` with a picture instead of a dataset
- **THEN** the system SHALL raise `TypeError`, stating that `evaluate()` needs a dataset from `load_dataset()` and that `predict()` classifies a single image

#### Scenario: A file path passed where a dataset is expected
- **WHEN** a student calls `build_classifier("train.npz")`
- **THEN** the system SHALL raise `TypeError`, stating that the path must first be loaded with `load_dataset()`

#### Scenario: A two-dimensional image array
- **WHEN** a student calls `predict(model, image)` with a uint8 array of shape (28, 28) on a model trained on 28×28×1 images
- **THEN** the system SHALL raise the model's shape error, reporting shape (28, 28) and showing how to make it (28, 28, 1), rather than reporting a truncated shape

#### Scenario: A float image array
- **WHEN** a student calls `predict(model, image)` with a float array of the model's image shape
- **THEN** the system SHALL raise the model's shape error, stating that pixel values must be uint8 whole numbers from 0 to 255, and SHALL NOT return a prediction

#### Scenario: A batch passed to predict
- **WHEN** a student calls `predict(model, data.images)` with every image in a dataset
- **THEN** the system SHALL raise the model's shape error, stating that `predict()` takes one image, and pointing to `evaluate()` or to `predict(model, data.images[i])`
