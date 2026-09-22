## Purpose

The functions that tie the classifier arm (Arm 1, the independent judge) and the generator arm (Arm 2, the student's model) together into the project's closing capstone exercise: checking whether generated images read as their intended class, and visualizing what the classifier has learned to associate with a class.

## ADDED Requirements

### Requirement: classify_generated()
The system SHALL provide a `classify_generated(classifier_model, generator_model, data, n)` function that, for each class in `data`, produces `n` images intended for that class from `generator_model`'s trained latent space, classifies each with `classifier_model`, and reports how often `classifier_model`'s prediction matches the class the image was intended for — per class and overall.

#### Scenario: Student checks generator output against the classifier, per class
- **WHEN** a student calls `classify_generated(classifier_model, generator_model, data, n=10)` with a trained CNN, a trained VAE, and the labeled `Dataset` the VAE was trained on
- **THEN** they SHALL receive a report of how often each class's intended images were classified as that class, without writing any glue code between the two models

#### Scenario: Intended-class images reflect the dataset's labels
- **WHEN** `classify_generated()` produces images intended for a given class
- **THEN** those images SHALL be drawn from the region of `generator_model`'s latent space that class's labeled images in `data` actually occupy, not from an unconditioned sample of the whole latent space

#### Scenario: The report shows the images it is about
- **WHEN** `classify_generated()` returns its report
- **THEN** the report SHALL include every generated image in the library's `uint8` image format, together with the class each was intended for and the class the classifier predicted

#### Scenario: A generator without a latent space is rejected clearly
- **WHEN** a student calls `classify_generated()` with a `generator_model` that has no probabilistic latent space to sample class regions from (e.g. a model built by `build_autoencoder()`)
- **THEN** the system SHALL raise the library's capability-error type, naming what the model cannot do, rather than failing inside internals

### Requirement: classify_generated() requires a shared class taxonomy
`classify_generated()` SHALL verify that `classifier_model`, `generator_model`, and `data` all use the same set of class names, and that the two models use the same image shape, before generating or classifying anything. The two models are trained independently and nothing else guarantees they agree. Intended and predicted classes SHALL be compared by class name, so the order in which each model lists its classes does not matter.

#### Scenario: Mismatched class taxonomies are rejected
- **WHEN** a student calls `classify_generated()` with a classifier and a generator built from datasets with different class names
- **THEN** the system SHALL raise a clear error naming the mismatch, rather than producing a misleading report or failing inside internals

#### Scenario: Mismatched image shapes are rejected
- **WHEN** a student calls `classify_generated()` with a classifier and a generator built for different image shapes
- **THEN** the system SHALL raise a clear error naming both shapes, rather than failing inside the classifier

#### Scenario: Matching taxonomies proceed normally
- **WHEN** a student calls `classify_generated()` with a classifier and a generator built from datasets with the same class names, in the same order or not
- **THEN** the call SHALL proceed and produce a report

### Requirement: activation_maximize()
The system SHALL provide an `activation_maximize(model, target_class)` function that, given any trained model with a classification capability and a target class name, produces an image that maximizes that class's predicted score via gradient ascent on pixel values starting from random noise, without the student implementing backpropagation-into-input themselves.

#### Scenario: Student visualizes what the classifier imagines
- **WHEN** a student calls `activation_maximize(classifier_model, target_class="triangle")` on a trained CNN
- **THEN** they SHALL receive a generated image, in the model's image shape and `uint8` dtype, visualizing the classifier's learned representation of that class

#### Scenario: Works on any classifying model, not only the CNN
- **WHEN** a student calls `activation_maximize(model, target_class=...)` with a trained model that has a classification capability, whether from `build_classifier()` or `build_vae()`
- **THEN** the call SHALL succeed using that model's own `classify` capability

#### Scenario: A non-classifying model is rejected clearly
- **WHEN** a student calls `activation_maximize()` with a model that lacks a classification capability (e.g. a `build_autoencoder()` model)
- **THEN** the system SHALL raise the library's capability-error type, naming what the model cannot do

#### Scenario: An unknown target class is rejected clearly
- **WHEN** a student calls `activation_maximize(model, target_class=...)` with a name that is not one of the model's class names
- **THEN** the system SHALL raise a clear error naming the problem, rather than an internal indexing failure

### Requirement: Linkage operates on in-memory models
The linkage functions SHALL accept trained model objects directly (not file paths or serialized checkpoints), so the capstone exercise can run within a single notebook session without requiring a model save/load step.

#### Scenario: Capstone run within one session
- **WHEN** a student has trained a classifier and a generator earlier in the same notebook session
- **THEN** they SHALL be able to pass both model objects directly into `classify_generated()` without saving either to disk
