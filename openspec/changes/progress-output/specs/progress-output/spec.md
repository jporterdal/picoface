## Purpose

Defines what picoface prints while it works, so a student can see what each call did and how far a long-running call has got. Also defines how to turn that output off without changing any results.

## ADDED Requirements

### Requirement: Output is on by default and can be turned off per call
Each of these public functions SHALL accept a keyword argument `verbose` whose default is `True`:
- `load_dataset()`;
- `build_classifier()`, `build_classifier_from_shape()`, `build_autoencoder()` and `build_vae()`;
- `train()`, `evaluate()`, `predict()` and `generate()`;
- `classify_generated()` and `activation_maximize()`;
- `save_images()`, from the `mediacomp-bridge` capability.

With `verbose=True` each SHALL print the messages this capability specifies. With `verbose=False` it SHALL print nothing. Code that does not pass `verbose` SHALL keep working unchanged.

The other functions of the `mediacomp-bridge` capability (`picture_to_array()`, `crop_and_center()` and `scale_down()`) SHALL print nothing and SHALL NOT take a `verbose` argument.

#### Scenario: A default call prints a message
- **WHEN** a student calls any of the listed functions without passing `verbose`
- **THEN** the call SHALL print at least one line to standard output describing what it did

#### Scenario: verbose=False is silent
- **WHEN** a student calls any of the listed functions with `verbose=False`
- **THEN** nothing SHALL be written to standard output or standard error by that call, apart from any existing warning such as `DatasetWarning`

#### Scenario: Existing calls still work
- **WHEN** code written before this capability existed calls any of the listed functions with its original arguments
- **THEN** the call SHALL succeed and return what it returned before

### Requirement: Output never changes results
Printing SHALL NOT change any function's result. With the same inputs, a function SHALL return equal values and raise the same errors whether `verbose` is `True` or `False`. Output SHALL NOT consume random numbers, and SHALL NOT change a model's parameters or its training/evaluation mode compared with a silent call.

#### Scenario: Training results match with and without output
- **WHEN** the same model kind is built and trained twice from the same random seed on the same data, once with `verbose=True` and once with `verbose=False`
- **THEN** the two trained models SHALL have identical parameters, and the two training histories SHALL hold identical per-epoch values

#### Scenario: Errors are unchanged
- **WHEN** a function is called with input it rejects, once with `verbose=True` and once with `verbose=False`
- **THEN** both calls SHALL raise the same error type with the same message

### Requirement: Output format suits a student's Python shell
All output SHALL go to standard output, never standard error, because some student editors show standard error in an alarming colour. Each message SHALL be flushed as it is printed, so it appears while the call is still running. Output SHALL consist only of printable ASCII characters, plus newline and carriage return, so it cannot fail to encode on a Windows console or in a redirected output file. Numbers SHALL be written for a general reader:
- counts of 1,000 or more use thousands separators;
- fractions are shown as percentages;
- durations are shown in seconds, or minutes and seconds.

#### Scenario: ASCII-only output
- **WHEN** a full walkthrough (load, build both models, train both, evaluate, predict, generate, classify_generated, activation_maximize, save_images) runs with default output
- **THEN** every character written SHALL be printable ASCII, a newline, or a carriage return

#### Scenario: Nothing goes to standard error
- **WHEN** the same walkthrough runs on a dataset with no empty classes
- **THEN** nothing SHALL be written to standard error

#### Scenario: Readable numbers
- **WHEN** a dataset of 7,000 images is loaded and a model is trained on it for 65 seconds
- **THEN** the output SHALL write the count as "7,000", and the duration in a form like "1m 05s", not as a raw float of seconds

### Requirement: train() announces the job before starting
Before training begins, `train()` SHALL print a header describing the work ahead. The header SHALL give:
- the kind of model;
- how many images it trains on, the image size and colour, and the number of classes;
- the number of epochs, the batch size, the number of batches per epoch, and the learning rate;
- the number of weights the model will learn.

For a model whose total loss can be negative, the header SHALL also say so and that lower is still better.

#### Scenario: Header describes a classifier job
- **WHEN** a student calls `train(model, data)` on a `build_classifier()` model and a 7,000-image dataset with the defaults
- **THEN** before the first epoch the output SHALL state:
  - that a CNN classifier is being trained;
  - 7,000 images, their size, and the class count;
  - 10 epochs, a batch size of 16, 438 batches per epoch, and the learning rate;
  - the model's weight count

#### Scenario: Header explains a negative loss
- **WHEN** a student calls `train()` on a `build_vae()` model
- **THEN** the header SHALL state that this model's total loss can go below zero and that lower is still better

### Requirement: train() shows progress within each epoch
While an epoch runs, `train()` SHALL show how far through the epoch it is, as batches done out of the batches in the epoch. It SHALL update that display at least once per second of training, and at least once for every tenth of the epoch, whichever comes first. The display SHALL also estimate the time remaining for the whole run. On a display that honours carriage returns, the progress indicator SHALL update in place on one line, not add a new line for each update.

#### Scenario: Progress is visible during the first epoch
- **WHEN** an epoch takes several seconds
- **THEN** output showing the batches completed so far SHALL appear before that epoch finishes

#### Scenario: An estimate is shown during the first epoch
- **WHEN** the first epoch has completed some of its batches
- **THEN** the progress display SHALL include an estimated time remaining for the whole run, based on the pace so far

#### Scenario: In-place updates
- **WHEN** the in-epoch progress indicator updates
- **THEN** each update SHALL begin with a carriage return and rewrite the same line, and the line SHALL be ended with a newline only when the epoch's result is printed

### Requirement: train() reports every recorded metric after each epoch
After each epoch, `train()` SHALL print a lasting record of that epoch, giving:
- the epoch number out of the total;
- the epoch's duration, the total elapsed time, and the estimated time remaining;
- the epoch's value of every metric the model recorded in the training history for that epoch.

Each metric SHALL carry a plain-language label. Accuracy measured on the training batches SHALL be labelled as training accuracy, so it isn't mistaken for held-out accuracy. No metric the history records SHALL be left out of the per-epoch output.

#### Scenario: Classifier epoch line
- **WHEN** an epoch of a `build_classifier()` model finishes
- **THEN** the output for that epoch SHALL show its loss, its epoch time, its elapsed time, and its estimated time remaining

#### Scenario: VAE epoch output includes every metric
- **WHEN** an epoch of a `build_vae()` model finishes
- **THEN** that epoch's output SHALL show every metric the history records for it: total loss, reconstruction error, KL divergence, classification loss, training accuracy, the KL weight, and both learned task weights

#### Scenario: Training accuracy is labelled as such
- **WHEN** a model records accuracy during training
- **THEN** the per-epoch output SHALL label it as training accuracy

### Requirement: train() summarises the finished run
When training finishes, `train()` SHALL print a summary that gives:
- the total training time;
- the final epoch's value of every recorded metric;
- how to see the per-epoch record: printing the returned history, or plotting it with the library's training-history plot helper.

#### Scenario: Summary after training
- **WHEN** `train()` completes
- **THEN** its last output SHALL include the total time, the final value of each recorded metric, and a pointer to printing or plotting the returned history

### Requirement: load_dataset() summarises the dataset
`load_dataset()` SHALL print, after a successful load:
- the file it read;
- the number of images;
- the image height, width, and whether the images are grayscale or colour;
- the number of classes;
- each class's name with its image count.

It SHALL print nothing before raising an error for a bundle it rejects.

#### Scenario: Summary of a loaded bundle
- **WHEN** a student loads a valid bundle of 7,000 28×28 grayscale images in 7 classes of 1,000 images each
- **THEN** the output SHALL name the file and state 7,000 images, 28x28, grayscale, 7 classes, and list each class with a count of 1,000

#### Scenario: No summary for a rejected bundle
- **WHEN** `load_dataset()` raises an error
- **THEN** it SHALL have printed nothing

### Requirement: Building a model summarises the model
Each `build_*` function SHALL print:
- the kind of model built;
- the image shape it accepts;
- for a model that classifies, its classes;
- the number of weights it will learn;
- that the model is untrained until `train()` is called.

#### Scenario: Building a VAE
- **WHEN** a student calls `build_vae(data)`
- **THEN** the output SHALL name the model as a VAE, give its image shape and classes, give its weight count, and say that it needs training with `train()`

### Requirement: evaluate() and predict() explain their answers
`evaluate()` SHALL print:
- the number of images evaluated;
- the overall accuracy as a percentage, with the count correct;
- the accuracy for each class, with that class's count correct out of its image count.

`predict()` SHALL print the predicted class with the model's probability for it, followed by the next most likely classes and their probabilities.

#### Scenario: Evaluation breakdown
- **WHEN** a student calls `evaluate(model, data)`
- **THEN** the output SHALL give the overall accuracy with its correct count, and one line per class with that class's accuracy and correct count

#### Scenario: Prediction with probabilities
- **WHEN** a student calls `predict(model, image)` on a model with at least three classes
- **THEN** the output SHALL name the predicted class with its probability, and at least the next two most likely classes with theirs

### Requirement: Generation and capstone functions summarise their results
The generation and capstone functions SHALL print:
- `generate()`: how many images it made, their size and colour, and the shape of the returned array.
- `classify_generated()`, before generating: how many images it will generate per class and in total, and for how many classes. After judging them: each class's agreement as a percentage with its count, and the overall agreement.
- `activation_maximize()`: the target class, the number of optimisation steps, and the model's score and probability for the target class at the start and at the end.

#### Scenario: Generating images
- **WHEN** a student calls `generate(vae_model, 8)`
- **THEN** the output SHALL state that 8 images were generated, their size, and the returned array's shape

#### Scenario: Capstone agreement report
- **WHEN** a student calls `classify_generated(classifier_model, vae_model, data, n=10)` on 7 classes
- **THEN** the output SHALL state that 70 images are generated (10 per class), then list each class's agreement and the overall agreement

#### Scenario: Activation maximization before and after
- **WHEN** a student calls `activation_maximize(model, "circle")`
- **THEN** the output SHALL name `circle`, the number of steps, and the model's score and probability for `circle` before and after the optimisation

### Requirement: save_images() reports what it saved
`save_images()` SHALL print one line after writing its files, giving the number of images saved and the folder they were written to.

#### Scenario: Saving generated images
- **WHEN** a student calls `save_images(generate(vae_model, 8), "generated")`
- **THEN** the output SHALL state that 8 images were saved, and name the `generated` folder
