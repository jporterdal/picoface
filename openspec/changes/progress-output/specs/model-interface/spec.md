## MODIFIED Requirements

### Requirement: Model-agnostic train()
The system SHALL provide a single `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3, verbose=True)` function that trains any model built by this library's `build_*` functions on a `Dataset` and returns a training-history object. The loss being optimized SHALL be determined by the model itself, not by which module `train` was imported from or by the caller. `train()` SHALL make the model aware of its progress through training — the current epoch relative to the caller's requested `epochs` — so that a model can schedule its own loss terms without `train()` knowing what kind of model it is training. For the same reason, the model SHALL determine its own optimizer settings per parameter, relative to the caller's `learning_rate`, and SHALL be able to enforce constraints on its parameters after each optimizer step. `verbose` controls only what `train()` prints (Requirement: Output is on by default and can be turned off per call, in `progress-output`); it SHALL NOT affect training.

#### Scenario: Same call trains different model kinds
- **WHEN** a student calls `train(model, data)` with, in turn, a model from `build_classifier()`, `build_autoencoder()`, and `build_vae()`
- **THEN** each call SHALL run to completion and return a training history, with no change to the call and no student-written training loop

#### Scenario: Each model is trained against its own objective
- **WHEN** `train()` is run on a classifier-only model, on an autoencoder model, and on a VAE model
- **THEN** the classifier's training SHALL optimize classification loss only, the autoencoder's reconstruction loss only, and the VAE's reconstruction, KL-divergence, and classification loss together, without the caller specifying any of them

#### Scenario: Loss schedules follow the requested epoch count
- **WHEN** the same model kind is trained once with a small `epochs` value and once with a large one
- **THEN** any loss schedule the model applies SHALL complete within each run, rather than being defined in absolute epochs

#### Scenario: Learning rate stays a single student-facing value
- **WHEN** a student calls `train()` with a `learning_rate` on a model that trains some of its parameters at a different rate
- **THEN** those rates SHALL be set by the model relative to the given `learning_rate`, and the student SHALL NOT need to pass any per-parameter setting

#### Scenario: Training a non-model is rejected clearly
- **WHEN** a student calls `train()` with an object that was not built by this library
- **THEN** the system SHALL raise a clear error naming the problem, rather than an internal attribute or framework failure

### Requirement: Shared training history
`train()` SHALL return one history type for all model kinds, containing per-epoch total `loss` and the total `wall_clock_seconds`, plus any additional per-epoch metrics the model reports (reconstruction loss, KL-divergence, classification loss, accuracy, the KL weight applied, and learned task-weighting values). Metrics a model does not report SHALL be present as empty lists, not absent.

The history's `repr`, which a Python shell echoes after `train()` returns, SHALL be short: at most a few lines, whatever the number of epochs. It SHALL still state:
- the number of epochs and the total training time;
- the final epoch's value of every metric the history records;
- the names of the fields that hold the per-epoch lists.

Converting the history to a string (as `print(history)` does) SHALL give a table with one row per epoch and one column per recorded metric. Metrics the model does not report SHALL be left out of both forms.

#### Scenario: History fields for a classifier-only model
- **WHEN** `train()` completes on a `build_classifier()` model
- **THEN** the history SHALL have a per-epoch `loss` and `wall_clock_seconds`, and its reconstruction and KL metric lists SHALL be empty

#### Scenario: History fields for an autoencoder
- **WHEN** `train()` completes on a `build_autoencoder()` model
- **THEN** the history SHALL have per-epoch `loss` and `reconstruction_loss`, and its KL, classification, accuracy, KL-weight, and task-weighting lists SHALL be empty

#### Scenario: History fields for a joint VAE
- **WHEN** `train()` completes on a `build_vae()` model
- **THEN** the history SHALL have per-epoch `loss`, `reconstruction_loss`, `kl_loss`, `classification_loss`, `accuracy`, KL weight, and learned task-weighting values, each with one entry per epoch

#### Scenario: Shell echo is short but complete
- **WHEN** a student calls `train(model, data)` at an interactive prompt without assigning the result, on a `build_vae()` model trained for 10 epochs
- **THEN** the echoed `repr` SHALL fit on a few lines and SHALL state 10 epochs, the training time, the final value of each of the model's recorded metrics, and the names of the per-epoch fields

#### Scenario: Printing the history shows every epoch
- **WHEN** a student calls `print(history)` on a history from a 10-epoch run
- **THEN** the output SHALL be a table with 10 rows, one per epoch, and a column for each metric the model recorded

#### Scenario: Unreported metrics are left out of both forms
- **WHEN** a student prints, or has the shell echo, the history of a `build_classifier()` model
- **THEN** neither form SHALL mention reconstruction, KL, or task-weighting metrics
