## Context

See `proposal.md` for the motivation, and `specs/progress-output/spec.md` for what must be printed. The current state is as follows.

- **No output today.** No public function prints anything. `load_dataset()` issues a `DatasetWarning` (via `warnings`, so to stderr) for an empty class. Nothing else speaks.
- **Measured timings.** From the exploration run, on the default seed-0 export (7,000 train images, 28×28×1, 7 classes) on an 8-thread WSL machine:

  | Call | Time |
  |---|---|
  | `load_dataset` | 0.02 s |
  | `build_*` | < 0.1 s |
  | `train` on the CNN | 67 s: 6.6 s per epoch, 438 batches of 16 per epoch |
  | `train` on the VAE | 179 s: 18 s per epoch |
  | `evaluate` | 0.1–0.4 s |
  | `generate(8)` | 0.07 s |
  | `classify_generated` | 0.2 s |
  | `activation_maximize` | 0.4 s |

  Epochs are uniform in cost, so a time-remaining estimate based on batch pace is reliable.
- **The VAE's metrics.** The VAE records eight per-epoch metrics. On that run, their values across the 10 epochs were:
  - **total `loss`:** negative, from −0.02 at epoch 1 down to −2.30 at epoch 10;
  - **`kl_loss`:** falls to 0.239 by epoch 6, then rises back to 0.273 as `kl_weight` reaches 1.0, which happens halfway through training;
  - **`accuracy`:** mean training-batch accuracy, not held-out accuracy.

  The CNN records only `loss` and `accuracy`, and the autoencoder only `loss` and `reconstruction_loss`.
- **`train()`'s loop.** It lives in `_internals/model_api.py`. It already accumulates each epoch's metrics into a dict whose keys are `TrainingHistory` field names.
- **Model hooks.** Model classes already carry class-level metadata (`built_by`, `capabilities`) and hooks (`training_step`, `on_epoch_start`, `optimizer_param_groups`, `on_step_end`).
- **The students' environment.** Students run picoface in Thonny on Windows. Thonny's shell shows stderr in red. Instructors may also run it in a plain terminal, and `dataset_forge/smoke.py` runs it in batch.
- **`smoke.py`** calls `predict()` once per test image in `confusion_matrix()` and `reconstruction_report()`. That is thousands of calls per smoke run. `mediacomp-bridge`, applied before this change, adds an `activation_maximize()` diagnostic to it: several random starts per class, for both the CNN and the VAE.
- **`picoface.pictures`** (from `mediacomp-bridge`) has four public functions. `picture_to_array()`, `crop_and_center()` and `scale_down()` return a picture or image array at once. `save_images()` writes PNG files and returns their paths.

## Goals / Non-Goals

**Goals:**
- A student is never left looking at a silent shell for more than about a second while `train()` runs.
- Every number the library records is visible somewhere without extra code: in the per-epoch lines, the final summary, the history's `repr`, or `print(history)`. The user has said they will tune the amount of output down later, so the design favours showing things and makes each message easy to find and edit.
- One place owns the wording and formatting, so tuning it later is a single-file job.

**Non-Goals:**
- **Making training faster.** That is a separate change; see Decision 12.
- **A library-wide on/off switch or output levels.** Only a per-call `verbose=True/False` is in scope. A global switch can be added later without breaking callers.
- **Output for `viz.plot_training_history()`.** It returns a figure, and the figure is its output.
- **Progress output in the Dataset Forge CLI's own export and validate steps.**
- **Output for `picture_to_array()`, `crop_and_center()` and `scale_down()`.** They return at once, and students look at the result with mediaComp's `show()`. They take no `verbose` argument. `save_images()` is covered, because the files it writes are otherwise invisible from the shell.

## Decisions

### 1. Plain `print()` to stdout, from one internal module

A new module, `src/picoface/_internals/progress.py`, holds all of it:
- the formatting helpers: counts, percentages, durations, image descriptions, and metric labels;
- a `say(verbose, *lines)` function;
- the training-progress reporter (Decision 3).

Public functions build their message lines through these helpers and never call `print` directly. Every write goes to `sys.stdout`, looked up at call time rather than import time so that Thonny's and pytest's replacement streams are honoured, with `flush=True`.

**Alternatives considered:**

| Alternative | Why not |
|---|---|
| `tqdm` | Adds a dependency. Its default output (`438/438 [00:06<00:00, 66.1it/s]`) is jargon for first-years. |
| `logging` | INFO messages are hidden unless the student configures a handler, so the default would stay silent. |
| A callback parameter | A concept students never asked to learn. It can be layered on later. |

### 2. `verbose: bool = True` as a keyword on each public function

The parameter goes last on every function listed in the spec. Keyword-only isn't enforced, to match the existing plain signatures. Internal library code that composes public functions passes `verbose=False` to the inner call, so each student-facing call prints only its own summary. For example, if a linkage helper ever called `evaluate()`, it would pass `verbose=False`.

### 3. A training-progress reporter object, with a silent variant

`train()` creates `progress = _TrainingProgress(...)` when `verbose` is true, and a silent reporter with the same methods otherwise. This keeps the training loop free of `if verbose:` branches. The reporter's methods are:
- `start(model, data, epochs, batch_size, learning_rate, batches_per_epoch)`: prints the header;
- `batch_done()`: counts the batch and redraws the progress bar when due;
- `epoch_done(epoch, metrics)`: ends the bar line and prints the epoch record;
- `finish(history)`: prints the summary.

The reporter takes an injectable `clock` (default `time.monotonic`) so tests can drive the update throttle deterministically.

`train()` wraps its loop in `try/finally`, calling the reporter's `close()` so that a partly drawn progress line is ended with a newline. That way an exception or a `KeyboardInterrupt` (the student pressing Stop) prints its traceback on a clean line. The reporter only reads values `train()` already computes. It never touches the model or the random generators (spec: Output never changes results).

### 4. Progress bar: redraw in place, throttled

- **What the bar line shows.** Each update writes `\r` followed by the line:
  ```
    Epoch  3/10 [########------------]  175/438 batches | elapsed 43s | about 2m 17s left
  ```
  It is padded with spaces to the previous line's length, so a shorter line fully overwrites a longer one.
- **When it redraws.** It redraws when at least 0.25 s have passed since the last draw, when the batch count crosses a 10% boundary of the epoch, or on the epoch's last batch. That satisfies the spec's "every second and every tenth" floor. On this machine a batch takes about 15 ms, so drawing on every batch would be ~70 writes a second. That's wasted work, and on Windows consoles it causes visible flicker.
- **Why not detect the terminal first.** `sys.stdout.isatty()` is false in Jupyter, which does honour `\r`, so it can't be used to decide whether to redraw in place. The in-place redraw is used everywhere.

  The one known non-honouring shell is IDLE: each update appears as a new line there, which is noisy but still readable. IDLE isn't the students' editor, so that is accepted.

  Whether Thonny's shell honours `\r` is verified by hand (tasks.md). If it doesn't, the fallback is to print a new line at each 25% checkpoint in place of the in-place redraw. That fallback would need the spec scenario "In-place updates" to be revised first.

### 5. Time remaining from overall batch pace

Remaining time is estimated as `remaining_batches × (elapsed / batches_done)`, counted across all epochs, not per epoch. Epochs cost the same, so this is accurate from the first few batches onward. No estimate is shown until the first 1% of batches, or 5 batches, whichever is larger, has been measured, since the very first batches include warm-up. Until then the bar shows "estimating...".

Durations display as `42s` below one minute, `3m 05s` below an hour, and `1h 02m` above that.

### 6. Show every recorded metric, with a label table and model-supplied notes

**Per-epoch output lists every key in that epoch's metrics dict, plus `loss`, in `TrainingHistory` field order.** This is data-driven, so a model gains output simply by recording a metric, and nothing can be accidentally hidden. A table in `progress.py` maps each field to a plain label and a number format:

| Field | Label | Format |
|---|---|---|
| `loss` | loss | 4 significant figures |
| `reconstruction_loss` | reconstruction error | 4 significant figures |
| `kl_loss` | KL divergence | 4 significant figures |
| `classification_loss` | classification loss | 4 significant figures |
| `accuracy` | training accuracy | percentage, 1 decimal place |
| `kl_weight` | KL weight | 3 decimal places |
| `reconstruction_log_var` | reconstruction log-variance (learned task weight) | 4 significant figures |
| `classification_log_var` | classification log-variance (learned task weight) | 4 significant figures |

A field with no label falls back to the field name, so a future metric still prints.

**Per-epoch layout.** The epoch record is one timing line, then the metrics wrapped at about 78 characters onto indented continuation lines, `|`-separated:
```
  Epoch  3/10 done in 18s | elapsed 54s | about 2m 06s left
      loss -1.494 | reconstruction error 0.01003 | KL divergence 0.2940 | classification loss 0.2690
      training accuracy 89.2% | KL weight 0.444 | reconstruction log-variance (learned ...) ...
```

**Model-supplied notes.** A model explains its own quirks through a class attribute on `_Model`: `training_notes: tuple[str, ...] = ()`. These lines are printed in the header. The VAE's notes say:
- its total loss includes learned task weights, so it can go below zero, and lower is still better;
- the KL weight rises from 0 to 1 over the first half of training, so the KL divergence can rise later in the run, and that is expected.

**Model names.** A second class attribute, `display_name` ("CNN classifier", "autoencoder", or "VAE (variational autoencoder)"), names the model in the header and in `build_*` messages.

**Alternative considered:** a per-model hook that chooses which metrics to show. It was rejected because the user asked for more rather than less, and a hook invites hiding. The notes attribute covers explanation without hiding anything.

### 7. Summaries for the fast calls

Each summary is built after the function's work succeeds, so a call that raises prints nothing. What each one computes, all from values the function already has:

| Function | What it prints |
|---|---|
| `load_dataset()` | The path; count; size and colour (`28x28 grayscale` or `28x28 colour (RGB)`); class count; then one line per class: `name  count`, with `(no images)` marking empty classes. It reuses the per-class counts from the empty-class check, and prints after that check's warning. |
| `build_*()` | `display_name`; input shape; classes, if the model classifies; trainable weight count (`sum(p.numel() for p in model.parameters() if p.requires_grad)`); and a closing line saying the model is untrained, "train it with train(model, data)". `build_classifier_from_shape()` says its class names are placeholders until `train()`. |
| `evaluate()` | Image count, overall accuracy and correct count, then per class: `name  correct/total  (pct)`. Everything comes from the predictions it already makes. |
| `predict()` | `Predicted 'circle' (92.1%)`, then every other class sorted by probability on one line: `others: ring 6.0%, square 1.2%, ...`. It applies softmax to the logits it already computes. |
| `generate()` | Count, image size and colour, and the returned array's shape. |
| `classify_generated()` | A line before generating (`Generating 10 images for each of 7 classes (70 images) ...`). Afterwards, per class: agreement, count, and what the disagreeing images were called instead (`star  6/10 (60.0%)  others called: smiley 3, circle 1`). Then the overall agreement. It uses only the report's own `intended` and `predicted` lists. |
| `activation_maximize()` | The target class, the step count, and the score and softmax probability before and after. `_ascend()` returns the final pixels as now, and the public function computes both scores with a no-grad forward pass. That consumes no random numbers and happens after the random start image is drawn. |
| `save_images()` | One line: the image count, the saved size (`224x224`), and the folder, e.g. `Saved 8 images (224x224 PNG) to generated/`. It prints after every file is written. |

### 8. `TrainingHistory.__repr__` short, `__str__` a full table

**The repr** is one logical line, wrapped by the shell, with a fixed size whatever the epoch count. The field names are kept in it because they are how students access the data.
```
TrainingHistory(10 epochs, 3m 00s; final loss=-2.298, reconstruction_loss=0.005986, kl_loss=0.2730,
classification_loss=0.06997, accuracy=0.9750, kl_weight=1.000, reconstruction_log_var=..., classification_log_var=...;
per-epoch lists: .loss, .reconstruction_loss, ...; print() it for a per-epoch table)
```

**The str** is a header line with the epoch count and time, then a table with an `epoch` column and one column per recorded, non-empty metric. The column headers are the field names.

Both forms leave out empty lists (spec: Unreported metrics are left out). The class stays a plain `@dataclass`; defining `__repr__` in the class body overrides the generated one, and equality and fields are unchanged. `viz.plot_training_history()` is unaffected.

### 9. ASCII-only output

The progress bar uses `#` and `-`, image sizes use `x` rather than `×`, and separators use `|`. On Windows, a redirected stdout uses the locale code page, and a non-encodable character there raises `UnicodeEncodeError`. That would turn a progress message into a crash. A test asserts that the whole walkthrough's captured output is ASCII.

### 10. `dataset_forge/smoke.py` silences only its per-image loops

`confusion_matrix()` and `reconstruction_report()` pass `verbose=False` to `predict()`, and the `activation_maximize()` diagnostic from `mediacomp-bridge` passes `verbose=False` to `activation_maximize()`. The other calls keep their output: `load_dataset`, `build_*`, `train`, `evaluate` and `classify_generated`. A smoke run takes minutes, so the output doubles as the instructor's heartbeat. It is printed before `format_results()`'s final table, so nothing that `smoke.py` prints today changes.

### 11. Tests

A new test module, `tests/test_progress_output.py`, uses the internal stub dataset and short runs (1–2 epochs), and checks the following:
- For each function listed in the spec, default output is non-empty and `verbose=False` output is empty. This uses `capsys` and also checks stderr.
- **Determinism.** With the same seed, `train()` with and without output gives equal `state_dict`s and equal histories. `generate()`, `activation_maximize()` and `classify_generated()` give equal arrays under a fixed seed.
- The header contains the batches-per-epoch count and the VAE's negative-loss note.
- The per-epoch output contains a label for every non-empty history field, and "training accuracy".
- With a fake clock, the reporter draws at least 10 in-place updates per epoch, and each update starts with `\r`.
- The whole walkthrough's output is ASCII.
- `repr(history)` is short (a bounded length, independent of the epoch count) and names every non-empty field. `str(history)` has one row per epoch.
- A failing `load_dataset()` prints nothing.
- A `train()` that raises partway through leaves no unterminated line. To test this, a model whose `training_step` raises on the third batch is used.

The existing tests need no change: pytest captures stdout.

### 12. The training speed finding goes to the ROADMAP, not this change

The README's "roughly 10 and 20 seconds" claim doesn't match the measured 67 s and 179 s. This change doesn't touch the claim or training speed. Its one action is a ROADMAP "potential future change" entry with the measurements, so the finding isn't lost.

## Risks / Trade-offs

- **[Thonny's shell may not honour `\r`, so each redraw becomes a new line]** → A manual check in Thonny on Windows is part of tasks.md. The fallback of checkpoint lines at every 25% is pre-planned in Decision 4, and would need one spec scenario revised.
- **[The amount of output may overwhelm students, e.g. a VAE's eight metrics on every epoch]** → This is intentional for now, at the user's direction. All wording and the label table live in `progress.py`, so trimming later is a one-file edit that needs no spec change for anything the spec doesn't require.
- **[`predict()` in a student's own loop floods the shell]** → `verbose=False` is available, and the README's mention of `verbose` uses exactly this case as its example.
- **[Printing slows training]** → Draws are throttled to at most 4 per second, and epoch lines are printed once per epoch, so the overhead is negligible next to 15 ms batches.
- **[Dependency on `mediacomp-bridge`]** → That change is applied and archived first (tasks.md 1.1). This change edits `predict()` after its picture conversion, adds `verbose` to its `save_images()`, and silences its smoke diagnostic. Its `mediacomp-bridge` spec delta copies a requirement from that change, so the copy must be refreshed from the archived main spec if the requirement's wording changed during implementation.
- **[`dataset-validation` is committed but not archived]** → Archive it before applying this change (tasks.md 1.1).

## Migration Plan

The change is additive. Existing code keeps working and now prints; to restore the old silence, pass `verbose=False`. There is no data or format migration. To roll back, revert the change.

## Open Questions

- **Exact wording and layout of each message.** The user will tune these by hand after implementation. They are deliberately left as editable strings in one module.
- **Paths that aren't ASCII.** `load_dataset()` prints the file it read, and `save_images()` prints the folder it wrote to. Either path can contain characters outside ASCII, for example a Windows user name in `C:\Users\...`, which conflicts with Decision 9 and the ASCII-only requirement. Options: escape non-ASCII characters in printed paths (Python's `backslashreplace`), print only the file or folder name, or exempt paths the caller supplied from the requirement.
