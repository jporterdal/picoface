## 1. Prerequisites

- [ ] 1.1 Make sure `dataset-validation` and `mediacomp-bridge` are both committed and archived before editing anything. This change builds on `mediacomp-bridge`: its picture conversion in `predict()`, its `save_images()`, and its `activation_maximize()` diagnostic in `smoke.py`. Compare `specs/mediacomp-bridge/spec.md` in this change with the "Saving picoface images for mediaComp" requirement in `openspec/specs/mediacomp-bridge/spec.md`, and re-copy the requirement text if its wording changed during implementation, keeping only this change's signature and `verbose` sentence. Verify: `openspec list` shows neither change as active; `git status` is clean; the full `pytest` suite passes before any edit; `openspec validate progress-output --strict` passes.

## 2. Formatting and printing helpers

- [ ] 2.1 Create `src/picoface/_internals/progress.py` with these helpers:
  - `say(verbose, *lines)`, which writes to `sys.stdout`, looked up at call time, with `flush=True`;
  - formatters for counts (`7,000`), percentages (`92.1%`), durations (`42s`, `3m 05s`, `1h 02m`), and image descriptions (`28x28 grayscale`, `28x28 colour (RGB)`);
  - the metric label and format table from design.md Decision 6, falling back to the field name for unknown fields;
  - a function that lays out `label value` pairs `|`-separated, wrapped at about 78 characters onto indented lines.

  Verify: unit tests in `tests/test_progress_output.py` cover:
  - each duration band;
  - thousands separators;
  - the fallback label;
  - wrapping;
  - `say(False, ...)` printing nothing;
  - all formatter output being ASCII.
- [ ] 2.2 Add the training reporter `_TrainingProgress` and its silent counterpart, which share the methods `start`, `batch_done`, `epoch_done`, `finish` and `close` (design.md Decisions 3–5). The reporter needs:
  - an injectable `clock`;
  - a `\r` redraw padded to the previous line's length;
  - the redraw throttle (0.25 s, 10% boundaries, and the last batch);
  - an overall-pace estimate of time remaining, showing "estimating..." until the larger of 1% of batches or 5 batches has been measured;
  - `close()`, which ends an open bar line.

  Verify: with a fake clock and 438 batches per epoch, at least 10 draws happen per epoch, and each draw starts with `\r`. No time-remaining estimate appears before the warm-up threshold, and a correct one appears after it. `close()` after a partial epoch leaves output ending in a newline.

## 3. Model metadata

- [ ] 3.1 Add class attributes to `_Model`: `display_name` (default "picoface model") and `training_notes: tuple[str, ...] = ()`. Set `display_name` on `_CNNClassifier` ("CNN classifier"), `_Autoencoder` ("autoencoder") and `_VAE` ("VAE (variational autoencoder)"). Give `_VAE` the two notes from design.md Decision 6: the negative total loss, and the KL weight rising over the first half of training. Check the "first half" wording against `_kl_weight()`. Verify: a test checks each builder's model has a non-default `display_name`, and that only the VAE has notes.

## 4. train() and TrainingHistory

- [ ] 4.1 Add `verbose: bool = True` to `train()` and wire in the reporter. Specifically:
  - print the header after all input checks pass;
  - call `batch_done()` per batch and `epoch_done()` per epoch, with the same averaged metrics that go into the history;
  - call `finish()` at the end;
  - put the loop in `try/finally` with `close()`.

  Update the docstring to mention `verbose`. Verify, on the stub dataset:
  - the header states the model's `display_name`, the image count, the batches per epoch, the batch size, the learning rate, and the weight count, and a VAE's header includes its notes;
  - each epoch prints a line with every non-empty history field's label, with "training accuracy" for `accuracy`;
  - the summary includes the total time, the final values, and pointers to `print(history)` and `plot_training_history`;
  - a model whose `training_step` raises on batch 3 leaves stdout ending in a newline.
- [ ] 4.2 Test that output never changes training (spec: Output never changes results). Seed, build, and train a CNN and a VAE twice each, once with `verbose=True` and once with `verbose=False`. Verify: the `state_dict` tensors are equal and the history lists are equal. `verbose=False` writes nothing to stdout or stderr.
- [ ] 4.3 Implement `TrainingHistory.__repr__` and `__str__` per design.md Decision 8, leaving out empty lists. Verify:
  - `repr` of a 2-epoch and of a 30-epoch history are about the same length (bounded, independent of the epoch count), and name every non-empty field and its final value;
  - `str` has one row per epoch and one column per non-empty field;
  - a CNN history's `repr` and `str` never mention `kl_loss`, `reconstruction_loss` or the log-variance fields;
  - `plot_training_history()` still works, as its existing tests show.

## 5. Summaries for the fast calls

- [ ] 5.1 Add `verbose=True` to `load_dataset()` and print the summary described in design.md Decision 7, after the empty-class warning, marking empty classes. Verify:
  - loading a stub bundle prints the path, the count, the size and colour, the class count, and one line per class with its count;
  - a rejected bundle prints nothing before raising;
  - `verbose=False` prints nothing, and still warns for an empty class.
- [ ] 5.2 Add `verbose=True` to `build_classifier()`, `build_classifier_from_shape()`, `build_autoencoder()` and `build_vae()`, and print the build summary: `display_name`, the input shape, the classes (for classifying models), the trainable weight count, and the "untrained, use train(model, data)" line. `build_classifier_from_shape()` notes that its class names are placeholders until `train()`. Verify: each builder's output contains its weight count, which equals `sum(p.numel() for p in model.parameters() if p.requires_grad)`. `verbose=False` is silent.
- [ ] 5.3 Add `verbose=True` to `evaluate()` and `predict()`:
  - `evaluate()` prints the image count, the overall accuracy with its correct count, and each class's correct/total count and percentage;
  - `predict()` prints the predicted class with its probability, and every other class sorted by probability.

  The return values are unchanged. Verify:
  - `evaluate()`'s per-class correct counts sum to the overall correct count;
  - `predict()`'s printed class equals its return value, and its printed probabilities sum to about 100%;
  - `verbose=False` is silent.
- [ ] 5.4 Add `verbose=True` to `generate()`, printing the count, the size and colour, and the array shape. Verify: under a fixed seed, `generate()` returns equal arrays with `verbose` on and off, and its output states the returned array's shape.
- [ ] 5.5 Add `verbose=True` to `classify_generated()` and `activation_maximize()`:
  - `classify_generated()` prints its plan line before generating. Afterwards it prints each class's agreement with its count and what the other images were called, then the overall agreement;
  - `activation_maximize()` prints the target class, the step count, and the score and probability before and after, computed with a no-grad forward pass after the random start image is drawn.

  Verify:
  - under a fixed seed, both functions return equal results with `verbose` on and off;
  - the printed per-class agreement matches `report.per_class`;
  - the "after" probability printed by `activation_maximize()` is greater than or equal to the "before" probability on a trained stub classifier.

- [ ] 5.6 Add `verbose=True` to `save_images()` in `picoface.pictures`, printing one line after every file is written: the image count, the saved size, and the folder (design.md Decision 7). Leave `picture_to_array()`, `crop_and_center()` and `scale_down()` unchanged, with no `verbose` argument. Verify:
  - saving 8 images prints one line stating 8 and the folder;
  - the returned paths and the written files are the same with `verbose` on and off;
  - `verbose=False` is silent;
  - the other three `pictures` functions print nothing.

## 6. Cross-cutting checks

- [ ] 6.1 Add a walkthrough test that runs the README Quickstart sequence on the stub dataset with 1-epoch training and default output, followed by `save_images()` of the generated images into a temporary folder. Verify:
  - every captured stdout character is printable ASCII, `\n` or `\r`;
  - stderr is empty (the stub dataset has no empty classes);
  - every function listed in the spec printed at least one line.

  Then re-run it with `verbose=False` everywhere, and verify that stdout is empty.
- [ ] 6.2 Update `dataset_forge/smoke.py` so that `confusion_matrix()` and `reconstruction_report()` call `predict(..., verbose=False)`, and the `activation_maximize()` diagnostic from `mediacomp-bridge` calls `activation_maximize(..., verbose=False)`. Leave its other calls printing. Verify: the Forge test suite passes, and a smoke run on an export with `--figures` prints the training progress without one line per prediction or per `activation_maximize()` start.

## 7. Documentation

- [ ] 7.1 Update the README:
  - in the Quickstart, add a short sample of what `train()` prints, and one sentence on `verbose=False`, with `predict()` inside a loop as the example;
  - mention that `print(history)` shows a per-epoch table.

  Leave the "roughly 10 and 20 seconds" timing claim for the speed change, but don't contradict it. Verify: the sample output is copied from a real run of the implementation, and the user reviews the README diff.
- [ ] 7.2 Add a "Potential future change" entry to `openspec/ROADMAP.md` for training speed. Record:
  - the exploration measurements: 67 s for the CNN and 179 s for the VAE at defaults on an 8-thread machine, which is 6.6 s and 18 s per epoch, with 438 batches of 16 on 7,000 images;
  - that the README's 10 s / 20 s claim is stale.

  Verify: the user reviews the ROADMAP diff.

## 8. Manual check and final validation

- [ ] 8.1 Manual check in Thonny on Windows: run the Quickstart with a real export. Verify:
  - the in-epoch progress bar redraws in place on one line, not a new line per update;
  - nothing appears in red;
  - the time remaining counts down sensibly.

  Record the Thonny version and the result in this change's directory, in `manual-check.md`. If `\r` isn't honoured, stop and revise the spec scenario "In-place updates" and design.md Decision 4 before switching to the 25% checkpoint fallback.
- [ ] 8.2 Run the full `pytest` suite (Forge included) and `openspec validate progress-output --strict`. Verify: both pass.
