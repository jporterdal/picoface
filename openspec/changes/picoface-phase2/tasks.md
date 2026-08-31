## 1. Internals: model and training loop

- [x] 1.1 Create `src/picoface/_internals/classifier_internals.py` with a small CNN `nn.Module` (two conv→ReLU→maxpool blocks, `padding=1` on each conv as the starting default, kernel/channel counts as tuned constants; FC head to `num_classes`)
- [x] 1.2 Implement `_build_classifier(num_classes, input_shape)`: computes the fixed architecture's minimum viable spatial size analytically from the same padding/kernel/stride constants used to build the layers, raising `ShapeError` naming that minimum if `input_shape`'s H or W is at or below it; otherwise builds the model and determines the FC head's flatten dimension via a single dummy pass-through (`torch.zeros(1, C, H, W)`)
- [x] 1.3 Implement an internal training-loop function (DataLoader batching over raw dataset arrays, cross-entropy loss, Adam optimizer, CPU device, epoch iteration, per-epoch loss + wall-clock timing)
- [x] 1.4 Implement a shared internal `_forward(model, images) -> logits` helper: validates the incoming image shape against `model.input_shape` (raising `ShapeError` on mismatch), converts NHWC uint8 images to NCHW float32 (permute + cast), normalizes pixel values to [0, 1] (`/255.0`), then runs the forward pass — used by `train()`'s per-batch loop, `evaluate()`, and `predict()` so preprocessing and shape validation never diverge between them
- [x] 1.5 Add a `TrainingHistory` dataclass (per-epoch `loss` list, `wall_clock_seconds`)

## 2. Public API: `src/picoface/classifier.py`

- [x] 2.1 Define `ShapeError(ValueError)`: project-specific exception for all shape/class-count validation failures, with per-call-site messages tailored for a first-year student (e.g. naming the minimum viable input size, or naming which of shape/class-count is mismatched)
- [x] 2.2 Implement `build_classifier(data)`: derives `num_classes`/`input_shape` from the given `Dataset` and delegates to `_build_classifier`
- [x] 2.3 Implement `build_classifier_from_shape(num_classes, input_shape)`: delegates directly to `_build_classifier`
- [x] 2.4 Implement `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3) -> TrainingHistory`: validates `len(data.class_names) == model.num_classes` once before the loop starts (raising `ShapeError` on mismatch); per-batch image-shape validation happens automatically inside `_forward`
- [x] 2.5 Implement `evaluate(model, data) -> float` (accuracy), with the same once-per-call class-count check as `train()`
- [x] 2.6 Implement `predict(model, image) -> str` (class name, resolved via the model's associated class-name list) — no class-count check applies here (no incoming class count to compare against); image-shape validation happens via `_forward`
- [x] 2.7 Verify no `nn.Module` subclasses, loss functions, or training-loop code are importable from `picoface.classifier` itself

## 3. Visualization

- [x] 3.1 Add `plot_training_history(history)` to `src/picoface/viz.py` (loss-vs-epoch line chart)

## 4. Tests

- [x] 4.1 Add `tests/test_classifier.py`: build (via `build_classifier(data)`) → train → evaluate against `make_stub_dataset()`, asserting the API runs end to end and accuracy is a valid probability
- [x] 4.2 Add a `build_classifier_from_shape()` test: build without a `Dataset`, train/evaluate against a separately-constructed matching `Dataset`, confirming it behaves equivalently to `build_classifier(data)`
- [x] 4.3 Add a `predict()` test: single stub image in, class name (member of `class_names`) out
- [x] 4.4 Add a training-time regression test: assert `TrainingHistory.wall_clock_seconds` stays under a generous ceiling (e.g. 60s) on the default-sized stub dataset
- [x] 4.5 Add a shape-agnosticism test: build/train against two differently-shaped stub datasets (varying `input_shape` and `num_classes`) to confirm no dimension is hardcoded
- [x] 4.6 Add a minimum-input-size test: confirm `build_classifier_from_shape()` raises `ShapeError` for an `input_shape` at or below the architecture's analytically-computed floor
- [x] 4.7 Add a class-count mismatch test: confirm `train()`/`evaluate()` raise `ShapeError` when `data.class_names`'s length doesn't match the model's `num_classes`
- [x] 4.8 Add an image-shape mismatch test: confirm `train()`/`evaluate()`/`predict()` raise `ShapeError` when given an image/dataset whose shape doesn't match the model's `input_shape`
- [x] 4.9 Add a `plot_training_history()` smoke test (runs without error, e.g. using a non-interactive matplotlib backend)

## 5. Verification

- [x] 5.1 Run the full test suite (`pytest`) and confirm it passes on a CPU-only environment
- [x] 5.2 Manually record observed wall-clock training time against the stub dataset, per the roadmap's "measure from Phase 2 onward" mitigation
