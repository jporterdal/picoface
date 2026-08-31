## 1. Internals: model and training loop

- [ ] 1.1 Create `src/picoface/_internals/classifier_internals.py` with a small CNN `nn.Module` (two conv→ReLU→maxpool blocks sized from `input_shape`, FC head to `num_classes`)
- [ ] 1.2 Implement an internal training-loop function (DataLoader batching, cross-entropy loss, Adam optimizer, CPU device, epoch iteration, per-epoch loss + wall-clock timing)
- [ ] 1.3 Implement a shared internal `_forward(model, images) -> logits` helper used by both evaluation and prediction paths
- [ ] 1.4 Add a `TrainingHistory` dataclass (per-epoch `loss` list, `wall_clock_seconds`)

## 2. Public API: `src/picoface/classifier.py`

- [ ] 2.1 Implement `build_classifier(num_classes, input_shape=None)` returning a `PicoClassifier` model handle
- [ ] 2.2 Implement `train(model, data, epochs=10, batch_size=16, learning_rate=1e-3) -> TrainingHistory`
- [ ] 2.3 Implement `evaluate(model, data) -> float` (accuracy)
- [ ] 2.4 Implement `predict(model, image) -> str` (class name, resolved via the model's associated class-name list)
- [ ] 2.5 Verify no `nn.Module` subclasses, loss functions, or training-loop code are importable from `picoface.classifier` itself

## 3. Visualization

- [ ] 3.1 Add `plot_training_history(history)` to `src/picoface/viz.py` (loss-vs-epoch line chart)

## 4. Tests

- [ ] 4.1 Add `tests/test_classifier.py`: build → train → evaluate against `make_stub_dataset()`, asserting the API runs end to end and accuracy is a valid probability
- [ ] 4.2 Add a `predict()` test: single stub image in, class name (member of `class_names`) out
- [ ] 4.3 Add a training-time regression test: assert `TrainingHistory.wall_clock_seconds` stays under a generous ceiling (e.g. 60s) on the default-sized stub dataset
- [ ] 4.4 Add a shape-agnosticism test: build/train against two differently-shaped stub datasets (varying `input_shape` and `num_classes`) to confirm no dimension is hardcoded
- [ ] 4.5 Add a `plot_training_history()` smoke test (runs without error, e.g. using a non-interactive matplotlib backend)

## 5. Verification

- [ ] 5.1 Run the full test suite (`pytest`) and confirm it passes on a CPU-only environment
- [ ] 5.2 Manually record observed wall-clock training time against the stub dataset, per the roadmap's "measure from Phase 2 onward" mitigation
