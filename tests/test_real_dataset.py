"""Real-dataset counterparts of model-quality tests (picoface-phase6, tasks 1.1-1.2).

Stub-based tests that validate model *quality* rather than plumbing, in scope
for a real-dataset counterpart per design.md's "stub stays, real dataset is
added" decision, and the requirements they correspond to in this change's
`specs/shape-classifier/spec.md` and `specs/shape-generator/spec.md`:

- `tests/test_classifier.py::test_training_wall_clock_under_generous_ceiling`
  -> `shape-classifier`'s "CPU training time budget" requirement.
- `tests/test_generator.py::test_vae_training_wall_clock_under_generous_ceiling`
  -> `shape-generator`'s "CPU training time budget" requirement.
- `tests/test_linkage.py::test_classify_generated_reports_every_class_from_in_memory_models`
  -> `capstone-linkage`'s existing structural requirements on
  `classify_generated()`, exercised here against real content instead of the
  stub, per design.md's Goals (verify the two binding constraints hold
  against real data).

All three keep their original stub-based test untouched (fast plumbing
coverage) and gain the real-dataset test below alongside it.
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from picoface.classifier import build_classifier
from picoface.classifier import evaluate as classifier_evaluate
from picoface.classifier import train as classifier_train
from picoface.generator import build_vae
from picoface.generator import train as generator_train
from picoface.linkage import classify_generated

_CPU_TIME_BUDGET_SECONDS = 300.0


def test_real_dataset_fixture_loads_expected_shape_and_classes(real_dataset):
    train_data, test_data = real_dataset

    assert train_data.images.shape == (14000, 28, 28, 1)
    assert test_data.images.shape == (1400, 28, 28, 1)
    assert len(train_data.class_names) == len(test_data.class_names) == 7
    assert train_data.class_names == test_data.class_names


def test_classifier_training_completes_within_budget_on_real_dataset(real_dataset):
    train_data, _test_data = real_dataset
    model = build_classifier(train_data)

    history = classifier_train(model, train_data)

    assert history.wall_clock_seconds < _CPU_TIME_BUDGET_SECONDS


def test_vae_training_completes_within_budget_on_real_dataset(real_dataset):
    train_data, _test_data = real_dataset
    model = build_vae(train_data)

    history = generator_train(model, train_data)

    assert history.wall_clock_seconds < _CPU_TIME_BUDGET_SECONDS


def test_classify_generated_reports_every_class_on_real_dataset(real_dataset):
    train_data, _test_data = real_dataset
    torch.manual_seed(0)
    cnn = build_classifier(train_data)
    classifier_train(cnn, train_data)
    vae = build_vae(train_data)
    generator_train(vae, train_data)

    report = classify_generated(cnn, vae, train_data, n=10)

    assert list(report.per_class) == train_data.class_names
    assert all(0.0 <= rate <= 1.0 for rate in report.per_class.values())
    assert 0.0 <= report.overall <= 1.0
    assert report.images.shape == (10 * len(train_data.class_names), 28, 28, 1)
    assert report.images.dtype == np.uint8
    assert set(report.predicted) <= set(train_data.class_names)
