"""Session-scoped real Dataset Forge fixtures.

For tests whose purpose is validating actual model quality (accuracy bounds,
CPU time budget, `classify_generated()` agreement) rather than plumbing —
picoface-phase6's "stub stays, real dataset is added, not swapped in
wholesale" decision (design.md). Plumbing tests keep using
`picoface._internals.stub_data.make_stub_dataset`, unaffected by this file.

Nothing here is committed as a dataset: `real_dataset` renders a fresh export
via the Forge into a session-scoped `tmp_path_factory` directory, the same
way the Forge's own `dataset_forge.tests.test_smoke` produces one on demand.
"""

import pytest

# Needs the Forge's own requirements; real-dataset tests are skipped cleanly
# without them, same convention as dataset_forge/tests/test_smoke.py.
pytest.importorskip("PIL")

from dataset_forge.config import ForgeConfig  # noqa: E402
from dataset_forge.export import DEFAULT_CONFIG, export  # noqa: E402
from picoface.datasets import Dataset, load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def real_dataset(tmp_path_factory) -> tuple[Dataset, Dataset]:
    """The shipped default Dataset Forge export (seed 0): (train, test)."""
    config = ForgeConfig.load(DEFAULT_CONFIG)
    out_dir = export(config, seed=0, out_dir=tmp_path_factory.mktemp("real_export"))
    return load_dataset(out_dir / "train.npz"), load_dataset(out_dir / "test.npz")
