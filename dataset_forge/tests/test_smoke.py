import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import numpy as np  # noqa: E402

from dataset_forge.export import export  # noqa: E402
from dataset_forge.smoke import format_results, smoke, write_figures  # noqa: E402
from dataset_forge.tests._configs import tiny_config  # noqa: E402


def test_smoke_check_measures_both_models_on_an_export(tmp_path):
    out_dir = export(tiny_config(), seed=0, out_dir=tmp_path)

    results = smoke(out_dir, n=2, epochs=2)

    assert set(results["models"]) == {"cnn", "vae"}
    for r in results["models"].values():
        assert r["train_seconds"] > 0
        assert 0.0 <= r["test_accuracy"] <= 1.0
        assert r["confusion"].shape == (3, 3)
        assert r["confusion"].sum() == 3 * 4  # every test image, once
        assert np.trace(r["confusion"]) / 12 == pytest.approx(r["test_accuracy"])
    assert list(results["classify_generated"].per_class) == ["circle", "ring", "square"]
    for r in results["reconstruction"].values():
        assert r["reconstructed"].shape == r["real"].shape == (2, 28, 28, 1)
        assert 0.0 <= r["agreement"] <= 1.0

    table = format_results(results)
    assert "| cnn |" in table and "| vae |" in table and "| overall |" in table

    paths = write_figures(results, tmp_path / "figures")
    assert [p.name for p in paths] == ["generated.png", "reconstructed.png"]
    assert all(p.stat().st_size > 0 for p in paths)
