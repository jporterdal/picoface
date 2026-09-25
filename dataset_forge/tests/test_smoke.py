import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import numpy as np  # noqa: E402

from dataset_forge.export import export  # noqa: E402
from dataset_forge.smoke import edge_darkening, format_results, smoke, write_figures  # noqa: E402
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

    assert list(results["activation_maximize"]) == ["circle", "ring", "square"]
    for per_model in results["activation_maximize"].values():
        assert set(per_model) == {"cnn", "vae"}
        for r in per_model.values():
            assert r["images"].shape == (4, 28, 28, 1)
            assert -255 <= r["edge_darkening"] <= 255

    table = format_results(results)
    assert "| cnn |" in table and "| vae |" in table and "| overall |" in table
    assert "edge darkening" in table
    for name in ("circle", "ring", "square"):
        assert sum(line.startswith(f"| {name} |") for line in table.splitlines()) == 4

    paths = write_figures(results, tmp_path / "figures")
    assert [p.name for p in paths] == [
        "generated.png",
        "reconstructed.png",
        "activation_maximize.png",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_edge_darkening_compares_the_outer_ring_with_the_next_one_in():
    images = np.full((2, 12, 12, 1), 200, np.uint8)
    images[:, :2, :] = images[:, -2:, :] = images[:, :, :2] = images[:, :, -2:] = 150

    assert edge_darkening(images) == pytest.approx(-50)
