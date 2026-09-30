import pytest

# Needs the Forge's own requirements; skipped cleanly without them.
pytest.importorskip("PIL")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset_forge.export import export  # noqa: E402
from dataset_forge.smoke import (  # noqa: E402
    active_mask,
    active_units_report,
    edge_darkening,
    format_results,
    reconstruct,
    smoke,
    write_figures,
)
from dataset_forge.tests._configs import tiny_config  # noqa: E402
from picoface.datasets import Dataset  # noqa: E402


class _StandInVAE(torch.nn.Module):
    """Encodes a 1-pixel image x to a known μ and logvar; decodes z to a 1-by-d image of z.

    μ = (0.5, x, 0.1x + 0.45, 0.8x + 0.1), so on images half black and half
    white Var(μ) is (0, 0.25, 0.0025, 0.16). logvar is 0 for the constant
    dimension and log 0.25 for the others.
    """

    input_shape = (1, 1, 1)
    shape_error_cls = ValueError

    def encode(self, x):
        x = x.flatten(1)
        mu = torch.cat([torch.full_like(x, 0.5), x, 0.1 * x + 0.45, 0.8 * x + 0.1], dim=1)
        logvar = torch.log(torch.tensor([1.0, 0.25, 0.25, 0.25])).expand_as(mu)
        return mu, logvar

    def encode_mu(self, x):
        return self.encode(x)[0]

    def decode(self, z):
        return z.reshape(len(z), 1, 1, -1)


def _half_black_half_white() -> Dataset:
    images = np.array([0, 0, 0, 255, 255, 255], np.uint8).reshape(6, 1, 1, 1)
    return Dataset(images, np.array([0, 0, 0, 1, 1, 1]), ["black", "white"])


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

    units = results["active_units"]
    assert units["latent_dim"] == len(units["dimension"]) == len(units["variance_of_mean"])
    assert list(units["active"]) == [0.001, 0.01, 0.1]
    assert set(results["masked_reconstruction"]) == {"inactive_masked", "active_masked"}
    for report in results["masked_reconstruction"].values():
        assert list(report) == ["circle", "ring", "square"]
        assert all(0.0 <= r["agreement"] <= 1.0 for r in report.values())
    assert "| Nominal size | Active > 0.001 | Active > 0.01 | Active > 0.1 |" in table
    assert "| Rank | Dimension | Var(μ) | Mean posterior variance | Mean KL |" in table
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


def test_active_units_report_sorts_dimensions_by_the_variance_of_their_mean():
    report = active_units_report(_StandInVAE(), _half_black_half_white())

    assert report["latent_dim"] == 4
    assert list(report["dimension"]) == [1, 3, 2, 0]
    assert report["variance_of_mean"] == pytest.approx([0.25, 0.16, 0.0025, 0.0])
    assert report["posterior_variance"] == pytest.approx([0.25, 0.25, 0.25, 1.0])
    # KL(N(μ, σ²) ‖ N(0, 1)) = ½(μ² + σ² − 1 − log σ²); constant dimension: μ = 0.5, σ² = 1.
    assert report["kl"][3] == pytest.approx(0.5 * 0.5**2)
    assert report["kl"][0] == pytest.approx(0.5 * (0.5 + 0.25 - 1 - np.log(0.25)))
    assert report["active"] == {0.001: 3, 0.01: 2, 0.1: 2}
    assert list(active_mask(report)) == [False, True, False, True]


def test_masking_sets_left_out_latent_dimensions_to_the_prior_mean():
    vae, data = _StandInVAE(), _half_black_half_white()
    active = active_mask(active_units_report(vae, data))
    unmasked = reconstruct(vae, data.images)

    assert np.array_equal(reconstruct(vae, data.images, np.ones(4, bool)), unmasked)
    inactive_masked = reconstruct(vae, data.images, active)
    active_masked = reconstruct(vae, data.images, ~active)
    assert not inactive_masked[:, :, ~active].any() and not active_masked[:, :, active].any()
    assert np.array_equal(inactive_masked + active_masked, unmasked)
