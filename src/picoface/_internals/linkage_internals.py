"""Internal machinery for the capstone linkage functions.

Not part of the public API — `picoface.linkage` wraps everything here behind
named, student-facing functions. Works only through the abstract `_Model`
capabilities (`classify`, `latent_access`), never a concrete model class.
"""

import numpy as np
import torch

from picoface._internals.model_api import _Model, _preprocess_images, _to_uint8_images

# Activation-maximization gradient ascent (phase4 Decision 4). Provisional;
# Phase 6 owns the values, along with whether the ascent needs regularizing.
ASCENT_STEPS = 200
ASCENT_LEARNING_RATE = 0.05


def _class_latent_clusters(model: _Model, data) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-class mean and standard deviation of `model`'s latent means over `data`'s images."""
    model.eval()
    with torch.no_grad():
        mu = model.encode_mu(_preprocess_images(data.images, model))

    labels = torch.from_numpy(np.asarray(data.labels)).long()
    clusters = {}
    for index, name in enumerate(data.class_names):
        class_mu = mu[labels == index]
        if len(class_mu) == 0:
            raise ValueError(
                f"data has no images of class {name!r}, so there is nothing to locate "
                "that class in the generator's latent space from."
            )
        clusters[name] = (class_mu.mean(dim=0), class_mu.std(dim=0, correction=0))
    return clusters


def _sample_near_clusters(
    model: _Model, clusters: dict[str, tuple[torch.Tensor, torch.Tensor]], n: int
) -> tuple[np.ndarray, list[str]]:
    """`n` decoded images per class, drawn around each class's latent cluster."""
    latents = []
    intended = []
    for name, (mean, std) in clusters.items():
        latents.append(mean + std * torch.randn(n, mean.shape[0]))
        intended.extend([name] * n)

    model.eval()
    with torch.no_grad():
        images = _to_uint8_images(model.decode(torch.cat(latents)))
    return images, intended


def _predicted_class_names(model: _Model, images: np.ndarray) -> list[str]:
    model.eval()
    with torch.no_grad():
        logits = model.classify(_preprocess_images(images, model))
    return [model.class_names[i] for i in logits.argmax(dim=1).tolist()]


def _agreement(
    intended: list[str], predicted: list[str], class_names: list[str]
) -> tuple[dict[str, float], float]:
    """Fraction of images predicted as their intended class: per class, and overall."""
    per_class = {}
    for name in class_names:
        hits = [p == name for i, p in zip(intended, predicted) if i == name]
        per_class[name] = sum(hits) / len(hits)
    overall = sum(i == p for i, p in zip(intended, predicted)) / len(intended)
    return per_class, overall


def _ascend(model: _Model, class_index: int, pixels: torch.Tensor) -> torch.Tensor:
    """Gradient ascent on `pixels` (NCHW in [0, 1]) to raise `class_index`'s logit."""
    model.eval()
    pixels = pixels.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pixels], lr=ASCENT_LEARNING_RATE)

    for _ in range(ASCENT_STEPS):
        score = model.classify(pixels)[:, class_index].sum()
        # autograd.grad rather than backward(), so the model's own parameter
        # gradients are left untouched.
        (grad,) = torch.autograd.grad(-score, pixels)
        pixels.grad = grad
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(0.0, 1.0)

    return pixels.detach()
