"""Internal machinery for the capstone linkage functions.

Not part of the public API — `picoface.linkage` wraps everything here behind
named, student-facing functions. Works only through the abstract `_Model`
capabilities (`classify`, `latent_access`), never a concrete model class.
"""

import numpy as np
import torch
import torch.nn.functional as F

from picoface._internals.model_api import _Model, _preprocess_images, _to_uint8_images

# Activation-maximization gradient ascent (phase4 Decision 4). Provisional;
# Phase 6 owns the values, along with whether the ascent needs regularizing.
ASCENT_STEPS = 200
ASCENT_LEARNING_RATE = 0.05

# Periodic Gaussian blur on the ascended image (phase6 diagnostics.md): plain
# gradient ascent on pixels converges to high-frequency adversarial noise the
# CNN scores very highly but that carries no visible class structure.
# Blurring every few steps removes that noise while the ascent keeps
# climbing, at the cost of a lower (but still clearly positive) final score.
# `None` disables blurring.
ASCENT_BLUR_EVERY = 10
ASCENT_BLUR_SIGMA = 1.0
_ASCENT_BLUR_KERNEL_SIZE = 5

# Scale factor on each class's per-dimension latent spread when sampling around
# its cluster (phase6 diagnostics.md): 1.0 (the full observed spread) draws
# from the edges of a class's cluster more often than a slightly tighter draw
# does, which costs `classify_generated()` agreement. 0.6-0.8 consistently
# beat 1.0 across 6 sampling seeds; 0.75 is the middle of that range.
CLUSTER_SAMPLE_SPREAD = 0.75


def _gaussian_blur_kernel(channels: int, sigma: float, ksize: int) -> torch.Tensor:
    axis = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, ksize, ksize).repeat(channels, 1, 1, 1)


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
        latents.append(mean + CLUSTER_SAMPLE_SPREAD * std * torch.randn(n, mean.shape[0]))
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
    """Gradient ascent on `pixels` (NCHW in [0, 1]) to raise `class_index`'s logit.

    Periodically blurred (`ASCENT_BLUR_EVERY`) to keep the result visibly
    shaped like the class rather than adversarial noise (phase6 diagnostics.md).
    """
    model.eval()
    channels = pixels.shape[1]
    kernel = (
        _gaussian_blur_kernel(channels, ASCENT_BLUR_SIGMA, _ASCENT_BLUR_KERNEL_SIZE)
        if ASCENT_BLUR_EVERY
        else None
    )
    pixels = pixels.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pixels], lr=ASCENT_LEARNING_RATE)

    for step in range(ASCENT_STEPS):
        score = model.classify(pixels)[:, class_index].sum()
        # autograd.grad rather than backward(), so the model's own parameter
        # gradients are left untouched.
        (grad,) = torch.autograd.grad(-score, pixels)
        pixels.grad = grad
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(0.0, 1.0)
            if kernel is not None and (step + 1) % ASCENT_BLUR_EVERY == 0:
                blurred = F.conv2d(
                    pixels, kernel, padding=_ASCENT_BLUR_KERNEL_SIZE // 2, groups=channels
                )
                pixels.copy_(blurred)

    return pixels.detach()
