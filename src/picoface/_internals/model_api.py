"""Abstract model interface and the model-agnostic verbs built on it.

Not part of the public API — `picoface.classifier` and `picoface.generator`
re-export the verbs defined here (`train`, `evaluate`, `predict`, `generate`).
This module is arm-neutral: it imports nothing from the public arm modules or
`viz`, and neither arm's internals import the other's.

Every model object in the project subclasses `_Model`, which supplies the one
thing the shared training loop cannot know (`training_step`, i.e. the loss) and
declares optional capabilities (`classify`, `sample`, `encode_mu`) that the
verbs check before use.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from picoface._internals.errors import CapabilityError, GeneratorError


@dataclass
class TrainingHistory:
    """Per-epoch loss record and total wall-clock training duration.

    `loss` is the model's total per-epoch loss. The other lists are populated
    only for models that report the matching metric (empty otherwise):
    `reconstruction_loss`/`kl_loss` for autoencoders/VAEs,
    `classification_loss`/`accuracy` for models with a classification branch.
    `accuracy` is the mean per-batch training accuracy — a training-time
    indicator, not held-out performance.
    """

    loss: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    reconstruction_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    classification_loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)


class _Model(nn.Module):
    """Abstract base for every model built by a `build_*` function.

    Subclasses must set `input_shape`, `shape_error_cls`, `built_by` (the name
    of the public builder, for error messages), and — for models that
    classify — `num_classes` and `class_names`.
    """

    capabilities: frozenset[str] = frozenset()
    input_shape: tuple[int, int, int]
    shape_error_cls: type[Exception]
    built_by: str = "a picoface model"
    num_classes: int | None = None
    class_names: list[str]

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return this model's total loss for one batch, plus named metrics.

        `inputs` is a preprocessed NCHW float tensor in [0, 1]; `labels` is a
        long tensor of class indices. Metric names must match `TrainingHistory`
        fields.
        """
        raise NotImplementedError

    def check_data(self, data) -> None:
        """Reject `data` whose image shape or class count doesn't match this model."""
        image_shape = tuple(data.images.shape[1:])
        if image_shape != tuple(self.input_shape):
            raise self.shape_error_cls(
                f"image shape {image_shape} does not match this model's expected "
                f"input_shape {tuple(self.input_shape)}."
            )
        if self.num_classes is not None and len(data.class_names) != self.num_classes:
            raise self.shape_error_cls(
                f"data has {len(data.class_names)} class(es) "
                f"({data.class_names!r}), but this model expects "
                f"{self.num_classes} class(es)."
            )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for a preprocessed NCHW float batch (differentiable, mode-free)."""
        raise NotImplementedError

    def sample(self, n: int) -> torch.Tensor:
        """`n` newly sampled images as an NCHW float tensor in [0, 1]."""
        raise NotImplementedError

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """The latent mean for a preprocessed NCHW float batch."""
        raise NotImplementedError


def _require_model(model, verb: str) -> None:
    if not isinstance(model, _Model):
        raise CapabilityError(
            f"{verb}() requires a model built by one of picoface's build_*() "
            f"functions; got {type(model).__name__!r}."
        )


def _require_capability(model, capability: str, verb: str, error_cls=CapabilityError) -> None:
    _require_model(model, verb)
    if capability not in model.capabilities:
        raise error_cls(
            f"{verb}() requires a model that can {_CAPABILITY_PHRASES[capability]}; "
            f"got a model from {model.built_by}, which cannot."
        )


_CAPABILITY_PHRASES = {
    "classify": "classify images",
    "sample": "generate images (only build_vae() models can)",
    "latent_mean": "report a latent mean (only build_vae() models can)",
}


def _preprocess_images(images: np.ndarray | torch.Tensor, model: _Model) -> torch.Tensor:
    """Validate shape, preprocess NHWC uint8 images to NCHW float32 in [0, 1].

    Accepts a single image (H, W, C) or a batch (N, H, W, C); always returns a
    tensor with a leading batch dimension.
    """
    tensor = images if isinstance(images, torch.Tensor) else torch.from_numpy(np.asarray(images))

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    image_shape = tuple(tensor.shape[1:])
    if image_shape != tuple(model.input_shape):
        raise model.shape_error_cls(
            f"image shape {image_shape} does not match this model's expected "
            f"input_shape {tuple(model.input_shape)}."
        )

    return tensor.permute(0, 3, 1, 2).float() / 255.0


def train(
    model,
    data,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> TrainingHistory:
    """Train `model` on `data` for `epochs` epochs, returning a training history.

    Runs the full training loop (batching, forward pass, loss, backward pass,
    optimizer step, epoch iteration) internally — no training loop to write.
    Works unchanged for every model from a `build_*()` function: each model
    decides its own loss (a classifier trains on classification loss; a VAE on
    reconstruction, KL-divergence, and classification loss together).
    """
    _require_model(model, "train")
    model.check_data(data)
    if model.num_classes is not None:
        model.class_names = list(data.class_names)

    device = torch.device("cpu")
    model.to(device)
    model.train()

    labels_tensor = torch.from_numpy(np.asarray(data.labels)).long()
    dataset = TensorDataset(torch.from_numpy(np.asarray(data.images)), labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainingHistory()
    start_time = time.monotonic()

    for _epoch in range(epochs):
        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {}
        num_batches = 0

        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            inputs = _preprocess_images(batch_images, model).to(device)
            loss, metrics = model.training_step(inputs, batch_labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for name, value in metrics.items():
                epoch_metrics[name] = epoch_metrics.get(name, 0.0) + value
            num_batches += 1

        denom = max(num_batches, 1)
        history.loss.append(epoch_loss / denom)
        for name, total in epoch_metrics.items():
            getattr(history, name).append(total / denom)

    history.wall_clock_seconds = time.monotonic() - start_time
    return history


def evaluate(model, data) -> float:
    """Return classification accuracy (0 to 1) of `model` on `data`."""
    _require_capability(model, "classify", "evaluate")
    model.check_data(data)
    model.eval()

    with torch.no_grad():
        logits = model.classify(_preprocess_images(data.images, model))
        predicted = logits.argmax(dim=1).numpy()

    labels = np.asarray(data.labels)
    return float((predicted == labels).mean())


def predict(model, image: np.ndarray) -> str:
    """Return the predicted class name for a single `image`."""
    _require_capability(model, "classify", "predict")
    model.eval()

    with torch.no_grad():
        logits = model.classify(_preprocess_images(image, model))
        predicted_index = int(logits.argmax(dim=1).item())

    return model.class_names[predicted_index]


def generate(model, n: int) -> np.ndarray:
    """Sample `n` new images from a trained `build_vae()` model's latent space.

    Raises `GeneratorError` for any model that cannot generate (e.g. one built
    by `build_autoencoder()`, which has no probabilistic prior to sample from).
    """
    _require_capability(model, "sample", "generate", error_cls=GeneratorError)
    model.eval()

    with torch.no_grad():
        decoded = model.sample(n)
    images = decoded.permute(0, 2, 3, 1).numpy() * 255.0
    return np.clip(images, 0, 255).astype(np.uint8)


def _latent_mean(model, images: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return the model's latent mean (not a sampled z) for `images`."""
    _require_capability(model, "latent_mean", "show_latent_space", error_cls=GeneratorError)
    tensor = _preprocess_images(images, model)
    model.eval()
    with torch.no_grad():
        mu = model.encode_mu(tensor)
    return mu.numpy()
