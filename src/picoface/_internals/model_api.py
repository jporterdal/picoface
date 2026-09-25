"""Abstract model interface and the model-agnostic verbs built on it.

Not part of the public API — `picoface.classifier` and `picoface.generator`
re-export the verbs defined here (`train`, `evaluate`, `predict`, `generate`).
This module is arm-neutral: it imports nothing from the public arm modules or
`viz`, and neither arm's internals import the other's.

Every model object in the project subclasses `_Model`, which supplies the one
thing the shared training loop cannot know (`training_step`, i.e. the loss) and
declares optional capabilities (`classify`, `sample`, `latent_access`) that
callers check before use. `latent_access` backs no public verb; only
`picoface.linkage` uses it. `train()` also tells each model where it is in training
(`on_epoch_start`), so a model can schedule its own loss terms, and lets each
model set its own per-parameter learning rates (`optimizer_param_groups`) and
constrain its parameters after every optimizer step (`on_step_end`).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from picoface._internals.errors import CapabilityError, GeneratorError
from picoface._internals.image_checks import check_image_array
from picoface.datasets import Dataset


@dataclass
class TrainingHistory:
    """Per-epoch loss record and total wall-clock training duration.

    `loss` is the model's total per-epoch loss. The other lists are populated
    only for models that report the matching metric (empty otherwise):
    `reconstruction_loss` (per-pixel mean squared error) for autoencoders and
    VAEs; `kl_loss` (KL divergence per pixel value, i.e. the per-image KL
    divided by height x width x channels), `classification_loss`, `accuracy`,
    `kl_weight` (the annealed weight applied to `kl_loss`), and the learned
    `reconstruction_log_var`/`classification_log_var` task weights for VAEs.
    `accuracy` is the mean per-batch training accuracy — a training-time
    indicator, not held-out performance. A VAE's total `loss` includes the
    learned log-variance terms and can therefore be negative.
    """

    loss: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    reconstruction_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)
    classification_loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    kl_weight: list[float] = field(default_factory=list)
    reconstruction_log_var: list[float] = field(default_factory=list)
    classification_log_var: list[float] = field(default_factory=list)


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

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Called by `train()` before each epoch (`epoch` counts from 0).

        No-op by default. A model whose loss depends on training progress
        (e.g. an annealed term) overrides this; progress is relative to the
        caller's `epochs`, so schedules complete within any run length.
        """

    def optimizer_param_groups(self, learning_rate: float) -> list[dict]:
        """Parameter groups for `train()`'s optimizer, given the caller's rate.

        One group of every parameter at `learning_rate` by default. A model
        that trains some parameters at a different rate overrides this, setting
        that rate relative to `learning_rate`; every parameter must appear in
        exactly one group.
        """
        return [{"params": list(self.parameters()), "lr": learning_rate}]

    def on_step_end(self) -> None:
        """Called by `train()` after every optimizer step.

        No-op by default. A model with constraints on its parameters (e.g. a
        lower bound) overrides this to enforce them in place.
        """

    def check_data(self, data) -> None:
        """Reject `data` whose image shape or class count doesn't match this model."""
        image_shape = tuple(data.images.shape[1:])
        if image_shape != tuple(self.input_shape):
            raise self.shape_error_cls(
                f"image shape {image_shape} does not match this model's expected "
                f"input_shape {tuple(self.input_shape)}."
            )
        if self.num_classes is not None and data.num_classes != self.num_classes:
            raise self.shape_error_cls(
                f"data has {data.num_classes} class(es) "
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
        """Latent mean vectors for a preprocessed NCHW float batch (`latent_access`)."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Images for a batch of latent vectors, NCHW float in [0, 1] (`latent_access`)."""
        raise NotImplementedError


def _require_model(model, verb: str) -> None:
    if not isinstance(model, _Model):
        raise CapabilityError(
            f"{verb}() requires a model built by one of picoface's build_*() "
            f"functions; got {type(model).__name__!r}."
        )


def _require_dataset(data, verb: str) -> None:
    """Raise `TypeError`, naming what was given, unless `data` is a `Dataset`."""
    if isinstance(data, Dataset):
        return
    if isinstance(data, (str, Path)):
        raise TypeError(
            f"{verb}() needs a dataset, but was given the file path {str(data)!r}. "
            f"Load it first: data = load_dataset({str(data)!r})"
        )
    if hasattr(data, "getImage"):
        given, single_image = "a picture", True
    elif isinstance(data, np.ndarray):
        given, single_image = f"an image array of shape {data.shape}", True
    else:
        given, single_image = f"a {type(data).__name__}", False
    message = f"{verb}() needs a dataset from load_dataset(), but was given {given}."
    if single_image:
        message += " To classify a single image, use predict(model, image)."
    raise TypeError(message)


def _require_capability(model, capability: str, verb: str, error_cls=CapabilityError) -> None:
    _require_model(model, verb)
    if capability not in model.capabilities:
        raise error_cls(
            f"{verb}() requires a model that can {_CAPABILITY_PHRASES[capability]}; "
            f"got a model from {model.built_by}, which cannot."
        )


_CAPABILITY_PHRASES = {
    "classify": "classify images (build_autoencoder() models cannot)",
    "sample": "generate images (only build_vae() models can)",
    "latent_access": "encode images to and decode them from a latent space "
    "(only build_vae() models can)",
}


def _to_uint8_images(images: torch.Tensor) -> np.ndarray:
    """NCHW float images in [0, 1] to NHWC uint8, the library's image format."""
    array = images.detach().permute(0, 2, 3, 1).numpy() * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


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
    data: Dataset,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> TrainingHistory:
    """Train `model` on `data` for `epochs` epochs, returning a training history.

    `data` is a dataset from `load_dataset()`.

    Runs the full training loop (batching, forward pass, loss, backward pass,
    optimizer step, epoch iteration) internally — no training loop to write.
    Works unchanged for every model from a `build_*()` function: each model
    decides its own loss (a classifier trains on classification loss; a VAE on
    reconstruction, KL-divergence, and classification loss together).
    """
    _require_model(model, "train")
    _require_dataset(data, "train")
    model.check_data(data)
    if model.num_classes is not None:
        model.class_names = list(data.class_names)

    device = torch.device("cpu")
    model.to(device)
    model.train()

    labels_tensor = torch.from_numpy(np.asarray(data.labels)).long()
    dataset = TensorDataset(torch.from_numpy(np.asarray(data.images)), labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.optimizer_param_groups(learning_rate))

    history = TrainingHistory()
    start_time = time.monotonic()

    for epoch in range(epochs):
        model.on_epoch_start(epoch, epochs)
        epoch_loss = 0.0
        epoch_metrics: dict[str, float] = {}
        num_batches = 0

        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            inputs = _preprocess_images(batch_images, model).to(device)
            loss, metrics = model.training_step(inputs, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            model.on_step_end()

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


def evaluate(model, data: Dataset) -> float:
    """Return classification accuracy (0 to 1) of `model` on `data`.

    `data` is a dataset from `load_dataset()`.

    Raises `GeneratorError` (a `CapabilityError`) for a model that cannot
    classify, such as one built by `build_autoencoder()`.
    """
    _require_capability(model, "classify", "evaluate", error_cls=GeneratorError)
    _require_dataset(data, "evaluate")
    model.check_data(data)
    model.eval()

    with torch.no_grad():
        logits = model.classify(_preprocess_images(data.images, model))
        predicted = logits.argmax(dim=1).numpy()

    labels = np.asarray(data.labels)
    return float((predicted == labels).mean())


def predict(model, image: np.ndarray) -> str:
    """Return the predicted class name for a single image array.

    `image` is one uint8 image array shaped (height, width, channels), the
    same size as the images the model was built for, e.g. `data.images[0]`.
    Raises the model's `ShapeError` for an image array of the wrong shape,
    dtype, or size, or for a whole batch of images; raises `GeneratorError`
    (a `CapabilityError`) for a model that cannot classify, such as one built
    by `build_autoencoder()`.
    """
    _require_capability(model, "classify", "predict", error_cls=GeneratorError)
    # A picture (mediacomp-bridge) is converted to an image array here, so the
    # checks below apply to it too.
    _check_single_image(image, model)
    model.eval()

    with torch.no_grad():
        logits = model.classify(_preprocess_images(image, model))
        predicted_index = int(logits.argmax(dim=1).item())

    return model.class_names[predicted_index]


def _check_single_image(image, model: _Model) -> None:
    """Reject anything but one uint8 H×W×C image array, with the model's shape error."""
    if isinstance(image, np.ndarray) and image.ndim == 4:
        raise model.shape_error_cls(
            f"predict() takes one image, but was given a batch of {image.shape[0]:,} "
            f"images (shape {image.shape}). To score many labeled images, use "
            f"evaluate(model, data); to classify one of them, use "
            f"predict(model, data.images[i])."
        )
    check_image_array(image, name="image", batch=False, error_cls=model.shape_error_cls)


def generate(model, n: int) -> np.ndarray:
    """Sample `n` new images from a trained `build_vae()` model's latent space.

    Returns a uint8 image array shaped (n, height, width, channels).

    Raises `GeneratorError` for any model that cannot generate (e.g. one built
    by `build_autoencoder()`, which has no probabilistic prior to sample from).
    """
    _require_capability(model, "sample", "generate", error_cls=GeneratorError)
    model.eval()

    with torch.no_grad():
        return _to_uint8_images(model.sample(n))

