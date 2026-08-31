"""Internal CNN model, training loop, and shared forward-pass helper for the classifier arm.

Not part of the public API — `picoface.classifier` wraps everything here
behind named, student-facing functions. Nothing exported from this module
is meant to be imported by student code.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Kernel/padding/stride constants shared by both layer construction and the
# minimum-input-size calculation, so the two can never drift apart.
_CONV_KERNEL_SIZE = 3
_CONV_PADDING = 1
_CONV_STRIDE = 1
_POOL_KERNEL_SIZE = 2
_POOL_STRIDE = 2
_NUM_POOLS = 2

_CONV1_OUT_CHANNELS = 8
_CONV2_OUT_CHANNELS = 16
_FC_HIDDEN_SIZE = 32


class _CNNClassifier(nn.Module):
    """A small, fixed CNN: two conv->ReLU->maxpool blocks plus an FC head."""

    def __init__(self, num_classes: int, input_shape: tuple[int, int, int]):
        super().__init__()
        height, width, channels = input_shape

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.class_names = [f"class_{i}" for i in range(num_classes)]

        self.conv1 = nn.Conv2d(
            channels,
            _CONV1_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            padding=_CONV_PADDING,
            stride=_CONV_STRIDE,
        )
        self.conv2 = nn.Conv2d(
            _CONV1_OUT_CHANNELS,
            _CONV2_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            padding=_CONV_PADDING,
            stride=_CONV_STRIDE,
        )
        self.pool = nn.MaxPool2d(kernel_size=_POOL_KERNEL_SIZE, stride=_POOL_STRIDE)
        self.relu = nn.ReLU()

        flatten_dim = self._compute_flatten_dim(height, width, channels)
        self.fc1 = nn.Linear(flatten_dim, _FC_HIDDEN_SIZE)
        self.fc2 = nn.Linear(_FC_HIDDEN_SIZE, num_classes)

    def _compute_flatten_dim(self, height: int, width: int, channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            features = self._features(dummy)
        return features.numel()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def minimum_input_spatial_size() -> int:
    """The smallest H/W that survives the fixed conv/pool stack (analytically)."""
    size = 1
    for _ in range(_NUM_POOLS):
        # Reverse the pool: each pool halves (stride 2, kernel 2) with floor division,
        # so the smallest surviving pre-pool size is size * stride.
        size = size * _POOL_STRIDE
    return size


def _build_classifier(
    num_classes: int,
    input_shape: tuple[int, int, int],
    shape_error_cls: type[Exception],
) -> "_CNNClassifier":
    """Construct the fixed CNN architecture for `num_classes`/`input_shape`.

    Raises `shape_error_cls` if `input_shape`'s H or W is at or below the
    architecture's analytically-computed minimum viable spatial size.
    """
    height, width, _channels = input_shape
    min_size = minimum_input_spatial_size()
    if height <= min_size or width <= min_size:
        raise shape_error_cls(
            f"input_shape {input_shape} is too small for this CNN architecture: "
            f"height and width must both be greater than {min_size} "
            f"(got height={height}, width={width})."
        )

    return _CNNClassifier(num_classes=num_classes, input_shape=input_shape)


def _forward(
    model: "_CNNClassifier", images: np.ndarray | torch.Tensor, shape_error_cls: type[Exception]
) -> torch.Tensor:
    """Validate shape, preprocess NHWC uint8 images, and run the forward pass.

    Accepts a single image (H, W, C) or a batch (N, H, W, C); always returns
    logits with a leading batch dimension.
    """
    tensor = images if isinstance(images, torch.Tensor) else torch.from_numpy(np.asarray(images))

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    image_shape = tuple(tensor.shape[1:])
    if image_shape != tuple(model.input_shape):
        raise shape_error_cls(
            f"image shape {image_shape} does not match this model's expected "
            f"input_shape {tuple(model.input_shape)}."
        )

    tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
    return model(tensor)


@dataclass
class TrainingHistory:
    """Per-epoch loss record and total wall-clock training duration."""

    loss: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0


def _train_loop(
    model: "_CNNClassifier",
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shape_error_cls: type[Exception],
) -> TrainingHistory:
    """Run the full training loop: DataLoader batching, cross-entropy loss, Adam, CPU."""
    device = torch.device("cpu")
    model.to(device)
    model.train()

    labels_tensor = torch.from_numpy(np.asarray(labels)).long()
    dataset = TensorDataset(torch.from_numpy(np.asarray(images)), labels_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = TrainingHistory()
    start_time = time.monotonic()

    for _epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch_images, batch_labels in loader:
            optimizer.zero_grad()
            logits = _forward(model, batch_images, shape_error_cls)
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        history.loss.append(epoch_loss / max(num_batches, 1))

    history.wall_clock_seconds = time.monotonic() - start_time
    return history
