"""Small CNN for the classifier arm.

Internal: `picoface.classifier` wraps this behind its public functions.
"""

import torch
from torch import nn

from picoface._internals.model_api import _Model

# Pooling stages and their size, shared with `minimum_input_spatial_size()`.
_NUM_POOLS = 2
_POOL_SIZE = 2


class _CNNClassifier(_Model):
    """Two conv->ReLU->maxpool blocks, then a fully connected head."""

    capabilities = frozenset({"classify"})
    built_by = "build_classifier()"

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int],
        shape_error_cls: type[Exception],
    ):
        super().__init__()
        height, width, channels = input_shape
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.shape_error_cls = shape_error_cls
        self.class_names = [f"class_{i}" for i in range(num_classes)]

        def conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=_POOL_SIZE, stride=_POOL_SIZE),
            ]

        self.features = nn.Sequential(  # one block per pooling stage (_NUM_POOLS)
            *conv_block(channels, 8),
            *conv_block(8, 16),
        )

        with torch.no_grad():
            flatten_dim = self.features(torch.zeros(1, channels, height, width)).numel()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(x)

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = nn.functional.cross_entropy(self.classify(inputs), labels)
        return loss, {}


def minimum_input_spatial_size() -> int:
    """The H/W that the pooling stages reduce to 1; inputs must be larger."""
    return _POOL_SIZE**_NUM_POOLS


def _build_classifier(
    num_classes: int,
    input_shape: tuple[int, int, int],
    shape_error_cls: type[Exception],
) -> "_CNNClassifier":
    """Build the CNN, raising `shape_error_cls` if H or W is at or below
    `minimum_input_spatial_size()`.
    """
    height, width, _channels = input_shape
    min_size = minimum_input_spatial_size()
    if height <= min_size or width <= min_size:
        raise shape_error_cls(
            f"input_shape {input_shape} is too small for this CNN architecture: "
            f"height and width must both be greater than {min_size} "
            f"(got height={height}, width={width})."
        )

    return _CNNClassifier(
        num_classes=num_classes,
        input_shape=input_shape,
        shape_error_cls=shape_error_cls,
    )
