"""Encoder, decoder, VAE and autoencoder for the generator arm.

Internal: `picoface.generator` wraps these behind its public functions.
Pointers such as "phase3c Decision 4" refer to the archived OpenSpec changes
`archive/2026-09-22-picoface-phase3c/design.md` and
`archive/2026-09-23-picoface-phase6/diagnostics.md` under `openspec/changes/`.
"""

from typing import NamedTuple

import torch
from torch import nn

from picoface._internals.model_api import _Model

# Size of the latent vector. See phase6 diagnostics.md, "`latent_dim` retuning".
LATENT_DIM = 128

# Fraction of training over which the KL weight ramps from 0 to 1 (phase3c Decision 3).
KL_ANNEAL_FRACTION = 0.5

# Floor and learning-rate multiplier for the learned task log-variances (phase3c Decision 4).
LOG_VAR_FLOOR = -6.0
LOG_VAR_LR_MULTIPLIER = 10


class _Reshape(nn.Module):
    """Reshape a flat batch to `(batch, *shape)`."""

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, *self.shape)


class _ConvEncoder(nn.Module):
    """Two stride-2 conv->ReLU blocks, producing flattened features."""

    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__()
        height, width, channels = input_shape

        def conv_block(in_channels: int, out_channels: int) -> list[nn.Module]:
            return [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ]

        self.convs = nn.Sequential(
            *conv_block(channels, 8),
            *conv_block(8, 16),
        )

        with torch.no_grad():
            features = self.convs(torch.zeros(1, channels, height, width))
        self.feature_shape = tuple(features.shape[1:])
        self.flatten_dim = features.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x).flatten(start_dim=1)


class _Decoder(nn.Module):
    """Linear projection up to the encoder's feature map, then two stride-2
    transpose convs back to image size, with a sigmoid output.
    """

    def __init__(self, latent_dim: int, feature_shape: tuple[int, int, int], out_channels: int):
        super().__init__()
        feature_channels, feature_height, feature_width = feature_shape

        def deconv(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            )

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, feature_channels * feature_height * feature_width),
            _Reshape(feature_shape),
            deconv(feature_channels, 128),  # width: phase6 diagnostics.md, "Decoder capacity escalation"
            nn.ReLU(),
            deconv(128, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


def _build_classification_head(latent_dim: int, num_classes: int) -> nn.Sequential:
    """Small MLP from the latent vector to class logits (phase3c Decision 1)."""
    return nn.Sequential(
        nn.Linear(latent_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample z ~ N(mu, exp(logvar)) so gradients flow through mu and logvar."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, exp(logvar)) || N(0, I)) per image, averaged over the batch."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def _kl_weight(epoch: int, total_epochs: int, anneal_fraction: float = KL_ANNEAL_FRACTION) -> float:
    """Linear ramp from 0 to 1 over the first `anneal_fraction` of training,
    then 1. Always 1 on the final epoch (phase3c Decision 3).
    """
    if total_epochs <= 1:
        return 1.0
    progress = epoch / (total_epochs - 1)
    return min(1.0, progress / anneal_fraction)


def _classification_terms(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Cross-entropy loss and batch accuracy."""
    ce = nn.functional.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return ce, accuracy


class _LearnedTaskWeights(nn.Module):
    """Learned log-variances that balance reconstruction against classification
    (Kendall et al., 2018; phase3c Decision 4).
    """

    def __init__(self):
        super().__init__()
        self.reconstruction_log_var = nn.Parameter(torch.zeros(()))
        self.classification_log_var = nn.Parameter(torch.zeros(()))

    def weigh_reconstruction(self, mse: torch.Tensor) -> torch.Tensor:
        log_var = self.reconstruction_log_var
        return mse / (2 * log_var.exp()) + 0.5 * log_var

    def weigh_classification(self, ce: torch.Tensor) -> torch.Tensor:
        log_var = self.classification_log_var
        return ce / log_var.exp() + 0.5 * log_var

    def clamp_to_floor(self, floor: float = LOG_VAR_FLOOR) -> None:
        # Clamp the stored parameters, not just the values the loss reads: one left
        # below the floor would get zero gradient and could not recover.
        with torch.no_grad():
            for log_var in self.parameters():
                log_var.clamp_(min=floor)


class _VAEOutput(NamedTuple):
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    logits: torch.Tensor


class _Autoencoder(_Model):
    """Plain encoder/decoder, reconstruction-only (phase3c Decision 7)."""

    capabilities = frozenset()
    built_by = "build_autoencoder()"

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        shape_error_cls: type[Exception],
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        _height, _width, channels = input_shape
        self.input_shape = input_shape
        self.shape_error_cls = shape_error_cls
        self.latent_dim = latent_dim

        self.encoder = _ConvEncoder(input_shape)
        self.to_latent = nn.Linear(self.encoder.flatten_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, self.encoder.feature_shape, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.to_latent(self.encoder(x)))

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss = nn.functional.mse_loss(self(inputs), inputs)
        return recon_loss, {"reconstruction_loss": recon_loss.item()}


class _VAE(_Model):
    """Supervised VAE: one model that classifies and generates.

    The sample `z` feeds both the decoder and the classification head;
    `classify()` reads `mu` instead, for deterministic predictions.
    """

    capabilities = frozenset({"classify", "sample", "latent_access"})
    built_by = "build_vae()"

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        shape_error_cls: type[Exception],
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        height, width, channels = input_shape
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.shape_error_cls = shape_error_cls
        self.latent_dim = latent_dim
        self.num_pixel_values = height * width * channels

        self.task_weights = _LearnedTaskWeights()
        self.encoder = _ConvEncoder(input_shape)
        self.fc_mu = nn.Linear(self.encoder.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.flatten_dim, latent_dim)
        self.decoder = _Decoder(latent_dim, self.encoder.feature_shape, channels)
        self.head = _build_classification_head(latent_dim, num_classes)

        # Full ELBO weight until `train()` starts a schedule.
        self.kl_weight = 1.0

    @property
    def reconstruction_log_var(self) -> nn.Parameter:
        return self.task_weights.reconstruction_log_var

    @property
    def classification_log_var(self) -> nn.Parameter:
        return self.task_weights.classification_log_var

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.fc_mu(features), self.fc_logvar(features)

    def forward(self, x: torch.Tensor) -> _VAEOutput:
        mu, logvar = self.encode(x)
        z = _reparameterize(mu, logvar)
        return _VAEOutput(self.decoder(z), mu, logvar, self.head(z))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_mu(x))

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        mu, _logvar = self.encode(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, n: int) -> torch.Tensor:
        """Draw `n` vectors from N(0, I) in the latent space and decode them."""
        return self.decode(torch.randn(n, self.latent_dim))

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        output = self(inputs)

        # Per-pixel scale for reconstruction and KL (phase3c Decision 5).
        mse = nn.functional.mse_loss(output.reconstruction, inputs)
        kl = _kl_divergence(output.mu, output.logvar) / self.num_pixel_values
        ce, accuracy = _classification_terms(output.logits, labels)

        loss = (
            self.task_weights.weigh_reconstruction(mse)
            + self.task_weights.weigh_classification(ce)
            + self.kl_weight * kl
        )
        return loss, {
            "reconstruction_loss": mse.item(),
            "kl_loss": kl.item(),
            "classification_loss": ce.item(),
            "accuracy": accuracy,
            "kl_weight": self.kl_weight,
            "reconstruction_log_var": self.reconstruction_log_var.item(),
            "classification_log_var": self.classification_log_var.item(),
        }

    # Training hooks, called by train()

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.kl_weight = _kl_weight(epoch, total_epochs)

    def optimizer_param_groups(self, learning_rate: float) -> list[dict]:
        log_vars = list(self.task_weights.parameters())
        log_var_ids = {id(p) for p in log_vars}
        others = [p for p in self.parameters() if id(p) not in log_var_ids]
        return [
            {"params": others, "lr": learning_rate},
            {"params": log_vars, "lr": learning_rate * LOG_VAR_LR_MULTIPLIER},
        ]

    def on_step_end(self) -> None:
        self.task_weights.clamp_to_floor()


def _validate_decode_shape(model: nn.Module, input_shape, shape_error_cls: type[Exception]) -> None:
    """Raise `shape_error_cls` unless encode->decode returns exactly `input_shape`."""
    height, width, channels = input_shape
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, channels, height, width)
        output = model(dummy)
        if isinstance(output, tuple):
            output = output[0]
    decoded_shape = tuple(output.shape[1:])
    expected_shape = (channels, height, width)
    if decoded_shape != expected_shape:
        raise shape_error_cls(
            f"decoder output shape {decoded_shape} does not match expected "
            f"{expected_shape} for input_shape {input_shape}: the encoder/decoder "
            "conv-transpose arithmetic doesn't round-trip for this image size."
        )
    model.train()


def _build_autoencoder(
    input_shape: tuple[int, int, int], shape_error_cls: type[Exception]
) -> "_Autoencoder":
    model = _Autoencoder(input_shape, shape_error_cls, LATENT_DIM)
    _validate_decode_shape(model, input_shape, shape_error_cls)
    return model


def _build_vae(
    input_shape: tuple[int, int, int], num_classes: int, shape_error_cls: type[Exception]
) -> "_VAE":
    model = _VAE(input_shape, num_classes, shape_error_cls, LATENT_DIM)
    _validate_decode_shape(model, input_shape, shape_error_cls)
    return model
