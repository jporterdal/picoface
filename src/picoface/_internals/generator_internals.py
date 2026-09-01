"""Internal encoder/decoder architecture, VAE machinery, and training loop for the generator arm.

Not part of the public API — `picoface.generator` wraps everything here
behind named, student-facing functions. Nothing exported from this module
is meant to be imported by student code. Shares no code with the classifier
arm's `_internals/classifier_internals.py`.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Fixed 2D latent space (Decision 2): keeps `show_latent_space()` a direct
# (x, y) scatter with no dimensionality-reduction step / new dependency.
LATENT_DIM = 2

# KL-divergence weight for the VAE loss (Decision 3): a guess tuned
# qualitatively against stub-dataset reconstructions (see tasks.md 5.2 for
# the tuning record) — not validated against real data. Flagged in
# openspec/ROADMAP.md for revisiting in Phase 6. Deliberately kept small:
# with mean-reduced per-pixel MSE, recon loss on the stub dataset sits in
# the ~0.01-0.03 range, and larger BETA values (0.1, 1.0) pushed KL toward
# collapse (near zero) faster and further than this value does, without
# improving reconstruction — so this stays a light regularizer rather than
# the dominant loss term.
BETA = 0.01

_CONV_KERNEL_SIZE = 3
_CONV_PADDING = 1
_CONV_STRIDE = 2
_ENC1_OUT_CHANNELS = 8
_ENC2_OUT_CHANNELS = 16


def _conv_output_size(size: int) -> int:
    """Spatial size after one stride-2, padding-1, kernel-3 conv."""
    return (size + 2 * _CONV_PADDING - _CONV_KERNEL_SIZE) // _CONV_STRIDE + 1


def _encoded_feature_shape(input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """(channels, height, width) of the conv trunk's output for `input_shape`."""
    height, width, _channels = input_shape
    h2 = _conv_output_size(_conv_output_size(height))
    w2 = _conv_output_size(_conv_output_size(width))
    return (_ENC2_OUT_CHANNELS, h2, w2)


class _ConvEncoderTrunk(nn.Module):
    """Two conv->ReLU stride-2 blocks, producing flattened features.

    Shared architecture between the plain autoencoder's encoder and the
    VAE's encoder (Decision 1) — each builds its own instance of this trunk.
    """

    def __init__(self, input_shape: tuple[int, int, int]):
        super().__init__()
        _height, _width, channels = input_shape
        self.conv1 = nn.Conv2d(
            channels,
            _ENC1_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            stride=_CONV_STRIDE,
            padding=_CONV_PADDING,
        )
        self.conv2 = nn.Conv2d(
            _ENC1_OUT_CHANNELS,
            _ENC2_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            stride=_CONV_STRIDE,
            padding=_CONV_PADDING,
        )
        self.relu = nn.ReLU()
        self.feature_shape = _encoded_feature_shape(input_shape)
        self.flatten_dim = (
            self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.flatten(start_dim=1)


class _Encoder(nn.Module):
    """Conv trunk + linear projection directly to a `latent_dim`-vector z.

    Used by the plain autoencoder, which has no probabilistic latent step.
    """

    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int):
        super().__init__()
        self.trunk = _ConvEncoderTrunk(input_shape)
        self.to_latent = nn.Linear(self.trunk.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.trunk(x))


class _Decoder(nn.Module):
    """Linear projection from latent space, then two transpose-conv->ReLU
    stride-2 blocks back up to `output_shape`, with a sigmoid output layer.
    """

    def __init__(self, latent_dim: int, output_shape: tuple[int, int, int]):
        super().__init__()
        _out_height, _out_width, out_channels = output_shape
        feat_channels, feat_height, feat_width = _encoded_feature_shape(output_shape)
        self.feature_shape = (feat_channels, feat_height, feat_width)
        self.fc = nn.Linear(latent_dim, feat_channels * feat_height * feat_width)
        self.deconv1 = nn.ConvTranspose2d(
            feat_channels,
            _ENC1_OUT_CHANNELS,
            kernel_size=_CONV_KERNEL_SIZE,
            stride=_CONV_STRIDE,
            padding=_CONV_PADDING,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            _ENC1_OUT_CHANNELS,
            out_channels,
            kernel_size=_CONV_KERNEL_SIZE,
            stride=_CONV_STRIDE,
            padding=_CONV_PADDING,
            output_padding=1,
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, *self.feature_shape)
        x = self.relu(self.deconv1(x))
        return self.sigmoid(self.deconv2(x))


def _build_encoder(input_shape: tuple[int, int, int], latent_dim: int) -> "_Encoder":
    return _Encoder(input_shape, latent_dim)


def _build_decoder(latent_dim: int, output_shape: tuple[int, int, int]) -> "_Decoder":
    return _Decoder(latent_dim, output_shape)


class _Autoencoder(nn.Module):
    """Plain (non-variational) encoder/decoder: a pedagogical step toward the VAE."""

    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.is_variational = False
        self.encoder = _build_encoder(input_shape, latent_dim)
        self.decoder = _build_decoder(latent_dim, input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _VAE(nn.Module):
    """Variational autoencoder: probabilistic latent space + reparameterization."""

    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.is_variational = True
        self.trunk = _ConvEncoderTrunk(input_shape)
        self.fc_mu = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.decoder = _build_decoder(latent_dim, input_shape)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def _validate_decode_shape(model: nn.Module, input_shape, shape_error_cls: type[Exception]) -> None:
    """Run a dummy tensor through the full encode->decode path and confirm
    the decoded shape exactly matches `input_shape` (Decision 7).
    """
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


def _build_autoencoder(input_shape: tuple[int, int, int], shape_error_cls: type[Exception]) -> "_Autoencoder":
    model = _Autoencoder(input_shape, LATENT_DIM)
    _validate_decode_shape(model, input_shape, shape_error_cls)
    return model


def _build_vae(input_shape: tuple[int, int, int], shape_error_cls: type[Exception]) -> "_VAE":
    model = _VAE(input_shape, LATENT_DIM)
    _validate_decode_shape(model, input_shape, shape_error_cls)
    return model


def _preprocess(
    images: np.ndarray | torch.Tensor, input_shape, shape_error_cls: type[Exception]
) -> torch.Tensor:
    """Validate shape, preprocess NHWC uint8 images to NCHW float32 in [0, 1].

    Local to this module — duplicated from, not imported from, the
    classifier arm's equivalent (Decision 9).
    """
    tensor = images if isinstance(images, torch.Tensor) else torch.from_numpy(np.asarray(images))

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    image_shape = tuple(tensor.shape[1:])
    if image_shape != tuple(input_shape):
        raise shape_error_cls(
            f"image shape {image_shape} does not match this model's expected "
            f"input_shape {tuple(input_shape)}."
        )

    return tensor.permute(0, 3, 1, 2).float() / 255.0


def _encode_mu(model, images, shape_error_cls: type[Exception]) -> np.ndarray:
    """Return the VAE encoder's mean output (not a sampled z) for `images`."""
    tensor = _preprocess(images, model.input_shape, shape_error_cls)
    model.eval()
    with torch.no_grad():
        mu, _logvar = model.encode(tensor)
    return mu.numpy()


def _sample_generate(model, n: int) -> np.ndarray:
    """Draw `n` vectors from N(0, I) in the fixed latent space and decode them."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim)
        decoded = model.decoder(z)
    images = decoded.permute(0, 2, 3, 1).numpy() * 255.0
    return np.clip(images, 0, 255).astype(np.uint8)


@dataclass
class TrainingHistory:
    """Per-epoch loss record and total wall-clock training duration.

    `reconstruction_loss`/`kl_loss` are populated only for VAE models
    (empty lists for a plain autoencoder).
    """

    loss: list[float] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    reconstruction_loss: list[float] = field(default_factory=list)
    kl_loss: list[float] = field(default_factory=list)


def _train_loop(
    model,
    images: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shape_error_cls: type[Exception],
) -> TrainingHistory:
    """Run the full training loop: DataLoader batching, Adam, CPU.

    Dispatches on `model.is_variational` for loss selection: MSE
    reconstruction-only (autoencoder) vs. reconstruction + BETA * KL (VAE).
    """
    device = torch.device("cpu")
    model.to(device)
    model.train()

    dataset = TensorDataset(torch.from_numpy(np.asarray(images)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainingHistory()
    start_time = time.monotonic()

    for _epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        num_batches = 0

        for (batch_images,) in loader:
            optimizer.zero_grad()
            inputs = _preprocess(batch_images, model.input_shape, shape_error_cls).to(device)

            if model.is_variational:
                recon, mu, logvar = model(inputs)
                recon_loss = nn.functional.mse_loss(recon, inputs)
                kl_loss = -0.5 * torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                )
                loss = recon_loss + BETA * kl_loss
            else:
                recon = model(inputs)
                recon_loss = nn.functional.mse_loss(recon, inputs)
                kl_loss = None
                loss = recon_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            if kl_loss is not None:
                epoch_kl += kl_loss.item()
            num_batches += 1

        history.loss.append(epoch_loss / max(num_batches, 1))
        if model.is_variational:
            history.reconstruction_loss.append(epoch_recon / max(num_batches, 1))
            history.kl_loss.append(epoch_kl / max(num_batches, 1))

    history.wall_clock_seconds = time.monotonic() - start_time
    return history
