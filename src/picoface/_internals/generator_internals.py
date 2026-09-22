"""Internal encoder/decoder architecture, VAE machinery, and classification head for the generator arm.

Not part of the public API — `picoface.generator` wraps everything here
behind named, student-facing functions. Nothing exported from this module
is meant to be imported by student code. Shares no code with the classifier
arm's `_internals/classifier_internals.py`; the shared training loop and
verbs live in the arm-neutral `_internals/model_api.py`.
"""

import torch
from torch import nn

from picoface._internals.model_api import _Model

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

# Weight on the cross-entropy classification loss in the joint objective:
# loss = reconstruction (+ BETA * KL for the VAE) + CLASSIFICATION_WEIGHT * CE.
# Not student-facing. Head sits on the shared conv-trunk features, not on the
# latent: in a spike on synthetic shapes (4 classes, varied position/size),
# that gave the best classifier without hurting reconstruction, whereas a head
# on `mu` clustered the latent but cost ~35% reconstruction. Accuracy and
# reconstruction were insensitive to this weight across 0.1-10 (Adam
# normalizes per-parameter, so CE's larger raw scale doesn't starve the
# decoder), so 1.0 is the plain default. Validated on synthetic data only —
# flagged in openspec/ROADMAP.md for revisiting in Phase 6 with BETA.
CLASSIFICATION_WEIGHT = 1.0

_HEAD_HIDDEN_SIZE = 32

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


def _build_classification_head(flatten_dim: int, num_classes: int) -> nn.Sequential:
    """Small MLP head (mirrors the classifier arm's FC head) on the trunk features."""
    return nn.Sequential(
        nn.Linear(flatten_dim, _HEAD_HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(_HEAD_HIDDEN_SIZE, num_classes),
    )


def _classification_terms(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Cross-entropy loss tensor and batch accuracy for `logits` against `labels`."""
    ce = nn.functional.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return ce, accuracy


def _build_decoder(latent_dim: int, output_shape: tuple[int, int, int]) -> "_Decoder":
    return _Decoder(latent_dim, output_shape)


class _Autoencoder(_Model):
    """Plain (non-variational) encoder/decoder: a pedagogical step toward the VAE.

    Two parallel branches off the shared conv trunk: a deterministic latent
    projection feeding the decoder, and a classification head.
    """

    capabilities = frozenset({"classify"})
    built_by = "build_autoencoder()"

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        shape_error_cls: type[Exception],
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.shape_error_cls = shape_error_cls
        self.latent_dim = latent_dim
        self.trunk = _ConvEncoderTrunk(input_shape)
        self.to_latent = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.decoder = _build_decoder(latent_dim, input_shape)
        self.head = _build_classification_head(self.trunk.flatten_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.decoder(self.to_latent(features)), self.head(features)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon, logits = self(inputs)
        recon_loss = nn.functional.mse_loss(recon, inputs)
        cls_loss, accuracy = _classification_terms(logits, labels)
        loss = recon_loss + CLASSIFICATION_WEIGHT * cls_loss
        return loss, {
            "reconstruction_loss": recon_loss.item(),
            "classification_loss": cls_loss.item(),
            "accuracy": accuracy,
        }


class _VAE(_Model):
    """Variational autoencoder: probabilistic latent space + reparameterization.

    Two parallel branches off the shared conv trunk: a generative branch
    (`mu`/`logvar` -> reparameterize -> decoder) and a classification head.
    """

    capabilities = frozenset({"classify", "sample", "latent_mean"})
    built_by = "build_vae()"

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        shape_error_cls: type[Exception],
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.shape_error_cls = shape_error_cls
        self.latent_dim = latent_dim
        self.trunk = _ConvEncoderTrunk(input_shape)
        self.fc_mu = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.decoder = _build_decoder(latent_dim, input_shape)
        self.head = _build_classification_head(self.trunk.flatten_dim, num_classes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        mu, logvar = self.fc_mu(features), self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.head(features)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[0]

    def sample(self, n: int) -> torch.Tensor:
        """Draw `n` vectors from N(0, I) in the fixed latent space and decode them."""
        return self.decoder(torch.randn(n, self.latent_dim))

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon, mu, logvar, logits = self(inputs)
        recon_loss = nn.functional.mse_loss(recon, inputs)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        cls_loss, accuracy = _classification_terms(logits, labels)
        loss = recon_loss + BETA * kl_loss + CLASSIFICATION_WEIGHT * cls_loss
        return loss, {
            "reconstruction_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "classification_loss": cls_loss.item(),
            "accuracy": accuracy,
        }


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


def _build_autoencoder(
    input_shape: tuple[int, int, int], num_classes: int, shape_error_cls: type[Exception]
) -> "_Autoencoder":
    model = _Autoencoder(input_shape, num_classes, shape_error_cls, LATENT_DIM)
    _validate_decode_shape(model, input_shape, shape_error_cls)
    return model


def _build_vae(
    input_shape: tuple[int, int, int], num_classes: int, shape_error_cls: type[Exception]
) -> "_VAE":
    model = _VAE(input_shape, num_classes, shape_error_cls, LATENT_DIM)
    _validate_decode_shape(model, input_shape, shape_error_cls)
    return model
