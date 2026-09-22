"""Internal encoder/decoder, VAE machinery, and latent classification head for the generator arm.

Not part of the public API — `picoface.generator` wraps everything here
behind named, student-facing functions. Nothing exported from this module
is meant to be imported by student code. Shares no code with the classifier
arm's `_internals/classifier_internals.py`; the shared training loop and
verbs live in the arm-neutral `_internals/model_api.py`.
"""

import torch
from torch import nn

from picoface._internals.model_api import _Model

# Latent dimensionality (phase3c Decision 2): no longer pinned to 2 now that
# `show_latent_space()` is gone and classification is routed through the
# latent. Provisional — see the phase3c diagnostics record; Phase 6 owns the
# final value.
LATENT_DIM = 8

# Fraction of training over which the KL weight ramps linearly from 0 to 1
# (phase3c Decision 3). The end point, 1, is the ELBO weight given the per-pixel loss
# scale (phase3c Decision 5) and is not a tuning knob; only this ramp length is.
# Provisional; Phase 6 owns the value.
KL_ANNEAL_FRACTION = 0.5

# Floor on both learned task log-variances (phase3c Decision 4): caps how large a
# task's weight can grow, bounding the lower-loss-earns-higher-weight
# starvation dynamic. Both variances live on per-pixel / per-label scales, so
# the floor means the same thing at every image resolution. Enforced on the
# stored parameters after every optimizer step (`_VAE.on_step_end`), not only
# on the values the loss reads. Provisional; Phase 6 owns the value.
LOG_VAR_FLOOR = -6.0

# Learning-rate multiplier for both learned task log-variances, relative to the
# rate passed to `train()` (phase3c Decision 4). Adam moves a parameter by about
# its learning rate per step, so at the shared rate the log-variances barely
# leave their initial value in a default-length run. 10 is deliberately
# conservative: it reaches equilibrium in about 340 steps, which is a few epochs
# on real data but not on the small stub. Provisional; Phase 6 retunes it once
# the real dataset's steps per epoch are known.
LOG_VAR_LR_MULTIPLIER = 10

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


def _build_classification_head(latent_dim: int, num_classes: int) -> nn.Sequential:
    """Small MLP head on the latent vector (phase3c Decision 1)."""
    return nn.Sequential(
        nn.Linear(latent_dim, _HEAD_HIDDEN_SIZE),
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


def _kl_weight(epoch: int, total_epochs: int) -> float:
    """KL weight for `epoch` (from 0): a linear ramp from 0 to 1 over the first
    `KL_ANNEAL_FRACTION` of training, then 1 (phase3c Decision 3). Always 1 on the
    final epoch, whatever `total_epochs` is.
    """
    if total_epochs <= 1:
        return 1.0
    progress = epoch / (total_epochs - 1)
    return min(1.0, progress / KL_ANNEAL_FRACTION)


def _build_decoder(latent_dim: int, output_shape: tuple[int, int, int]) -> "_Decoder":
    return _Decoder(latent_dim, output_shape)


class _Autoencoder(_Model):
    """Plain (non-variational) encoder/decoder, reconstruction-only.

    Optional, and not a prerequisite for the VAE (phase3c Decision 7): it neither
    classifies nor samples, and ignores a dataset's labels during training.
    """

    capabilities = frozenset()
    built_by = "build_autoencoder()"

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        shape_error_cls: type[Exception],
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.shape_error_cls = shape_error_cls
        self.latent_dim = latent_dim
        self.trunk = _ConvEncoderTrunk(input_shape)
        self.to_latent = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.decoder = _build_decoder(latent_dim, input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.to_latent(self.trunk(x)))

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss = nn.functional.mse_loss(self(inputs), inputs)
        return recon_loss, {"reconstruction_loss": recon_loss.item()}


class _VAE(_Model):
    """Supervised variational autoencoder: one model that classifies and generates.

    The reparameterized latent sample `z` feeds both the decoder and the
    classification head (phase3c Decision 1), so class supervision shapes the
    distribution `sample()` draws from; inference (`classify`) reads `mu`
    instead, for deterministic predictions.

    The loss is a per-pixel ELBO plus classification (phase3c Decisions 3-5):
    reconstruction is the per-pixel Gaussian negative log-likelihood with a
    learned noise scale, classification is cross-entropy with a learned
    uncertainty (Kendall et al., 2018), and the per-pixel KL divergence is
    weighted by an annealed schedule ending at 1.
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
        self.trunk = _ConvEncoderTrunk(input_shape)
        self.fc_mu = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.trunk.flatten_dim, latent_dim)
        self.decoder = _build_decoder(latent_dim, input_shape)
        self.head = _build_classification_head(latent_dim, num_classes)
        # Learned task log-variances (phase3c Decision 4): per-pixel reconstruction
        # noise and classification uncertainty.
        self.reconstruction_log_var = nn.Parameter(torch.zeros(()))
        self.classification_log_var = nn.Parameter(torch.zeros(()))
        # Full ELBO weight until `train()` starts a schedule.
        self.kl_weight = 1.0

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        self.kl_weight = _kl_weight(epoch, total_epochs)

    def _log_vars(self) -> list[nn.Parameter]:
        return [self.reconstruction_log_var, self.classification_log_var]

    def optimizer_param_groups(self, learning_rate: float) -> list[dict]:
        log_var_ids = {id(p) for p in self._log_vars()}
        others = [p for p in self.parameters() if id(p) not in log_var_ids]
        return [
            {"params": others, "lr": learning_rate},
            {"params": self._log_vars(), "lr": learning_rate * LOG_VAR_LR_MULTIPLIER},
        ]

    def on_step_end(self) -> None:
        # Floor the stored log-variances, not just the values the loss reads
        # (phase3c Decision 4): a parameter left below the floor would get zero
        # gradient and could not recover if its task loss rose again.
        with torch.no_grad():
            for log_var in self._log_vars():
                log_var.clamp_(min=LOG_VAR_FLOOR)

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.head(z)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_mu(x))

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def sample(self, n: int) -> torch.Tensor:
        """Draw `n` vectors from N(0, I) in the latent space and decode them."""
        return self.decode(torch.randn(n, self.latent_dim))

    def training_step(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon, mu, logvar, logits = self(inputs)

        # Per-pixel scale throughout (phase3c Decision 5): mean-reduced squared error,
        # and the per-image KL divided by the number of pixel values, so
        # reconstruction + KL is the negative ELBO / D up to a constant.
        mse = nn.functional.mse_loss(recon, inputs)
        kl_per_image = -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        kl = kl_per_image / self.num_pixel_values
        ce, accuracy = _classification_terms(logits, labels)

        # Kept at or above LOG_VAR_FLOOR by `on_step_end`.
        recon_log_var = self.reconstruction_log_var
        cls_log_var = self.classification_log_var
        recon_term = mse / (2 * recon_log_var.exp()) + 0.5 * recon_log_var
        cls_term = ce / cls_log_var.exp() + 0.5 * cls_log_var

        loss = recon_term + cls_term + self.kl_weight * kl
        return loss, {
            "reconstruction_loss": mse.item(),
            "kl_loss": kl.item(),
            "classification_loss": ce.item(),
            "accuracy": accuracy,
            "kl_weight": self.kl_weight,
            "reconstruction_log_var": recon_log_var.item(),
            "classification_log_var": cls_log_var.item(),
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
