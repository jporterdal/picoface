"""Shared exception base types for the classifier and generator arms.

Not part of the public API — `picoface.classifier` and `picoface.generator`
re-export the pieces students may want to catch.
"""


class BaseShapeError(ValueError):
    """Common base for the classifier arm's and generator arm's shape errors."""


class CapabilityError(ValueError):
    """Raised when a function is given a model that lacks a required capability."""


class GeneratorError(CapabilityError):
    """Raised when a function requiring a `build_vae()` model is given a
    model without that capability (e.g. a `build_autoencoder()` model).
    """
