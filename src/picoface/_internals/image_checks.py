"""The one check for picoface's image-array format, shared by `Dataset` and `predict()`.

Not part of the public API. An image array is uint8 with shape H×W×C (one
image) or N×H×W×C (a batch), where C is 1 (grayscale) or 3 (RGB). Each
violation raises the caller's error class with a message stating what was
expected, what was found, and how to fix it.
"""

import numpy as np

_ALLOWED_CHANNELS = (1, 3)


def check_image_array(array, *, name: str, batch: bool, error_cls: type[Exception]) -> None:
    """Raise `error_cls` unless `array` is a uint8 image array in picoface's format.

    `name` is how the message refers to the array (e.g. "images", "image").
    `batch=True` expects N×H×W×C; `batch=False` expects a single H×W×C image.
    Checks run rank, then channels, then dtype, so each message can assume the
    earlier checks passed.
    """
    if not isinstance(array, np.ndarray) or array.dtype == object:
        raise error_cls(
            f"{name} must be a numpy array of pixel values, but got "
            f"{type(array).__name__}"
            + (" holding non-numeric values" if isinstance(array, np.ndarray) else "")
            + "."
        )

    expected_layout = (
        "(count, height, width, channels)" if batch else "(height, width, channels)"
    )
    expected_ndim = 4 if batch else 3
    if array.ndim != expected_ndim:
        message = (
            f"{name} has shape {array.shape}, but picoface needs {name} shaped "
            f"{expected_layout} with 1 channel for grayscale or 3 for RGB."
        )
        if array.ndim == expected_ndim - 1:
            message += (
                f" For grayscale, add the channel axis: {name} = {name}[..., np.newaxis]"
            )
        raise error_cls(message)

    channels = array.shape[-1]
    if channels not in _ALLOWED_CHANNELS:
        raise error_cls(
            f"{name} has {channels} channels (shape {array.shape}), but picoface "
            f"supports only 1 channel (grayscale) or 3 channels (RGB)."
        )

    if array.dtype != np.uint8:
        message = (
            f"{name} has dtype {array.dtype}, but pixel values must be whole numbers "
            f"from 0 to 255 (dtype uint8)."
        )
        if np.issubdtype(array.dtype, np.floating):
            message += (
                f" If the values run from 0 to 1, convert with "
                f"{name} = ({name} * 255).round().astype(np.uint8)"
            )
        elif np.issubdtype(array.dtype, np.integer):
            message += (
                f" If the values are already 0 to 255, convert with "
                f"{name} = {name}.astype(np.uint8)"
            )
        raise error_cls(message)
