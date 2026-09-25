import numpy as np
import pytest

from picoface._internals.image_checks import check_image_array


class _Error(ValueError):
    pass


def _check(array, batch):
    check_image_array(array, name="images" if batch else "image", batch=batch, error_cls=_Error)


@pytest.mark.parametrize(
    "array,batch",
    [
        (np.zeros((2, 8, 8, 1), np.uint8), True),
        (np.zeros((2, 8, 6, 3), np.uint8), True),
        (np.zeros((8, 8, 1), np.uint8), False),
        (np.zeros((8, 6, 3), np.uint8), False),
    ],
)
def test_accepts_uint8_grayscale_and_rgb(array, batch):
    _check(array, batch)


@pytest.mark.parametrize(
    "shape,batch",
    [((2, 8, 8), True), ((8, 8), False)],
)
def test_missing_channel_axis_shows_the_fix(shape, batch):
    with pytest.raises(_Error) as info:
        _check(np.zeros(shape, np.uint8), batch)

    message = str(info.value)
    assert str(shape) in message
    assert "np.newaxis" in message


@pytest.mark.parametrize(
    "shape,batch",
    [((8, 8, 1), True), ((2, 2, 8, 8, 1), True), ((8,), False)],
)
def test_other_wrong_ranks_report_the_shape(shape, batch):
    with pytest.raises(_Error) as info:
        _check(np.zeros(shape, np.uint8), batch)

    assert str(shape) in str(info.value)


@pytest.mark.parametrize(
    "shape,batch",
    [((2, 8, 8, 4), True), ((2, 8, 8, 2), True), ((8, 8, 4), False)],
)
def test_unsupported_channel_counts_are_rejected(shape, batch):
    with pytest.raises(_Error) as info:
        _check(np.zeros(shape, np.uint8), batch)

    message = str(info.value)
    assert f"{shape[-1]} channels" in message
    assert "grayscale" in message and "RGB" in message


@pytest.mark.parametrize("batch", [True, False])
def test_float_arrays_are_rejected_with_a_conversion_hint(batch):
    shape = (2, 8, 8, 1) if batch else (8, 8, 1)
    with pytest.raises(_Error) as info:
        _check(np.zeros(shape, np.float32), batch)

    message = str(info.value)
    assert "float32" in message
    assert "0 to 255" in message
    assert "* 255" in message


@pytest.mark.parametrize("batch", [True, False])
def test_other_integer_dtypes_are_rejected(batch):
    shape = (2, 8, 8, 1) if batch else (8, 8, 1)
    with pytest.raises(_Error) as info:
        _check(np.zeros(shape, np.int64), batch)

    assert "int64" in str(info.value)


@pytest.mark.parametrize("value", [[[0, 1], [2, 3]], "not an image", np.array(object())])
def test_non_arrays_are_rejected(value):
    with pytest.raises(_Error) as info:
        _check(value, batch=False)

    assert "numpy array" in str(info.value)
