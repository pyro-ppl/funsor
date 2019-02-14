from __future__ import absolute_import, division, print_function

from funsor.terms import Funsor
from funsor.torch import Tensor


def assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    assert isinstance(actual, Funsor)
    assert isinstance(expected, Funsor)
    assert type(actual) == type(expected)
    assert actual.dims == expected.dims, (actual.dims, expected.dims)
    assert actual.shape == expected.shape
    if isinstance(actual, Tensor):
        diff = (actual.data.detach() - expected.data.detach()).abs()
        assert diff.max() < atol
        assert (diff / (atol + expected.data.detach().abs())).max() < rtol
    else:
        raise ValueError('cannot compare objects of type {}'.format(type(actual)))


def check_funsor(x, dims, shape, data=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, Funsor)
    assert set(x.dims) == set(dims)
    if shape is not None:
        assert dict(zip(x.dims, x.shape)) == dict(zip(dims, shape))
    if data is not None:
        if x.dims != dims:
            data = data.permute(tuple(dims.index(d) for d in x.dims))
        if dims:
            assert (x.data == data).all()
        else:
            assert x.data == data
