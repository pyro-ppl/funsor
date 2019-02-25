from __future__ import absolute_import, division, print_function

import os

import torch

from funsor.terms import Funsor
from funsor.torch import Tensor


if 'TORCH_TENSOR_TYPE' in os.environ:
    torch.set_default_tensor_type(os.environ['TORCH_TENSOR_TYPE'])


def assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    assert isinstance(actual, Funsor)
    assert isinstance(expected, Funsor)
    assert actual.inputs == expected.inputs, (actual.inputs, expected.inputs)
    assert actual.output == expected.output
    if isinstance(actual, Tensor):
        if actual.data.dtype in (torch.long, torch.uint8):
            assert (actual.data == expected.data).all()
        else:
            diff = (actual.data.detach() - expected.data.detach()).abs()
            assert diff.max() < atol
            assert (diff / (atol + expected.data.detach().abs())).max() < rtol
    else:
        raise ValueError('cannot compare objects of type {}'.format(type(actual)))


def check_funsor(x, inputs, output, data=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, Funsor)
    assert dict(x.inputs) == dict(inputs)
    if output is not None:
        assert x.output == output
    if data is not None:
        if x.inputs != inputs:
            # data = data.permute(tuple(dims.index(d) for d in x.dims))
            return  # TODO
        if inputs or output.shape:
            assert (x.data == data).all()
        else:
            assert x.data == data
