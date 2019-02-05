from __future__ import absolute_import, division, print_function

import math

import pytest
import torch

import funsor
import funsor.distributions as dist


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
def test_normal_call(batch_shape):
    batch_dims = ("x", "y", "z")[:len(batch_shape)]

    @funsor.of_shape("real", "real", "real")
    def normal(loc, scale, value):
        return -((value - loc) ** 2) / (2 * scale ** 2) - scale.log() - math.log(math.sqrt(2 * math.pi))

    loc = funsor.Tensor(batch_dims, torch.randn(batch_shape))
    scale = funsor.Tensor(batch_dims, torch.randn(batch_shape).exp())
    value = funsor.Tensor(batch_dims, torch.randn(batch_shape))
    actual = dist.Normal(loc, scale)(value)
    assert isinstance(actual, funsor.Tensor)
    assert actual.dims == batch_dims
    assert actual.shape == batch_shape
    expected = normal(loc, scale, value)
    assert isinstance(expected, funsor.Tensor)
    assert expected.dims == batch_dims
    assert expected.shape == batch_shape
    assert ((actual - expected) < 1e-5).all()
