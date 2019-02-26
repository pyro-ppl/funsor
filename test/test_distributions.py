from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import pytest
import torch

import funsor
import funsor.distributions as dist
from funsor import Tensor
from funsor.domains import ints, reals
from funsor.testing import assert_close, check_funsor


@pytest.mark.xfail(reason='TODO fix eager_binary_tensor_tensor(ops.getitem,-,-)')
@pytest.mark.parametrize('size', [4])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
def test_categorical_density(size, batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, ints(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.of_shape(reals(size), ints(size))
    def categorical(probs, value):
        return probs[value].log()

    check_funsor(categorical, {'probs': reals(size), 'value': ints(size)}, reals())

    probs = Tensor(torch.randn(batch_shape + (size,)), inputs)
    value = Tensor((torch.rand(batch_shape) * size).long().clamp(min=0, max=size-1), inputs)
    expected = categorical(probs, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Categorical(probs, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
def test_normal_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, ints(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.of_shape(reals(), reals(), reals())
    def normal(loc, scale, value):
        return -((value - loc) ** 2) / (2 * scale ** 2) - scale.log() - math.log(math.sqrt(2 * math.pi))

    check_funsor(normal, {'loc': reals(), 'scale': reals(), 'value': reals()}, reals())

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape), inputs)
    expected = normal(loc, scale, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Normal(loc, scale, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)
