from __future__ import absolute_import, division, print_function

import math

import pytest
import torch

import funsor
import funsor.distributions as dist


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
def test_normal_density(batch_shape):
    batch_dims = ('x', 'y', 'z')[:len(batch_shape)]

    @funsor.of_shape('real', 'real', 'real')
    def normal(loc, scale, value):
        return -((value - loc) ** 2) / (2 * scale ** 2) - scale.log() - math.log(math.sqrt(2 * math.pi))

    assert isinstance(normal, funsor.Funsor)
    assert normal.dims == ('loc', 'scale', 'value')
    assert normal.shape == ('real', 'real', 'real')

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


@pytest.mark.xfail(reason='missing log_abs_det_jacobian term')
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
def test_log_normal_density(batch_shape):
    batch_dims = ('x', 'y', 'z')[:len(batch_shape)]

    @funsor.of_shape('real', 'real', 'real')
    def normal(loc, scale, value):
        return -((value - loc) ** 2) / (2 * scale ** 2) - scale.log() - math.log(math.sqrt(2 * math.pi))

    assert isinstance(normal, funsor.Funsor)
    assert normal.dims == ('loc', 'scale', 'value')
    assert normal.shape == ('real', 'real', 'real')

    value = funsor.Variable('value', 'real')
    log_normal = normal(value=value.log()).align(('loc', 'scale', 'value'))
    assert log_normal.dims == ('loc', 'scale', 'value')
    assert log_normal.shape == ('real', 'real', 'real')

    loc = funsor.Tensor(batch_dims, torch.randn(batch_shape))
    scale = funsor.Tensor(batch_dims, torch.randn(batch_shape).exp())
    value = funsor.Tensor(batch_dims, torch.randn(batch_shape).exp())
    actual = dist.LogNormal(loc, scale)(value)
    assert isinstance(actual, funsor.Tensor)
    assert actual.dims == batch_dims
    assert actual.shape == batch_shape

    expected = normal(loc, scale, value)
    assert isinstance(expected, funsor.Tensor)
    assert expected.dims == batch_dims
    assert expected.shape == batch_shape
    assert ((actual - expected) < 1e-5).all()


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
def test_gamma_density(batch_shape):
    batch_dims = ('x', 'y', 'z')[:len(batch_shape)]

    concentration = funsor.Tensor(batch_dims, torch.randn(batch_shape).exp())
    rate = funsor.Tensor(batch_dims, torch.randn(batch_shape).exp())
    value = funsor.Tensor(batch_dims, torch.randn(batch_shape).exp())
    actual = dist.Gamma(concentration, rate)(value)
    assert isinstance(actual, funsor.Tensor)
    assert actual.dims == batch_dims
    assert actual.shape == batch_shape

    @funsor.Pointwise
    def gamma(concentration, rate, value):
        return torch.distributions.Gamma(concentration, rate).log_prob(value)

    expected = gamma(concentration, rate, value)
    print(actual.data)
    print(expected.data)
    assert isinstance(expected, funsor.Tensor)
    assert expected.dims == batch_dims
    assert expected.shape == batch_shape
    assert ((actual.data - expected.data) < 1e-5).all()
