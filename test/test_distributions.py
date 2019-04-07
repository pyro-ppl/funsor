from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import pytest
import torch

import funsor
import funsor.distributions as dist
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.joint import Joint
from funsor.terms import Independent, Variable
from funsor.testing import assert_close, check_funsor, random_tensor
from funsor.torch import Tensor


def test_categorical_defaults():
    probs = Variable('probs', reals(3))
    value = Variable('value', bint(3))
    assert dist.Categorical(probs) is dist.Categorical(probs, value)


@pytest.mark.parametrize('size', [4])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_categorical_density(size, batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.of_shape(reals(size), bint(size))
    def categorical(probs, value):
        return probs[value].log()

    check_funsor(categorical, {'probs': reals(size), 'value': bint(size)}, reals())

    probs_data = torch.randn(batch_shape + (size,)).exp()
    probs_data /= probs_data.sum(-1, keepdim=True)
    probs = Tensor(probs_data, inputs)
    value = random_tensor(inputs, bint(size))
    expected = categorical(probs, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Categorical(probs, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


def test_delta_defaults():
    v = Variable('v', reals())
    log_density = Variable('log_density', reals())
    assert isinstance(dist.Delta(v, log_density), dist.Delta)
    value = Variable('value', reals())
    assert dist.Delta(v, log_density, 'value') is dist.Delta(v, log_density, value)


@pytest.mark.parametrize('event_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_delta_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(*event_shape), reals(), reals(*event_shape), reals())
    def delta(v, log_density, value):
        eq = (v == value)
        for _ in range(len(event_shape)):
            eq = eq.all(dim=-1)
        return eq.type(v.dtype).log() + log_density

    check_funsor(delta, {'v': reals(*event_shape),
                         'log_density': reals(),
                         'value': reals(*event_shape)}, reals())

    v = Tensor(torch.randn(batch_shape + event_shape), inputs)
    log_density = Tensor(torch.randn(batch_shape).exp(), inputs)
    for value in [v, Tensor(torch.randn(batch_shape + event_shape), inputs)]:
        expected = delta(v, log_density, value)
        check_funsor(expected, inputs, reals())

        actual = dist.Delta(v, log_density, value)
        check_funsor(actual, inputs, reals())
        assert_close(actual, expected)


def test_delta_delta():
    v = Variable('v', reals(2))
    point = Tensor(torch.randn(2))
    log_density = Tensor(torch.tensor(0.5))
    d = dist.Delta(point, log_density, v)
    assert d is Delta('v', point, log_density)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_lognormal_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(), reals(), reals(), reals())
    def log_normal(loc, scale, value):
        return torch.distributions.LogNormal(loc, scale).log_prob(value)

    check_funsor(log_normal, {'loc': reals(), 'scale': reals(), 'value': reals()}, reals())

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape).exp(), inputs)
    expected = log_normal(loc, scale, value)
    check_funsor(expected, inputs, reals())

    actual = dist.LogNormal(loc, scale, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


def test_normal_defaults():
    loc = Variable('loc', reals())
    scale = Variable('scale', reals())
    value = Variable('value', reals())
    assert dist.Normal(loc, scale) is dist.Normal(loc, scale, value)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

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


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_1(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(loc, scale, 'value')
    assert isinstance(g, Joint)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_2(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('value', reals()), scale, loc)
    assert isinstance(g, Joint)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_3(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('loc', reals()), scale, 'value')
    assert isinstance(g, Joint)
    actual = g(loc=loc, value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


NORMAL_AFFINE_TESTS = [
    'dist.Normal(x+2, scale, y+2)',
    'dist.Normal(y, scale, x)',
    'dist.Normal(2 * y, 2 * scale, 2 * x)',
    'dist.Normal(x - y, scale, 0)',
    'dist.Normal(0, scale, y - x)',
    'dist.Normal(0, 1, (x - y) / scale)',
    'dist.Normal(2 * x - y, scale, x)',
]


@pytest.mark.parametrize('expr', NORMAL_AFFINE_TESTS)
def test_normal_affine(expr):

    scale = Tensor(torch.tensor(0.3), OrderedDict())
    x = Variable('x', reals())
    y = Variable('y', reals())

    expected = dist.Normal(x, scale, y)
    actual = eval(expr)

    assert isinstance(actual, Gaussian)
    assert dict(actual.inputs) == dict(expected.inputs), (actual.inputs, expected.inputs)
    actual = actual.align(tuple(expected.inputs))
    if isinstance(actual, Gaussian):
        assert_close(actual, expected)


@pytest.mark.parametrize('expr', NORMAL_AFFINE_TESTS)
def test_normal_affine_mvn(expr):

    scale = Tensor(torch.tensor(1.), OrderedDict())
    assert isinstance(scale, Tensor)
    loc = Tensor(torch.tensor([0., 0.]), OrderedDict())

    scale_tril = Tensor(torch.tensor([[1., 0.], [0., 1.]]), OrderedDict())

    x = Variable('x', reals())
    assert isinstance(x, Variable)
    y = Variable('y', reals())
    assert isinstance(y, Variable)

    data = torch.randn(2)

    expected = dist.MultivariateNormal(loc, scale_tril, 'value')
    actual = eval(expr)

    assert isinstance(actual, Joint)
    assert isinstance(expected, Joint)

    assert_close(actual.gaussian.loc, expected.gaussian.loc)
    assert_close(actual.gaussian.precision, expected.gaussian.precision)
    assert_close(actual(x=data[0], y=data[1]), expected(value=data))


def test_normal_independent():
    loc = random_tensor(OrderedDict(), reals(2))
    scale = random_tensor(OrderedDict(), reals(2)).exp()
    fn = dist.Normal(loc['i'], scale['i'], value='z')
    assert fn.inputs['z'] == reals()
    d = Independent(fn, 'z', 'i')
    assert d.inputs['z'] == reals(2)
    sample = d.sample(frozenset(['z']))
    assert isinstance(sample, Joint)
    assert sample.inputs['z'] == reals(2)


def test_mvn_defaults():
    loc = Variable('loc', reals(3))
    scale_tril = Variable('scale', reals(3, 3))
    value = Variable('value', reals(3))
    assert dist.MultivariateNormal(loc, scale_tril) is dist.MultivariateNormal(loc, scale_tril, value)


def _random_scale_tril(shape):
    data = torch.randn(shape)
    return torch.distributions.transform_to(torch.distributions.constraints.lower_cholesky)(data)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_mvn_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(3), reals(3, 3), reals(3), reals())
    def mvn(loc, scale_tril, value):
        return torch.distributions.MultivariateNormal(loc, scale_tril=scale_tril).log_prob(value)

    check_funsor(mvn, {'loc': reals(3), 'scale_tril': reals(3, 3), 'value': reals(3)}, reals())

    loc = Tensor(torch.randn(batch_shape + (3,)), inputs)
    scale_tril = Tensor(_random_scale_tril(batch_shape + (3, 3)), inputs)
    value = Tensor(torch.randn(batch_shape + (3,)), inputs)
    expected = mvn(loc, scale_tril, value)
    check_funsor(expected, inputs, reals())

    actual = dist.MultivariateNormal(loc, scale_tril, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_mvn_gaussian(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(torch.randn(batch_shape + (3,)), inputs)
    scale_tril = Tensor(_random_scale_tril(batch_shape + (3, 3)), inputs)
    value = Tensor(torch.randn(batch_shape + (3,)), inputs)

    expected = dist.MultivariateNormal(loc, scale_tril, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.MultivariateNormal(loc, scale_tril, 'value')
    assert isinstance(g, Joint)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-3, rtol=1e-4)
