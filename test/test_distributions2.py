# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import pyro
import pytest
import torch

import funsor
import funsor.distributions2 as dist
from funsor.cnf import Contraction, GaussianMixture
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.interpreter import interpretation, reinterpret
from funsor.pyro.convert import dist_to_funsor
from funsor.tensor import Einsum, Tensor
from funsor.terms import Independent, Variable, lazy
from funsor.testing import assert_close, check_funsor, random_mvn, random_tensor
from funsor.util import get_backend

funsor.set_backend("torch")

@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_beta_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def beta(concentration1, concentration0, value):
        return torch.distributions.Beta(concentration1, concentration0).log_prob(value)

    check_funsor(beta, {'concentration1': reals(), 'concentration0': reals(), 'value': reals()}, reals())

    concentration1 = Tensor(torch.randn(batch_shape).exp(), inputs)
    concentration0 = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.rand(batch_shape), inputs)
    expected = beta(concentration1, concentration0, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Beta(concentration1, concentration0, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_bernoulli_probs_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals())
    def bernoulli(probs, value):
        return torch.distributions.Bernoulli(probs=probs).log_prob(value)

    check_funsor(bernoulli, {'probs': reals(), 'value': reals()}, reals())

    probs = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.rand(batch_shape).round(), inputs)
    expected = bernoulli(probs, value)
    check_funsor(expected, inputs, reals())

    actual = dist.BernoulliProbs(probs, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_bernoulli_logits_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals())
    def bernoulli(logits, value):
        return torch.distributions.Bernoulli(logits=logits).log_prob(value)

    check_funsor(bernoulli, {'logits': reals(), 'value': reals()}, reals())

    logits = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.rand(batch_shape).round(), inputs)
    expected = bernoulli(logits, value)
    check_funsor(expected, inputs, reals())

    actual = dist.BernoulliLogits(logits, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('size', [4])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_categorical_probs_density(size, batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(size), bint(size), reals())
    def categorical_probs(probs, value):
        return torch.distributions.Categorical(probs=probs).log_prob(value)

    check_funsor(categorical_probs, {'probs': reals(size), 'value': bint(size)}, reals())

    probs_data = torch.randn(batch_shape + (size,)).exp()
    probs_data /= probs_data.sum(-1, keepdim=True)
    probs = Tensor(probs_data, inputs)
    value = random_tensor(inputs, bint(size))
    expected = categorical_probs(probs, value)
    check_funsor(expected, inputs, reals())

    actual = dist.CategoricalProbs(probs, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('size', [4])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_categorical_logits_density(size, batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(size), bint(size), reals())
    def categorical_logits(logits, value):
        return torch.distributions.Categorical(logits=logits).log_prob(value)

    check_funsor(categorical_logits, {'logits': reals(size), 'value': bint(size)}, reals())

    logits_data = torch.randn(batch_shape + (size,))
    logits_data /= logits_data.sum(-1, keepdim=True)
    logits = Tensor(logits_data, inputs)
    value = random_tensor(inputs, bint(size))
    expected = categorical_logits(logits, value)
    check_funsor(expected, inputs, reals())

    actual = dist.CategoricalLogits(logits, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_dirichlet_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(*event_shape), reals(*event_shape), reals())
    def dirichlet(concentration, value):
        return torch.distributions.Dirichlet(concentration).log_prob(value)

    check_funsor(dirichlet, {'concentration': reals(*event_shape), 'value': reals(*event_shape)}, reals())

    concentration = Tensor(torch.randn(batch_shape + event_shape).exp(), inputs)
    value_data = torch.rand(batch_shape + event_shape)
    value_data = value_data / value_data.sum(-1, keepdim=True)
    value = Tensor(value_data, inputs)
    expected = dirichlet(concentration, value)
    check_funsor(expected, inputs, reals())
    actual = dist.Dirichlet(concentration, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def normal(loc, scale, value):
        return torch.distributions.Normal(loc, scale).log_prob(value)

    check_funsor(normal, {'loc': reals(), 'scale': reals(), 'value': reals()}, reals())

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape), inputs)
    expected = normal(loc, scale, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Normal(loc, scale, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_poisson_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals())
    def poisson(rate, value):
        return torch.distributions.Poisson(rate).log_prob(value)

    check_funsor(poisson, {'rate': reals(), 'value': reals()}, reals())

    rate = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.randn(batch_shape).exp().round(), inputs)
    expected = poisson(rate, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Poisson(rate, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_gamma_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def gamma(concentration, rate, value):
        return torch.distributions.Gamma(concentration, rate).log_prob(value)

    check_funsor(gamma, {'concentration': reals(), 'rate': reals(), 'value': reals()}, reals())

    concentration = Tensor(torch.rand(batch_shape), inputs)
    rate = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.randn(batch_shape).exp(), inputs)
    expected = gamma(concentration, rate, value)
    check_funsor(expected, inputs, reals())

    actual = dist.Gamma(concentration, rate, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_von_mises_density(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.function(reals(), reals(), reals(), reals())
    def von_mises(loc, concentration, value):
        return pyro.distributions.VonMises(loc, concentration).log_prob(value)

    check_funsor(von_mises, {'concentration': reals(), 'loc': reals(), 'value': reals()}, reals())

    concentration = Tensor(torch.rand(batch_shape), inputs)
    loc = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.randn(batch_shape).abs(), inputs)
    expected = von_mises(loc, concentration, value)
    check_funsor(expected, inputs, reals())

    actual = dist.VonMises(loc, concentration, name='value')(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.xfail(reason="output shape inference not implemented")
@pytest.mark.parametrize("event_shape", [
    (),  # (5,), (4, 3),
], ids=str)
@pytest.mark.parametrize("batch_shape", [
    (), (2,), (2, 3),
], ids=str)
def test_normal_funsor_normal(batch_shape, event_shape):
    loc = torch.randn(batch_shape + event_shape)
    scale = torch.randn(batch_shape + event_shape).exp()
    d = pyro.distributions.Normal(loc, scale).to_event(len(event_shape))
    value = d.sample()
    name_to_dim = OrderedDict(
        (f'{v}', v) for v in range(-len(batch_shape), 0) if batch_shape[v] > 1)
    dim_to_name = OrderedDict((v, k) for k, v in name_to_dim.items())
    f = funsor.to_funsor(d, reals(), dim_to_name=dim_to_name)
    d2 = funsor.to_data(f, name_to_dim=name_to_dim)
    assert type(d) == type(d2)
    assert d.batch_shape == d2.batch_shape
    assert d.event_shape == d2.event_shape
    expected_log_prob = d.log_prob(value)
    actual_log_prob = d2.log_prob(value)
    assert_close(actual_log_prob, expected_log_prob)
    expected_funsor_log_prob = funsor.to_funsor(actual_log_prob, reals(), dim_to_name)
    actual_funsor_log_prob = f(value=funsor.to_funsor(value, reals(*event_shape), dim_to_name))
    assert_close(actual_funsor_log_prob, expected_funsor_log_prob)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
def test_normal_gaussian_1(batch_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    loc = Tensor(torch.randn(batch_shape), inputs)
    scale = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.randn(batch_shape), inputs)

    expected = dist.Normal(loc, scale)(value=value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(loc, scale)(value='value')
    assert isinstance(g, Contraction)
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

    expected = dist.Normal(loc, scale)(value=value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('value', reals()), scale, 'loc')(loc=loc)
    assert isinstance(g, Contraction)
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

    expected = dist.Normal(loc, scale)(value=value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('loc', reals()), scale)(value='value')
    assert isinstance(g, Contraction)
    actual = g(loc=loc, value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


NORMAL_AFFINE_TESTS = [
    'dist.Normal(x+2, scale)(value=y+2)',
    'dist.Normal(y, scale)(value=x)',
    'dist.Normal(x - y, scale)(value=0)',
    'dist.Normal(0, scale)(value=y - x)',
    'dist.Normal(2 * x - y, scale)(value=x)',
    'dist.Normal(0, 1)(value=(x - y) / scale) - scale.log()',
    'dist.Normal(2 * y, 2 * scale)(value=2 * x) + math.log(2)',
]


@pytest.mark.parametrize('expr', NORMAL_AFFINE_TESTS)
def test_normal_affine(expr):

    scale = Tensor(torch.tensor(0.3), OrderedDict())
    x = Variable('x', reals())
    y = Variable('y', reals())

    expected = dist.Normal(x, scale)(value=y)
    actual = eval(expr)

    assert isinstance(actual, Contraction)
    assert dict(actual.inputs) == dict(expected.inputs), (actual.inputs, expected.inputs)

    for ta, te in zip(actual.terms, expected.terms):
        assert_close(ta.align(tuple(te.inputs)), te)


def test_normal_independent():
    loc = random_tensor(OrderedDict(), reals(2))
    scale = random_tensor(OrderedDict(), reals(2)).exp()
    fn = dist.Normal(loc['i'], scale['i'], 'z_i')
    assert fn.inputs['z_i'] == reals()
    d = Independent(fn, 'z', 'i', 'z_i')
    assert d.inputs['z'] == reals(2)
    sample = d.sample(frozenset(['z']))
    assert isinstance(sample, Contraction)
    assert sample.inputs['z'] == reals(2)
