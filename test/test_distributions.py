import math
from collections import OrderedDict

import pyro
import pytest
import torch

import funsor
import funsor.distributions as dist
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.terms import Independent, Variable
from funsor.testing import assert_close, check_funsor, random_tensor
from funsor.torch import Tensor


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('eager', [False, True])
def test_beta_density(batch_shape, eager):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(), reals(), reals(), reals())
    def beta(concentration1, concentration0, value):
        return torch.distributions.Beta(concentration1, concentration0).log_prob(value)

    check_funsor(beta, {'concentration1': reals(), 'concentration0': reals(), 'value': reals()}, reals())

    concentration1 = Tensor(torch.randn(batch_shape).exp(), inputs)
    concentration0 = Tensor(torch.randn(batch_shape).exp(), inputs)
    value = Tensor(torch.rand(batch_shape), inputs)
    expected = beta(concentration1, concentration0, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    actual = dist.Beta(concentration1, concentration0, value) if eager else \
        dist.Beta(concentration1, concentration0, d)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy', 'generic'])
def test_bernoulli_probs_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(), reals(), reals())
    def bernoulli(probs, value):
        return torch.distributions.Bernoulli(probs).log_prob(value)

    check_funsor(bernoulli, {'probs': reals(), 'value': reals()}, reals())

    probs = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.rand(batch_shape).round(), inputs)
    expected = bernoulli(probs, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.BernoulliProbs(probs, value)
    elif syntax == 'lazy':
        actual = dist.BernoulliProbs(probs, d)(value=value)
    elif syntax == 'generic':
        actual = dist.Bernoulli(probs=probs)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy', 'generic'])
def test_bernoulli_logits_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(), reals(), reals())
    def bernoulli(logits, value):
        return torch.distributions.Bernoulli(logits=logits).log_prob(value)

    check_funsor(bernoulli, {'logits': reals(), 'value': reals()}, reals())

    logits = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.rand(batch_shape).round(), inputs)
    expected = bernoulli(logits, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.BernoulliLogits(logits, value)
    elif syntax == 'lazy':
        actual = dist.BernoulliLogits(logits, d)(value=value)
    elif syntax == 'generic':
        actual = dist.Bernoulli(logits=logits)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('eager', [False, True])
def test_binomial_density(batch_shape, eager):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))
    max_count = 10

    @funsor.torch.function(reals(), reals(), reals(), reals())
    def binomial(total_count, probs, value):
        return torch.distributions.Binomial(total_count, probs).log_prob(value)

    check_funsor(binomial, {'total_count': reals(), 'probs': reals(), 'value': reals()}, reals())

    value_data = random_tensor(inputs, bint(max_count)).data.float()
    total_count_data = value_data + random_tensor(inputs, bint(max_count)).data.float()
    value = Tensor(value_data, inputs)
    total_count = Tensor(total_count_data, inputs)
    probs = Tensor(torch.rand(batch_shape), inputs)
    expected = binomial(total_count, probs, value)
    check_funsor(expected, inputs, reals())

    m = Variable('value', reals())
    actual = dist.Binomial(total_count, probs, value) if eager else \
        dist.Binomial(total_count, probs, m)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


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
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_dirichlet_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(*event_shape), reals(*event_shape), reals())
    def dirichlet(concentration, value):
        return torch.distributions.Dirichlet(concentration).log_prob(value)

    check_funsor(dirichlet, {'concentration': reals(*event_shape), 'value': reals(*event_shape)}, reals())

    concentration = Tensor(torch.randn(batch_shape + event_shape).exp(), inputs)
    value_data = torch.rand(batch_shape + event_shape)
    value_data = value_data / value_data.sum(-1, keepdim=True)
    value = Tensor(value_data, inputs)
    expected = dirichlet(concentration, value)
    check_funsor(expected, inputs, reals())
    actual = dist.Dirichlet(concentration, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_dirichlet_multinomial_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))
    max_count = 10

    @funsor.torch.function(reals(*event_shape), reals(), reals(*event_shape), reals())
    def dirichlet_multinomial(concentration, total_count, value):
        return pyro.distributions.DirichletMultinomial(concentration, total_count).log_prob(value)

    check_funsor(dirichlet_multinomial, {'concentration': reals(*event_shape),
                                         'total_count': reals(),
                                         'value': reals(*event_shape)},
                 reals())

    concentration = Tensor(torch.randn(batch_shape + event_shape).exp(), inputs)
    value_data = torch.randint(0, max_count, size=batch_shape + event_shape).float()
    total_count_data = value_data.sum(-1) + torch.randint(0, max_count, size=batch_shape).float()
    value = Tensor(value_data, inputs)
    total_count = Tensor(total_count_data, inputs)
    expected = dirichlet_multinomial(concentration, total_count, value)
    check_funsor(expected, inputs, reals())
    actual = dist.DirichletMultinomial(concentration, total_count, value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)


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


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('event_shape', [(1,), (4,), (5,)], ids=str)
def test_multinomial_density(batch_shape, event_shape):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))
    max_count = 10

    @funsor.torch.function(reals(), reals(*event_shape), reals(*event_shape), reals())
    def multinomial(total_count, probs, value):
        total_count = total_count.max().item()
        return torch.distributions.Multinomial(total_count, probs).log_prob(value)

    check_funsor(multinomial, {'total_count': reals(), 'probs': reals(*event_shape), 'value': reals(*event_shape)},
                 reals())

    probs_data = torch.rand(batch_shape + event_shape)
    probs_data = probs_data / probs_data.sum(-1, keepdim=True)
    probs = Tensor(probs_data, inputs)
    value_data = torch.randint(0, max_count, size=batch_shape + event_shape).float()
    total_count_data = value_data.sum(-1) + torch.randint(0, max_count, size=batch_shape).float()
    value = Tensor(value_data, inputs)
    total_count = Tensor(total_count_data, inputs)
    expected = multinomial(total_count, probs, value)
    check_funsor(expected, inputs, reals())
    actual = dist.Multinomial(total_count, probs, value)
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

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('value', reals()), scale, loc)
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

    expected = dist.Normal(loc, scale, value)
    assert isinstance(expected, Tensor)
    check_funsor(expected, inputs, reals())

    g = dist.Normal(Variable('loc', reals()), scale, 'value')
    assert isinstance(g, Contraction)
    actual = g(loc=loc, value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-4)


NORMAL_AFFINE_TESTS = [
    'dist.Normal(x+2, scale, y+2)',
    'dist.Normal(y, scale, x)',
    'dist.Normal(x - y, scale, 0)',
    'dist.Normal(0, scale, y - x)',
    'dist.Normal(2 * x - y, scale, x)',
    # TODO should we expect these to work without correction terms?
    'dist.Normal(0, 1, (x - y) / scale) - scale.log()',
    'dist.Normal(2 * y, 2 * scale, 2 * x) + math.log(2)',
]


@pytest.mark.parametrize('expr', NORMAL_AFFINE_TESTS)
def test_normal_affine(expr):

    scale = Tensor(torch.tensor(0.3), OrderedDict())
    x = Variable('x', reals())
    y = Variable('y', reals())

    expected = dist.Normal(x, scale, y)
    actual = eval(expr)

    assert isinstance(actual, Contraction)
    assert dict(actual.inputs) == dict(expected.inputs), (actual.inputs, expected.inputs)

    for ta, te in zip(actual.terms, expected.terms):
        assert_close(ta.align(tuple(te.inputs)), te)


def test_normal_independent():
    loc = random_tensor(OrderedDict(), reals(2))
    scale = random_tensor(OrderedDict(), reals(2)).exp()
    fn = dist.Normal(loc['i'], scale['i'], value='z_i')
    assert fn.inputs['z_i'] == reals()
    d = Independent(fn, 'z', 'i', 'z_i')
    assert d.inputs['z'] == reals(2)
    sample = d.sample(frozenset(['z']))
    assert isinstance(sample, Independent)
    assert isinstance(sample.fn, Contraction)
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
    assert isinstance(g, Contraction)
    actual = g(value=value)
    check_funsor(actual, inputs, reals())

    assert_close(actual, expected, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('syntax', ['eager', 'lazy'])
def test_poisson_probs_density(batch_shape, syntax):
    batch_dims = ('i', 'j', 'k')[:len(batch_shape)]
    inputs = OrderedDict((k, bint(v)) for k, v in zip(batch_dims, batch_shape))

    @funsor.torch.function(reals(), reals(), reals())
    def poisson(rate, value):
        return torch.distributions.Poisson(rate).log_prob(value)

    check_funsor(poisson, {'rate': reals(), 'value': reals()}, reals())

    rate = Tensor(torch.rand(batch_shape), inputs)
    value = Tensor(torch.randn(batch_shape).exp().round(), inputs)
    expected = poisson(rate, value)
    check_funsor(expected, inputs, reals())

    d = Variable('value', reals())
    if syntax == 'eager':
        actual = dist.Poisson(rate, value)
    elif syntax == 'lazy':
        actual = dist.Poisson(rate, d)(value=value)
    check_funsor(actual, inputs, reals())
    assert_close(actual, expected)
