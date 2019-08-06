from collections import OrderedDict

import pyro.distributions as dist
import pytest
import torch
from pyro.distributions.torch_distribution import MaskedDistribution

from funsor.domains import bint, reals
from funsor.pyro.convert import dist_to_funsor, funsor_to_tensor, mvn_to_funsor, tensor_to_funsor
from funsor.terms import Funsor
from funsor.testing import assert_close
from funsor.torch import Tensor

EVENT_SHAPES = [(), (1,), (5,), (4, 3)]
BATCH_SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]
REAL_SIZES = [(1,), (1, 1), (1, 1, 1), (1, 2), (2, 1), (2, 3), (3, 1, 2)]


@pytest.mark.parametrize("event_shape,event_output", [
    (shape, size)
    for shape in EVENT_SHAPES
    for size in range(len(shape))
], ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_tensor_funsor_tensor(batch_shape, event_shape, event_output):
    event_inputs = ("foo", "bar", "baz")[:len(event_shape) - event_output]
    t = torch.randn(batch_shape + event_shape)
    f = tensor_to_funsor(t, event_inputs, event_output)
    t2 = funsor_to_tensor(f, t.dim(), event_inputs)
    assert_close(t2, t)


@pytest.mark.parametrize("event_sizes", REAL_SIZES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_mvn_to_funsor(batch_shape, event_shape, event_sizes):
    event_size = sum(event_sizes)
    loc = torch.randn(batch_shape + event_shape + (event_size,))
    cov = torch.randn(batch_shape + event_shape + (event_size, 2 * event_size))
    cov = cov.matmul(cov.transpose(-1, -2))
    mvn = dist.MultivariateNormal(loc, cov)
    int_inputs = OrderedDict((k, bint(size)) for k, size in zip("abc", event_shape))
    real_inputs = OrderedDict((k, reals(size)) for k, size in zip("xyz", event_sizes))

    f = mvn_to_funsor(mvn, tuple(int_inputs), real_inputs)
    assert isinstance(f, Funsor)
    for k, d in int_inputs.items():
        if d.num_elements == 1:
            assert d not in f.inputs
        else:
            assert k in f.inputs
            assert f.inputs[k] == d
    for k, d in real_inputs.items():
        assert k in f.inputs
        assert f.inputs[k] == d

    value = mvn.sample()
    subs = {}
    beg = 0
    for k, d in real_inputs.items():
        end = beg + d.num_elements
        subs[k] = tensor_to_funsor(value[..., beg:end], tuple(int_inputs), 1)
        beg = end
    actual_log_prob = f(**subs)
    expected_log_prob = tensor_to_funsor(mvn.log_prob(value), tuple(int_inputs))
    assert_close(actual_log_prob, expected_log_prob, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("cardinality", [2, 3, 5])
def test_dist_to_funsor_categorical(batch_shape, cardinality):
    logits = torch.randn(batch_shape + (cardinality,))
    logits -= logits.logsumexp(dim=-1, keepdim=True)
    d = dist.Categorical(logits=logits)
    f = dist_to_funsor(d)
    assert isinstance(f, Tensor)
    expected = tensor_to_funsor(logits, ("value",))
    assert_close(f, expected)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_bernoulli(batch_shape):
    logits = torch.randn(batch_shape)
    d = dist.Bernoulli(logits=logits)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample()
    actual_log_prob = f(value=tensor_to_funsor(value))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_normal(batch_shape):
    loc = torch.randn(batch_shape)
    scale = torch.randn(batch_shape).exp()
    d = dist.Normal(loc, scale)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample()
    actual_log_prob = f(value=tensor_to_funsor(value))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_size", [2, 3, 5])
def test_dist_to_funsor_mvn(batch_shape, event_size):
    loc = torch.randn(batch_shape + (event_size,))
    cov = torch.randn(batch_shape + (event_size, 2 * event_size))
    cov = cov.matmul(cov.transpose(-1, -2))
    scale_tril = torch.cholesky(cov)
    d = dist.MultivariateNormal(loc, scale_tril=scale_tril)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample()
    actual_log_prob = f(value=tensor_to_funsor(value, event_output=1))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("event_shape", [(), (6,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_independent(batch_shape, event_shape):
    loc = torch.randn(batch_shape + event_shape)
    scale = torch.randn(batch_shape + event_shape).exp()
    d = dist.Normal(loc, scale).to_event(len(event_shape))
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample()
    funsor_value = tensor_to_funsor(value, event_output=len(event_shape))
    actual_log_prob = f(value=funsor_value)
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_masked(batch_shape):
    loc = torch.randn(batch_shape)
    scale = torch.randn(batch_shape).exp()
    mask = torch.bernoulli(torch.full(batch_shape, 0.5)).byte()
    d = dist.Normal(loc, scale).mask(mask)
    assert isinstance(d, MaskedDistribution)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample()
    actual_log_prob = f(value=tensor_to_funsor(value))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)
