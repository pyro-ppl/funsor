# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pyro.distributions as dist
import pytest
import torch
from pyro.distributions.torch_distribution import MaskedDistribution

from funsor.domains import Bint, Reals
from funsor.pyro.convert import (
    dist_to_funsor,
    funsor_to_cat_and_mvn,
    funsor_to_mvn,
    funsor_to_tensor,
    matrix_and_mvn_to_funsor,
    mvn_to_funsor,
    tensor_to_funsor,
)
from funsor.tensor import Tensor
from funsor.terms import Funsor
from funsor.testing import assert_close, random_mvn

EVENT_SHAPES = [(), (1,), (5,), (4, 3)]
BATCH_SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]
REAL_SIZES = [(1,), (1, 1), (1, 1, 1), (1, 2), (2, 1), (2, 3), (3, 1, 2)]


@pytest.mark.parametrize(
    "event_shape,event_output",
    [(shape, size) for shape in EVENT_SHAPES for size in range(len(shape))],
    ids=str,
)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_tensor_funsor_tensor(batch_shape, event_shape, event_output):
    event_inputs = ("foo", "bar", "baz")[: len(event_shape) - event_output]
    t = torch.randn(batch_shape + event_shape)
    f = tensor_to_funsor(t, event_inputs, event_output)
    t2 = funsor_to_tensor(f, t.dim(), event_inputs)
    assert_close(t2, t)


@pytest.mark.parametrize("event_sizes", REAL_SIZES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_mvn_to_funsor(batch_shape, event_shape, event_sizes):
    event_size = sum(event_sizes)
    mvn = random_mvn(batch_shape + event_shape, event_size)
    int_inputs = OrderedDict((k, Bint[size]) for k, size in zip("abc", event_shape))
    real_inputs = OrderedDict((k, Reals[size]) for k, size in zip("xyz", event_sizes))

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


@pytest.mark.parametrize("x_size", [1, 2, 3])
@pytest.mark.parametrize("y_size", [1, 2, 3])
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_matrix_and_mvn_to_funsor(batch_shape, event_shape, x_size, y_size):
    matrix = torch.randn(batch_shape + event_shape + (x_size, y_size))
    y_mvn = random_mvn(batch_shape + event_shape, y_size)
    xy_mvn = random_mvn(batch_shape + event_shape, x_size + y_size)
    int_inputs = OrderedDict((k, Bint[size]) for k, size in zip("abc", event_shape))
    real_inputs = OrderedDict([("x", Reals[x_size]), ("y", Reals[y_size])])

    f1 = matrix_and_mvn_to_funsor(matrix, y_mvn, tuple(int_inputs), "x", "y")
    f2 = mvn_to_funsor(xy_mvn, tuple(int_inputs), real_inputs)
    f = f1 + f2
    assert isinstance(f, Funsor)
    for k, d in int_inputs.items():
        if d.num_elements == 1:
            assert d not in f.inputs
        else:
            assert k in f.inputs
            assert f.inputs[k] == d
    assert f.inputs["x"] == Reals[x_size]
    assert f.inputs["y"] == Reals[y_size]

    xy = torch.randn(x_size + y_size)
    x, y = xy[:x_size], xy[x_size:]
    y_pred = x.unsqueeze(-2).matmul(matrix).squeeze(-2)
    actual_log_prob = f(x=x, y=y)

    expected_log_prob = f1(x=x, y=y) + f2(x=x, y=y)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=1e-4)

    expected_log_prob = tensor_to_funsor(
        xy_mvn.log_prob(xy) + y_mvn.log_prob(y - y_pred), tuple(int_inputs)
    )
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_size", [1, 2, 3])
@pytest.mark.parametrize("y_size", [1, 2, 3])
def test_matrix_and_mvn_to_funsor_diag(batch_shape, x_size, y_size):
    matrix = torch.randn(batch_shape + (x_size, y_size))
    loc = torch.randn(batch_shape + (y_size,))
    scale = torch.randn(batch_shape + (y_size,)).exp()

    normal = dist.Normal(loc, scale).to_event(1)
    actual = matrix_and_mvn_to_funsor(matrix, normal)

    mvn = dist.MultivariateNormal(loc, scale_tril=scale.diag_embed())
    expected = matrix_and_mvn_to_funsor(matrix, mvn)

    y = tensor_to_funsor(torch.randn(batch_shape + (y_size,)), (), 1)
    actual_like = actual(value_y=y)
    expected_like = expected(value_y=y)
    assert_close(actual_like, expected_like, atol=1e-4, rtol=1e-4)

    x = tensor_to_funsor(torch.randn(batch_shape + (x_size,)), (), 1)
    actual_norm = actual_like(value_x=x)
    expected_norm = expected_like(value_x=x)
    assert_close(actual_norm, expected_norm, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("real_size", [1, 2, 3])
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_funsor_to_mvn(batch_shape, event_shape, real_size):
    expected = random_mvn(batch_shape + event_shape, real_size)
    event_dims = tuple("abc"[: len(event_shape)])
    ndims = len(expected.batch_shape)

    funsor_ = dist_to_funsor(expected, event_dims)(value="value")
    assert isinstance(funsor_, Funsor)

    actual = funsor_to_mvn(funsor_, ndims, event_dims)
    assert isinstance(actual, dist.MultivariateNormal)
    assert actual.batch_shape == expected.batch_shape
    assert_close(actual.loc, expected.loc, atol=1e-3, rtol=None)
    assert_close(
        actual.precision_matrix, expected.precision_matrix, atol=1e-3, rtol=None
    )


@pytest.mark.parametrize("int_size", [2, 3])
@pytest.mark.parametrize("real_size", [1, 2, 3])
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_funsor_to_cat_and_mvn(batch_shape, event_shape, int_size, real_size):
    logits = torch.randn(batch_shape + event_shape + (int_size,))
    expected_cat = dist.Categorical(logits=logits)
    expected_mvn = random_mvn(batch_shape + event_shape + (int_size,), real_size)
    event_dims = tuple("abc"[: len(event_shape)]) + ("component",)
    ndims = len(expected_cat.batch_shape)

    funsor_ = tensor_to_funsor(logits, event_dims) + dist_to_funsor(
        expected_mvn, event_dims
    )(value="value")
    assert isinstance(funsor_, Funsor)

    actual_cat, actual_mvn = funsor_to_cat_and_mvn(funsor_, ndims, event_dims)
    assert isinstance(actual_cat, dist.Categorical)
    assert isinstance(actual_mvn, dist.MultivariateNormal)
    assert actual_cat.batch_shape == expected_cat.batch_shape
    assert actual_mvn.batch_shape == expected_mvn.batch_shape
    assert_close(actual_cat.logits, expected_cat.logits, atol=1e-4, rtol=None)
    assert_close(actual_mvn.loc, expected_mvn.loc, atol=1e-4, rtol=None)
    assert_close(
        actual_mvn.precision_matrix, expected_mvn.precision_matrix, atol=1e-4, rtol=None
    )


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
    assert_close(actual_log_prob, expected_log_prob, rtol=1e-5)


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
    assert_close(actual_log_prob, expected_log_prob, rtol=1e-5)


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
