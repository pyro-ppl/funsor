# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import numpyro.distributions as dist
import pytest

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.numpyro.convert import (
    AffineNormal,
    dist_to_funsor,
    funsor_to_cat_and_mvn,
    funsor_to_mvn,
    funsor_to_tensor,
    matrix_and_mvn_to_funsor,
    mvn_to_funsor,
    tensor_to_funsor
)
from funsor.tensor import Tensor
from funsor.terms import Funsor, Variable
from funsor.testing import assert_close, ones, randn, random_mvn, random_tensor
from funsor.util import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "jax", reason="only test for jax backend")

EVENT_SHAPES = [(), (1,), (5,), (4, 3)]
BATCH_SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]
REAL_SIZES = [(1,), (1, 1), (1, 1, 1), (1, 2), (2, 1), (2, 3), (3, 1, 2)]


def get_rng_key(seed=0):
    return np.array([0, seed], dtype=np.uint32)


def get_unexpanded_dist(d):
    if isinstance(d, dist.ExpandedDistribution):
        return get_unexpanded_dist(d.base_dist)
    return d


@pytest.mark.parametrize("event_shape,event_output", [
    (shape, size)
    for shape in EVENT_SHAPES
    for size in range(len(shape))
], ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_tensor_funsor_tensor(batch_shape, event_shape, event_output):
    event_inputs = ("foo", "bar", "baz")[:len(event_shape) - event_output]
    t = randn(batch_shape + event_shape)
    f = tensor_to_funsor(t, event_inputs, event_output)
    t2 = funsor_to_tensor(f, len(t.shape), event_inputs)
    assert_close(t2, t)


@pytest.mark.parametrize("event_sizes", REAL_SIZES, ids=str)
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_mvn_to_funsor(batch_shape, event_shape, event_sizes):
    event_size = sum(event_sizes)
    mvn = random_mvn(batch_shape + event_shape, event_size)
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

    value = mvn.sample(get_rng_key())
    subs = {}
    beg = 0
    for k, d in real_inputs.items():
        end = beg + d.num_elements
        subs[k] = tensor_to_funsor(value[..., beg:end], tuple(int_inputs), 1)
        beg = end
    actual_log_prob = f(**subs)
    expected_log_prob = tensor_to_funsor(mvn.log_prob(value), tuple(int_inputs))
    assert_close(actual_log_prob, expected_log_prob, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("x_size", [1, 2])
@pytest.mark.parametrize("y_size", [1, 3])
@pytest.mark.parametrize("matrix_shape,loc_shape,scale_shape,x_shape,y_shape", [
    ((), (), (), (), ()),
    ((4,), (4,), (4,), (4,), (4,)),
    ((4, 5), (4, 5), (4, 5), (4, 5), (4, 5)),
    ((4,), (), (), (), ()),
    ((), (4,), (), (), ()),
    ((), (), (4,), (), ()),
    ((), (), (), (4,), ()),
    ((), (), (), (), (4,)),
], ids=str)
def test_affine_normal(matrix_shape, loc_shape, scale_shape, x_shape, y_shape,
                       x_size, y_size):

    def _rand(batch_shape, *event_shape):
        inputs = OrderedDict(zip("abcdef", map(bint, reversed(batch_shape))))
        return random_tensor(inputs, reals(*event_shape))

    matrix = _rand(matrix_shape, x_size, y_size)
    loc = _rand(loc_shape, y_size)
    scale = _rand(scale_shape, y_size).exp()
    value_x = _rand(x_shape, x_size)
    value_y = _rand(y_shape, y_size)

    f = AffineNormal(matrix, loc, scale,
                     Variable("x", reals(x_size)),
                     Variable("y", reals(y_size)))
    assert isinstance(f, AffineNormal)

    # Evaluate via two different patterns.
    expected = f(x=value_x)(y=value_y)
    actual = f(y=value_y)(x=value_x)
    assert_close(actual, expected, atol=1e-5, rtol=2e-4)


@pytest.mark.parametrize("x_size", [1, 2, 3])
@pytest.mark.parametrize("y_size", [1, 2, 3])
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_matrix_and_mvn_to_funsor(batch_shape, event_shape, x_size, y_size):
    matrix = randn(batch_shape + event_shape + (x_size, y_size))
    y_mvn = random_mvn(batch_shape + event_shape, y_size)
    xy_mvn = random_mvn(batch_shape + event_shape, x_size + y_size)
    int_inputs = OrderedDict((k, bint(size)) for k, size in zip("abc", event_shape))
    real_inputs = OrderedDict([("x", reals(x_size)), ("y", reals(y_size))])

    f = (matrix_and_mvn_to_funsor(matrix, y_mvn, tuple(int_inputs), "x", "y") +
         mvn_to_funsor(xy_mvn, tuple(int_inputs), real_inputs))
    assert isinstance(f, Funsor)
    for k, d in int_inputs.items():
        if d.num_elements == 1:
            assert d not in f.inputs
        else:
            assert k in f.inputs
            assert f.inputs[k] == d
    assert f.inputs["x"] == reals(x_size)
    assert f.inputs["y"] == reals(y_size)

    xy = randn(x_size + y_size)
    x, y = xy[:x_size], xy[x_size:]
    y_pred = ops.matmul(ops.unsqueeze(x, -2), matrix).squeeze(-2)
    actual_log_prob = f(x=x, y=y)
    expected_log_prob = tensor_to_funsor(
        xy_mvn.log_prob(xy) + y_mvn.log_prob(y - y_pred), tuple(int_inputs))
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_size", [1, 2, 3])
@pytest.mark.parametrize("y_size", [1, 2, 3])
def test_matrix_and_mvn_to_funsor_diag(batch_shape, x_size, y_size):
    matrix = randn(batch_shape + (x_size, y_size))
    loc = randn(batch_shape + (y_size,))
    scale = randn(batch_shape + (y_size,)).exp()

    normal = dist.Normal(loc, scale).to_event(1)
    actual = matrix_and_mvn_to_funsor(matrix, normal)
    assert isinstance(actual, AffineNormal)

    mvn = dist.MultivariateNormal(loc, scale_tril=scale.diag_embed())
    expected = matrix_and_mvn_to_funsor(matrix, mvn)

    y = tensor_to_funsor(randn(batch_shape + (y_size,)), (), 1)
    actual_like = actual(value_y=y)
    expected_like = expected(value_y=y)
    assert_close(actual_like, expected_like, atol=1e-4, rtol=1e-4)

    x = tensor_to_funsor(randn(batch_shape + (x_size,)), (), 1)
    actual_norm = actual_like(value_x=x)
    expected_norm = expected_like(value_x=x)
    assert_close(actual_norm, expected_norm, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("real_size", [1, 2, 3])
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_funsor_to_mvn(batch_shape, event_shape, real_size):
    expected = random_mvn(batch_shape + event_shape, real_size)
    event_dims = tuple("abc"[:len(event_shape)])
    ndims = len(expected.batch_shape)

    funsor_ = dist_to_funsor(expected, event_dims)(value="value")
    assert isinstance(funsor_, Funsor)

    actual = funsor_to_mvn(funsor_, ndims, event_dims)
    base_dist = get_unexpanded_dist(actual)
    assert isinstance(base_dist, dist.MultivariateNormal)
    assert actual.batch_shape == expected.batch_shape
    assert_close(ops.expand(base_dist.loc, expected.loc.shape),
                 expected.loc, atol=1e-3, rtol=None)
    assert_close(ops.expand(base_dist.precision_matrix, expected.precision_matrix.shape),
                 expected.precision_matrix, atol=1e-3, rtol=None)


@pytest.mark.parametrize("int_size", [2, 3])
@pytest.mark.parametrize("real_size", [1, 2, 3])
@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_funsor_to_cat_and_mvn(batch_shape, event_shape, int_size, real_size):
    logits = randn(batch_shape + event_shape + (int_size,))
    expected_cat = dist.Categorical(logits=logits)
    expected_mvn = random_mvn(batch_shape + event_shape + (int_size,), real_size)
    event_dims = tuple("abc"[:len(event_shape)]) + ("component",)
    ndims = len(expected_cat.batch_shape)

    funsor_ = (tensor_to_funsor(logits, event_dims) +
               dist_to_funsor(expected_mvn, event_dims)(value="value"))
    assert isinstance(funsor_, Funsor)

    actual_cat, actual_mvn = funsor_to_cat_and_mvn(funsor_, ndims, event_dims)
    cat_base_dist = get_unexpanded_dist(actual_cat)
    mvn_base_dist = get_unexpanded_dist(actual_mvn)
    assert isinstance(cat_base_dist, dist.CategoricalLogits)
    assert isinstance(mvn_base_dist, dist.MultivariateNormal)
    assert actual_cat.batch_shape == expected_cat.batch_shape
    assert actual_mvn.batch_shape == expected_mvn.batch_shape
    assert_close(ops.expand(cat_base_dist.logits, expected_cat.logits.shape),
                 expected_cat.logits, atol=1e-4, rtol=None)
    assert_close(ops.expand(mvn_base_dist.loc, expected_mvn.loc.shape),
                 expected_mvn.loc, atol=1e-4, rtol=None)
    assert_close(ops.expand(mvn_base_dist.precision_matrix, expected_mvn.precision_matrix.shape),
                 expected_mvn.precision_matrix, atol=1e-4, rtol=None)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("cardinality", [2, 3, 5])
def test_dist_to_funsor_categorical(batch_shape, cardinality):
    logits = randn(batch_shape + (cardinality,))
    logits -= ops.logsumexp(logits, -1)[..., None]
    d = dist.Categorical(logits=logits)
    f = dist_to_funsor(d)
    assert isinstance(f, Tensor)
    expected = tensor_to_funsor(logits, ("value",))
    assert_close(f, expected)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_bernoulli(batch_shape):
    logits = randn(batch_shape)
    d = dist.Bernoulli(logits=logits)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample(get_rng_key())
    actual_log_prob = f(value=tensor_to_funsor(value))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_normal(batch_shape):
    loc = randn(batch_shape)
    scale = ops.exp(randn(batch_shape))
    d = dist.Normal(loc, scale)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample(get_rng_key())
    actual_log_prob = f(value=tensor_to_funsor(value))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob, rtol=1e-5)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
@pytest.mark.parametrize("event_size", [2, 3, 5])
def test_dist_to_funsor_mvn(batch_shape, event_size):
    loc = randn(batch_shape + (event_size,))
    cov = randn(batch_shape + (event_size, 2 * event_size))
    cov = ops.matmul(cov, ops.transpose(cov, -1, -2))
    scale_tril = ops.cholesky(cov)
    d = dist.MultivariateNormal(loc, scale_tril=scale_tril)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample(get_rng_key())
    actual_log_prob = f(value=tensor_to_funsor(value, event_output=1))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("event_shape", [(), (6,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_independent(batch_shape, event_shape):
    loc = randn(batch_shape + event_shape)
    scale = ops.exp(randn(batch_shape + event_shape))
    d = dist.Normal(loc, scale).to_event(len(event_shape))
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample(get_rng_key())
    funsor_value = tensor_to_funsor(value, event_output=len(event_shape))
    actual_log_prob = f(value=funsor_value)
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob, rtol=1e-5)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES, ids=str)
def test_dist_to_funsor_masked(batch_shape):
    loc = randn(batch_shape)
    scale = ops.exp(randn(batch_shape))
    mask = ops.astype(dist.Bernoulli(ones(batch_shape) * 0.5).sample(get_rng_key()), 'uint8')
    d = dist.Normal(loc, scale).mask(mask)
    assert isinstance(d, dist.MaskedDistribution)
    f = dist_to_funsor(d)
    assert isinstance(f, Funsor)

    value = d.sample(get_rng_key())
    actual_log_prob = f(value=tensor_to_funsor(value))
    expected_log_prob = tensor_to_funsor(d.log_prob(value))
    assert_close(actual_log_prob, expected_log_prob)
