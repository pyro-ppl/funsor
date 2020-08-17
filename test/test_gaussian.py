# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import OrderedDict
from functools import reduce

import numpy as np
import pytest

import funsor.ops as ops
from funsor.cnf import Contraction, GaussianMixture
from funsor.domains import Bint, Real, Reals
from funsor.gaussian import BlockMatrix, BlockVector, Gaussian
from funsor.integrate import Integrate
from funsor.tensor import Einsum, Tensor, numeric_array
from funsor.terms import Number, Variable
from funsor.testing import (assert_close, id_from_inputs, ones, randn, random_gaussian,
                            random_tensor, zeros)
from funsor.util import get_backend

assert Einsum  # flake8


@pytest.mark.parametrize("size", [1, 2, 3], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
@pytest.mark.skipif(get_backend() != "torch", reason="The test is specific to 'torch' backend")
def test_cholesky_solve(batch_shape, size):
    import torch

    b = torch.randn(batch_shape + (size, 5))
    x = torch.randn(batch_shape + (size, size))
    x = x.transpose(-1, -2).matmul(x)
    u = x.cholesky()
    expected = torch.cholesky_solve(b, u)
    assert not expected.requires_grad
    actual = torch.cholesky_solve(b.requires_grad_(), u.requires_grad_())
    assert actual.requires_grad
    assert_close(expected, actual)


def naive_cholesky_inverse(u):
    import torch

    shape = u.shape
    return torch.stack([
        part.cholesky_inverse()
        for part in u.reshape((-1,) + u.shape[-2:])
    ]).reshape(shape)


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3)], ids=str)
@pytest.mark.skipif(get_backend() != "torch", reason="The test is specific to 'torch' backend")
def test_cholesky_inverse(batch_shape, size, requires_grad):
    import torch

    x = torch.randn(batch_shape + (size, size))
    x = x.transpose(-1, -2).matmul(x)
    u = x.cholesky()
    if requires_grad:
        u.requires_grad_()
    assert_close(ops.cholesky_inverse(u), naive_cholesky_inverse(u))
    if requires_grad:
        ops.cholesky_inverse(u).sum().backward()


def test_block_vector():
    shape = (10,)
    expected = zeros(shape)
    actual = BlockVector(shape)

    expected[1] = randn(())
    actual[1] = expected[1]

    expected[3:5] = randn((2,))
    actual[3:5] = expected[3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)])
def test_block_vector_batched(batch_shape):
    shape = batch_shape + (10,)
    expected = zeros(shape)
    actual = BlockVector(shape)

    expected[..., 1] = randn(batch_shape)
    actual[..., 1] = expected[..., 1]

    expected[..., 3:5] = randn(batch_shape + (2,))
    actual[..., 3:5] = expected[..., 3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize('sparse', [False, True])
def test_block_matrix(sparse):
    shape = (10, 10)
    expected = zeros(shape)
    actual = BlockMatrix(shape)

    expected[1, 1] = randn(())
    actual[1, 1] = expected[1, 1]

    if not sparse:
        expected[1, 3:5] = randn((2,))
        actual[1, 3:5] = expected[1, 3:5]

        expected[3:5, 1] = randn((2,))
        actual[3:5, 1] = expected[3:5, 1]

    expected[3:5, 3:5] = randn((2, 2))
    actual[3:5, 3:5] = expected[3:5, 3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)])
def test_block_matrix_batched(batch_shape, sparse):
    shape = batch_shape + (10, 10)
    expected = zeros(shape)
    actual = BlockMatrix(shape)

    expected[..., 1, 1] = randn(batch_shape)
    actual[..., 1, 1] = expected[..., 1, 1]

    if not sparse:
        expected[..., 1, 3:5] = randn(batch_shape + (2,))
        actual[..., 1, 3:5] = expected[..., 1, 3:5]

        expected[..., 3:5, 1] = randn(batch_shape + (2,))
        actual[..., 3:5, 1] = expected[..., 3:5, 1]

    expected[..., 3:5, 3:5] = randn(batch_shape + (2, 2))
    actual[..., 3:5, 3:5] = expected[..., 3:5, 3:5]

    assert_close(actual.as_tensor(), expected)


@pytest.mark.parametrize('expr,expected_type', [
    ('-g1', Gaussian),
    ('g1 + 1', Contraction),
    ('g1 - 1', Contraction),
    ('1 + g1', Contraction),
    ('g1 + shift', Contraction),
    ('g1 + shift', Contraction),
    ('shift + g1', Contraction),
    ('shift - g1', Contraction),
    ('g1 + g1', Gaussian),
    ('(g1 + g2 + g2) - g2', Gaussian),
    ('g1(i=i0)', Gaussian),
    ('g2(i=i0)', Gaussian),
    ('g1(i=i0) + g2(i=i0)', Gaussian),
    ('g1(i=i0) + g2', Gaussian),
    ('g1(x=x0)', Tensor),
    ('g2(y=y0)', Tensor),
    ('(g1 + g2)(i=i0)', Gaussian),
    ('(g1 + g2)(x=x0, y=y0)', Tensor),
    ('(g2 + g1)(x=x0, y=y0)', Tensor),
    ('g1.reduce(ops.logaddexp, "x")', Tensor),
    ('(g1 + g2).reduce(ops.logaddexp, "x")', Contraction),
    ('(g1 + g2).reduce(ops.logaddexp, "y")', Contraction),
    ('(g1 + g2).reduce(ops.logaddexp, frozenset(["x", "y"]))', Tensor),
])
def test_smoke(expr, expected_type):
    g1 = Gaussian(
        info_vec=numeric_array([[0.0, 0.1, 0.2],
                               [2.0, 3.0, 4.0]]),
        precision=numeric_array([[[1.0, 0.1, 0.2],
                                  [0.1, 1.0, 0.3],
                                  [0.2, 0.3, 1.0]],
                                 [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', Bint[2]), ('x', Reals[3])]))
    assert isinstance(g1, Gaussian)

    g2 = Gaussian(
        info_vec=numeric_array([[0.0, 0.1],
                                [2.0, 3.0]]),
        precision=numeric_array([[[1.0, 0.2],
                                  [0.2, 1.0]],
                                 [[1.0, 0.2],
                                  [0.2, 1.0]]]),
        inputs=OrderedDict([('i', Bint[2]), ('y', Reals[2])]))
    assert isinstance(g2, Gaussian)

    shift = Tensor(numeric_array([-1., 1.]), OrderedDict([('i', Bint[2])]))
    assert isinstance(shift, Tensor)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(numeric_array([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    y0 = Tensor(numeric_array([[0.2, 0.3],
                               [0.8, 0.9]]),
                inputs=OrderedDict([('i', Bint[2])]))
    assert isinstance(y0, Tensor)

    result = eval(expr)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('x', Reals[4])]),
    OrderedDict([('x', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
], ids=id_from_inputs)
def test_align(int_inputs, real_inputs):
    inputs1 = OrderedDict(list(sorted(int_inputs.items())) +
                          list(sorted(real_inputs.items())))
    inputs2 = OrderedDict(reversed(inputs1.items()))
    g1 = random_gaussian(inputs1)
    g2 = g1.align(tuple(inputs2))
    assert g2.inputs == inputs2
    g3 = g2.align(tuple(inputs1))
    assert_close(g3, g1)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('x', Reals[4])]),
    OrderedDict([('x', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
], ids=id_from_inputs)
def test_eager_subs_origin(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)
    g = random_gaussian(inputs)

    # Check that Gaussian log density at origin is zero.
    origin = {k: zeros(d.shape) for k, d in real_inputs.items()}
    actual = g(**origin)
    expected_data = zeros(tuple(d.size for d in int_inputs.values()))
    expected = Tensor(expected_data, int_inputs)
    assert_close(actual, expected)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('x', Reals[4])]),
    OrderedDict([('x', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
], ids=id_from_inputs)
def test_eager_subs(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)

    for order in itertools.permutations(inputs):
        ground_values = {}
        dependent_values = {}
        for i, name in enumerate(order):
            upstream = OrderedDict([(k, inputs[k]) for k in order[:i] if k in int_inputs])
            value = random_tensor(upstream, inputs[name])
            ground_values[name] = value(**ground_values)
            dependent_values[name] = value

        expected = g(**ground_values)
        actual = g
        for k in reversed(order):
            actual = actual(**{k: dependent_values[k]})
        assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_eager_subs_variable():
    inputs = OrderedDict([('i', Bint[2]), ('x', Real), ('y', Reals[2])])
    g1 = random_gaussian(inputs)

    g2 = g1(x='z')
    assert set(g2.inputs) == {'i', 'y', 'z'}
    assert g2.info_vec is g1.info_vec
    assert g2.precision is g1.precision

    g2 = g1(x='y', y='x')
    assert set(g2.inputs) == {'i', 'x', 'y'}
    assert g2.inputs['x'] == Reals[2]
    assert g2.info_vec is g1.info_vec
    assert g2.precision is g1.precision

    g2 = g1(i='j')
    assert set(g2.inputs) == {'j', 'x', 'y'}
    assert g2.info_vec is g1.info_vec
    assert g2.precision is g1.precision


@pytest.mark.parametrize('subs', [
    (('x', 'Variable("u", Real) * 2'),),
    (('y', 'Variable("v", Reals[4]) + 1'),),
    (('z', 'Variable("w", Reals[6]).reshape((2,3))'),),
    (('x', 'Variable("v", Reals[4]).sum()'),
     ('y', 'Variable("v", Reals[4]) - 1')),
    (('x', 'Variable("u", Real) * 2 + 1'),
     ('y', 'Variable("u", Real) * Tensor(ones((4,)))'),
     ('z', 'Variable("u", Real) * Tensor(ones((2, 3)))')),
    (('y', 'Einsum("abc,bc->a", (Tensor(randn((4, 3, 5))), Variable("v", Reals[3, 5])))'),),
])
@pytest.mark.parametrize('g_ints', ["", "i", "j", "ij"])
@pytest.mark.parametrize('subs_ints', ["", "i", "j", "ji"])
def test_eager_subs_affine(subs, g_ints, subs_ints):
    sizes = {'i': 5, 'j': 6}
    subs_inputs = OrderedDict((k, Bint[sizes[k]]) for k in subs_ints)
    g_inputs = OrderedDict((k, Bint[sizes[k]]) for k in g_ints)
    g_inputs['x'] = Real
    g_inputs['y'] = Reals[4]
    g_inputs['z'] = Reals[2, 3]
    g = random_gaussian(g_inputs)
    subs = {k: eval(v) + random_tensor(subs_inputs) for k, v in subs}

    inputs = g.inputs.copy()
    for v in subs.values():
        inputs.update(v.inputs)
    grounding_subs = {k: random_tensor(OrderedDict(), d) for k, d in inputs.items()}
    ground_subs = {k: v(**grounding_subs) for k, v in subs.items()}

    g_subs = g(**subs)
    assert issubclass(type(g_subs), GaussianMixture)
    actual = g_subs(**grounding_subs)
    expected = g(**ground_subs)(**grounding_subs)
    assert_close(actual, expected, atol=1e-3, rtol=2e-4)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('x', Reals[4])]),
    OrderedDict([('x', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
], ids=id_from_inputs)
def test_add_gaussian_number(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    n = Number(1.234)
    values = {name: random_tensor(int_inputs, domain)
              for name, domain in real_inputs.items()}

    assert_close((g + n)(**values), g(**values) + n, atol=1e-5, rtol=1e-5)
    assert_close((n + g)(**values), n + g(**values), atol=1e-5, rtol=1e-5)
    assert_close((g - n)(**values), g(**values) - n, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('x', Reals[4])]),
    OrderedDict([('x', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
], ids=id_from_inputs)
def test_add_gaussian_tensor(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    t = random_tensor(int_inputs, Real)
    values = {name: random_tensor(int_inputs, domain)
              for name, domain in real_inputs.items()}

    assert_close((g + t)(**values), g(**values) + t, atol=1e-5, rtol=1e-5)
    assert_close((t + g)(**values), t + g(**values), atol=1e-5, rtol=1e-5)
    assert_close((g - t)(**values), g(**values) - t, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('lhs_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('y', Reals[4])]),
    OrderedDict([('z', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Reals[4])]),
    OrderedDict([('y', Reals[4]), ('z', Reals[2, 3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('rhs_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('y', Reals[4])]),
    OrderedDict([('z', Reals[2, 3])]),
    OrderedDict([('x', Real), ('y', Reals[4])]),
    OrderedDict([('y', Reals[4]), ('z', Reals[2, 3])]),
], ids=id_from_inputs)
def test_add_gaussian_gaussian(lhs_inputs, rhs_inputs):
    lhs_inputs = OrderedDict(sorted(lhs_inputs.items()))
    rhs_inputs = OrderedDict(sorted(rhs_inputs.items()))
    inputs = lhs_inputs.copy()
    inputs.update(rhs_inputs)
    int_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype != 'real')
    real_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype == 'real')

    g1 = random_gaussian(lhs_inputs)
    g2 = random_gaussian(rhs_inputs)
    values = {name: random_tensor(int_inputs, domain)
              for name, domain in real_inputs.items()}

    assert_close((g1 + g2)(**values), g1(**values) + g2(**values), atol=1e-4, rtol=None)


@pytest.mark.parametrize('inputs', [
    OrderedDict([('i', Bint[2]), ('x', Real)]),
    OrderedDict([('i', Bint[3]), ('x', Real)]),
    OrderedDict([('i', Bint[2]), ('x', Reals[2])]),
    OrderedDict([('i', Bint[2]), ('x', Real), ('y', Real)]),
    OrderedDict([('i', Bint[3]), ('j', Bint[4]), ('x', Reals[2])]),
], ids=id_from_inputs)
def test_reduce_add(inputs):
    g = random_gaussian(inputs)
    actual = g.reduce(ops.add, 'i')

    gs = [g(i=i) for i in range(g.inputs['i'].dtype)]
    expected = reduce(ops.add, gs)
    assert_close(actual, expected)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
    OrderedDict([('w', Reals[5]), ('x', Reals[4]), ('y', Reals[2, 3]), ('z', Real)]),
], ids=id_from_inputs)
def test_reduce_logsumexp(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    g = random_gaussian(inputs)
    g_xy = g.reduce(ops.logaddexp, frozenset(['x', 'y']))
    assert_close(g_xy, g.reduce(ops.logaddexp, 'x').reduce(ops.logaddexp, 'y'), atol=1e-3, rtol=None)
    assert_close(g_xy, g.reduce(ops.logaddexp, 'y').reduce(ops.logaddexp, 'x'), atol=1e-3, rtol=None)


@pytest.mark.parametrize('int_inputs', [
    {},
    {'i': Bint[2]},
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    {'x': Real},
    {'x': Reals[4]},
    {'x': Reals[2, 3]},
], ids=id_from_inputs)
def test_integrate_variable(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    log_measure = random_gaussian(inputs)
    integrand = reduce(ops.add, [Variable(k, d) for k, d in real_inputs.items()])
    reduced_vars = frozenset(real_inputs)

    rng_key = None if get_backend() != 'jax' else np.array([0, 0], dtype=np.uint32)
    sampled_log_measure = log_measure.sample(reduced_vars, OrderedDict(particle=Bint[100000]), rng_key=rng_key)
    approx = Integrate(sampled_log_measure, integrand, reduced_vars | {'particle'})
    assert isinstance(approx, Tensor)

    exact = Integrate(log_measure, integrand, reduced_vars)
    assert isinstance(exact, Tensor)
    assert_close(approx, exact, atol=0.1, rtol=0.1)


@pytest.mark.parametrize('int_inputs', [
    OrderedDict(),
    OrderedDict([('i', Bint[2])]),
    OrderedDict([('i', Bint[2]), ('j', Bint[3])]),
], ids=id_from_inputs)
@pytest.mark.parametrize('real_inputs', [
    OrderedDict([('x', Real)]),
    OrderedDict([('x', Reals[2])]),
    OrderedDict([('x', Real), ('y', Real)]),
    OrderedDict([('x', Reals[2]), ('y', Reals[3])]),
    OrderedDict([('x', Reals[4]), ('y', Reals[2, 3])]),
], ids=id_from_inputs)
def test_integrate_gaussian(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    log_measure = random_gaussian(inputs)
    integrand = random_gaussian(inputs)
    reduced_vars = frozenset(real_inputs)

    rng_key = None if get_backend() != 'jax' else np.array([0, 0], dtype=np.uint32)
    sampled_log_measure = log_measure.sample(reduced_vars, OrderedDict(particle=Bint[100000]), rng_key=rng_key)
    approx = Integrate(sampled_log_measure, integrand, reduced_vars | {'particle'})
    assert isinstance(approx, Tensor)

    exact = Integrate(log_measure, integrand, reduced_vars)
    assert isinstance(exact, Tensor)
    assert_close(approx, exact, atol=0.1, rtol=0.1)


@pytest.mark.xfail(get_backend() == 'torch', reason="numerically unstable in torch backend")
def test_mc_plate_gaussian():
    log_measure = Gaussian(numeric_array([0.]), numeric_array([[1.]]),
                           (('loc', Real),)) + numeric_array(-0.9189)
    integrand = Gaussian(randn((100, 1)) + 3., ones((100, 1, 1)),
                         (('data', Bint[100]), ('loc', Real)))

    rng_key = None if get_backend() != 'jax' else np.array([0, 0], dtype=np.uint32)
    res = Integrate(log_measure.sample('loc', rng_key=rng_key), integrand, 'loc')
    res = res.reduce(ops.mul, 'data')
    assert not ((res == float('inf')) | (res == float('-inf'))).any()
