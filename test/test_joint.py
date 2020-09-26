# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import reduce

import numpy as np
import pytest

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import Bint, Real, Reals
from funsor.gaussian import Gaussian
from funsor.integrate import Integrate
from funsor.interpreter import interpretation
from funsor.montecarlo import MonteCarlo
from funsor.tensor import Tensor, numeric_array
from funsor.terms import Number, Variable, eager, moment_matching
from funsor.testing import (assert_close, randn, random_gaussian, random_tensor,
                            zeros, xfail_if_not_implemented)
from funsor.util import get_backend


def id_from_inputs(inputs):
    if not inputs:
        return '()'
    return ','.join(k + ''.join(map(str, d.shape)) for k, d in inputs.items())


SMOKE_TESTS = [
    ('dx + dy', Delta),
    ('dx + g', Contraction),
    ('dy + g', Contraction),
    ('g + dx', Contraction),
    ('g + dy', Contraction),
    ('dx + t', Contraction),
    ('dy + t', Contraction),
    ('dx - t', Contraction),
    ('dy - t', Contraction),
    ('t + dx', Contraction),
    ('t + dy', Contraction),
    ('g + 1', Contraction),
    ('g - 1', Contraction),
    ('1 + g', Contraction),
    ('g + t', Contraction),
    ('g - t', Contraction),
    ('t + g', Contraction),
    ('t - g', Contraction),
    ('g + g', Gaussian),
    ('-(g + g)', Gaussian),
    ('(dx + dy)(i=i0)', Delta),
    ('(dx + g)(i=i0)', Contraction),
    ('(dy + g)(i=i0)', Contraction),
    ('(g + dx)(i=i0)', Contraction),
    ('(g + dy)(i=i0)', Contraction),
    ('(dx + t)(i=i0)', Contraction),
    ('(dy + t)(i=i0)', Contraction),
    ('(dx - t)(i=i0)', Contraction),
    ('(dy - t)(i=i0)', Contraction),
    ('(t + dx)(i=i0)', Contraction),
    ('(t + dy)(i=i0)', Contraction),
    ('(g + 1)(i=i0)', Contraction),
    ('(g - 1)(i=i0)', Contraction),
    ('(1 + g)(i=i0)', Contraction),
    ('(g + t)(i=i0)', Contraction),
    ('(g - t)(i=i0)', Contraction),
    ('(t + g)(i=i0)', Contraction),
    ('(g + g)(i=i0)', Gaussian),
    ('(dx + dy)(x=x0)', Contraction),
    ('(dx + g)(x=x0)', Tensor),
    ('(dy + g)(x=x0)', Contraction),
    ('(g + dx)(x=x0)', Tensor),
    ('(g + dy)(x=x0)', Contraction),
    ('(dx + t)(x=x0)', Tensor),
    ('(dy + t)(x=x0)', Contraction),
    ('(dx - t)(x=x0)', Tensor),
    ('(dy - t)(x=x0)', Contraction),
    ('(t + dx)(x=x0)', Tensor),
    ('(t + dy)(x=x0)', Contraction),
    ('(g + 1)(x=x0)', Tensor),
    ('(g - 1)(x=x0)', Tensor),
    ('(1 + g)(x=x0)', Tensor),
    ('(g + t)(x=x0)', Tensor),
    ('(g - t)(x=x0)', Tensor),
    ('(t + g)(x=x0)', Tensor),
    ('(g + g)(x=x0)', Tensor),
    ('(g + dy).reduce(ops.logaddexp, "x")', Contraction),
    ('(g + dy).reduce(ops.logaddexp, "y")', Contraction),
    ('(t + g + dy).reduce(ops.logaddexp, "x")', Contraction),
    ('(t + g + dy).reduce(ops.logaddexp, "y")', Contraction),
    ('(t + g).reduce(ops.logaddexp, "x")', Tensor),
]


@pytest.mark.parametrize('expr,expected_type', SMOKE_TESTS)
def test_smoke(expr, expected_type):
    dx = Delta('x', Tensor(randn(2, 3), OrderedDict([('i', Bint[2])])))
    assert isinstance(dx, Delta)

    dy = Delta('y', Tensor(randn(3, 4), OrderedDict([('j', Bint[3])])))
    assert isinstance(dy, Delta)

    t = Tensor(randn(2, 3), OrderedDict([('i', Bint[2]), ('j', Bint[3])]))
    assert isinstance(t, Tensor)

    g = Gaussian(
        info_vec=numeric_array([[0.0, 0.1, 0.2],
                                [2.0, 3.0, 4.0]]),
        precision=numeric_array([[[1.0, 0.1, 0.2],
                                  [0.1, 1.0, 0.3],
                                  [0.2, 0.3, 1.0]],
                                 [[1.0, 0.1, 0.2],
                                  [0.1, 1.0, 0.3],
                                  [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', Bint[2]), ('x', Reals[3])]))
    assert isinstance(g, Gaussian)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(numeric_array([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

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
def test_reduce_logaddexp(int_inputs, real_inputs):
    int_inputs = OrderedDict(sorted(int_inputs.items()))
    real_inputs = OrderedDict(sorted(real_inputs.items()))
    inputs = int_inputs.copy()
    inputs.update(real_inputs)

    t = random_tensor(int_inputs)
    g = random_gaussian(inputs)
    truth = {name: random_tensor(int_inputs, domain) for name, domain in real_inputs.items()}

    state = 0
    state += g
    state += t
    for name, point in truth.items():
        with xfail_if_not_implemented():
            state += Delta(name, point)
    actual = state.reduce(ops.logaddexp, frozenset(truth))

    expected = t + g(**truth)
    assert_close(actual, expected, atol=1e-5, rtol=1e-4 if get_backend() == "jax" else 1e-5)


def test_reduce_logaddexp_deltas_lazy():
    a = Delta('a', Tensor(randn(3, 2), OrderedDict(i=Bint[3])))
    b = Delta('b', Tensor(randn(3), OrderedDict(i=Bint[3])))
    x = a + b
    assert isinstance(x, Delta)
    assert set(x.inputs) == {'a', 'b', 'i'}

    y = x.reduce(ops.logaddexp, 'i')
    # assert isinstance(y, Reduce)
    assert set(y.inputs) == {'a', 'b'}
    assert_close(x.reduce(ops.logaddexp), y.reduce(ops.logaddexp))


def test_reduce_logaddexp_deltas_discrete_lazy():
    a = Delta('a', Tensor(randn(3, 2), OrderedDict(i=Bint[3])))
    b = Delta('b', Tensor(randn(3), OrderedDict(i=Bint[3])))
    c = Tensor(randn(3), OrderedDict(i=Bint[3]))
    x = a + b + c
    assert isinstance(x, Contraction)
    assert set(x.inputs) == {'a', 'b', 'i'}

    y = x.reduce(ops.logaddexp, 'i')
    # assert isinstance(y, Reduce)
    assert set(y.inputs) == {'a', 'b'}
    assert_close(x.reduce(ops.logaddexp), y.reduce(ops.logaddexp))


def test_reduce_logaddexp_gaussian_lazy():
    a = random_gaussian(OrderedDict(i=Bint[3], a=Reals[2]))
    b = random_tensor(OrderedDict(i=Bint[3], b=Bint[2]))
    x = a + b
    assert isinstance(x, Contraction)
    assert set(x.inputs) == {'a', 'b', 'i'}

    y = x.reduce(ops.logaddexp, 'i')
    # assert isinstance(y, Reduce)
    assert set(y.inputs) == {'a', 'b'}
    assert_close(x.reduce(ops.logaddexp), y.reduce(ops.logaddexp))


@pytest.mark.parametrize('inputs', [
    OrderedDict([('i', Bint[2]), ('x', Real)]),
    OrderedDict([('i', Bint[3]), ('x', Real)]),
    OrderedDict([('i', Bint[2]), ('x', Reals[2])]),
    OrderedDict([('i', Bint[2]), ('x', Real), ('y', Real)]),
    OrderedDict([('i', Bint[3]), ('j', Bint[4]), ('x', Reals[2])]),
    OrderedDict([('j', Bint[2]), ('i', Bint[3]), ('k', Bint[2]), ('x', Reals[2])]),
], ids=id_from_inputs)
def test_reduce_add(inputs):
    int_inputs = OrderedDict((k, d) for k, d in inputs.items() if d.dtype != 'real')
    x = random_gaussian(inputs) + random_tensor(int_inputs)
    assert isinstance(x, Contraction)
    actual = x.reduce(ops.add, 'i')

    xs = [x(i=i) for i in range(x.inputs['i'].dtype)]
    expected = reduce(ops.add, xs)
    assert_close(actual, expected, atol=1e-3, rtol=1e-4)


def test_reduce_moment_matching_univariate():
    int_inputs = [('i', Bint[2])]
    real_inputs = [('x', Real)]
    inputs = OrderedDict(int_inputs + real_inputs)
    int_inputs = OrderedDict(int_inputs)
    real_inputs = OrderedDict(real_inputs)

    p = 0.8
    t = 1.234
    s1, s2, s3 = 2.0, 3.0, 4.0
    loc = numeric_array([[-s1], [s1]])
    precision = numeric_array([[[s2 ** -2]], [[s3 ** -2]]])
    info_vec = (precision @ ops.unsqueeze(loc, -1)).squeeze(-1)
    discrete = Tensor(ops.log(numeric_array([1 - p, p])) + t, int_inputs)
    gaussian = Gaussian(info_vec, precision, inputs)
    gaussian -= gaussian.log_normalizer
    joint = discrete + gaussian
    with interpretation(moment_matching):
        actual = joint.reduce(ops.logaddexp, 'i')
    assert_close(actual.reduce(ops.logaddexp), joint.reduce(ops.logaddexp))

    expected_loc = numeric_array([(2 * p - 1) * s1])
    expected_variance = (4 * p * (1 - p) * s1 ** 2
                         + (1 - p) * s2 ** 2
                         + p * s3 ** 2)
    expected_precision = numeric_array([[1 / expected_variance]])
    expected_info_vec = (expected_precision @ ops.unsqueeze(expected_loc, -1)).squeeze(-1)
    expected_gaussian = Gaussian(expected_info_vec, expected_precision, real_inputs)
    expected_gaussian -= expected_gaussian.log_normalizer
    expected_discrete = Tensor(numeric_array(t))
    expected = expected_discrete + expected_gaussian
    assert_close(actual, expected, atol=1e-5, rtol=None)


def _inverse(x):
    if get_backend() == "torch":
        return x.inverse()
    else:
        return np.linalg.inv(x)


def test_reduce_moment_matching_multivariate():
    int_inputs = [('i', Bint[4])]
    real_inputs = [('x', Reals[2])]
    inputs = OrderedDict(int_inputs + real_inputs)
    int_inputs = OrderedDict(int_inputs)
    real_inputs = OrderedDict(real_inputs)

    loc = numeric_array([[-10., -1.],
                         [+10., -1.],
                         [+10., +1.],
                         [-10., +1.]])
    precision = zeros(4, 1, 1) + ops.new_eye(loc, (2,))
    discrete = Tensor(zeros(4), int_inputs)
    gaussian = Gaussian(loc, precision, inputs)
    gaussian -= gaussian.log_normalizer
    joint = discrete + gaussian
    with interpretation(moment_matching):
        actual = joint.reduce(ops.logaddexp, 'i')
    assert_close(actual.reduce(ops.logaddexp), joint.reduce(ops.logaddexp))

    expected_loc = zeros(2)
    expected_covariance = numeric_array([[101., 0.], [0., 2.]])
    expected_precision = _inverse(expected_covariance)
    expected_gaussian = Gaussian(expected_loc, expected_precision, real_inputs)
    expected_gaussian -= expected_gaussian.log_normalizer
    expected_discrete = Tensor(ops.log(numeric_array(4.)))
    expected = expected_discrete + expected_gaussian
    assert_close(actual, expected, atol=1e-5, rtol=None)


@pytest.mark.parametrize('interp', [eager, moment_matching],
                         ids=lambda f: f.__name__)
def test_reduce_moment_matching_shape(interp):
    delta = Delta('x', random_tensor(OrderedDict([('h', Bint[7])])))
    discrete = random_tensor(OrderedDict(
        [('h', Bint[7]), ('i', Bint[6]), ('j', Bint[5]), ('k', Bint[4])]))
    gaussian = random_gaussian(OrderedDict(
        [('k', Bint[4]), ('l', Bint[3]), ('m', Bint[2]), ('y', Real), ('z', Reals[2])]))
    reduced_vars = frozenset(['i', 'k', 'l'])
    real_vars = frozenset(k for k, d in gaussian.inputs.items() if d.dtype == "real")
    joint = delta + discrete + gaussian
    with interpretation(interp):
        actual = joint.reduce(ops.logaddexp, reduced_vars)
    assert set(actual.inputs) == set(joint.inputs) - reduced_vars
    assert_close(actual.reduce(ops.logaddexp, real_vars),
                 joint.reduce(ops.logaddexp, real_vars | reduced_vars))


@pytest.mark.xfail(reason="missing pattern")
def test_reduce_moment_matching_moments():
    x = Variable('x', Reals[2])
    gaussian = random_gaussian(OrderedDict(
        [('i', Bint[2]), ('j', Bint[3]), ('x', Reals[2])]))
    with interpretation(moment_matching):
        approx = gaussian.reduce(ops.logaddexp, 'j')
    with interpretation(MonteCarlo(s=Bint[100000])):
        actual = Integrate(approx, Number(1.), 'x')
        expected = Integrate(gaussian, Number(1.), {'j', 'x'})
        assert_close(actual, expected, atol=1e-3, rtol=1e-3)

        actual = Integrate(approx, x, 'x')
        expected = Integrate(gaussian, x, {'j', 'x'})
        assert_close(actual, expected, atol=1e-2, rtol=1e-2)

        actual = Integrate(approx, x * x, 'x')
        expected = Integrate(gaussian, x * x, {'j', 'x'})
        assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_reduce_moment_matching_finite():
    delta = Delta('x', random_tensor(OrderedDict([('h', Bint[7])])))
    discrete = random_tensor(OrderedDict(
        [('i', Bint[6]), ('j', Bint[5]), ('k', Bint[3])]))
    gaussian = random_gaussian(OrderedDict(
        [('k', Bint[3]), ('l', Bint[2]), ('y', Real), ('z', Reals[2])]))

    discrete.data[1:, :] = -float('inf')
    discrete.data[:, 1:] = -float('inf')

    reduced_vars = frozenset(['j', 'k'])
    joint = delta + discrete + gaussian
    with interpretation(moment_matching):
        joint.reduce(ops.logaddexp, reduced_vars)
