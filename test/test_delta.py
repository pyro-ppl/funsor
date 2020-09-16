# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import Real, Reals
from funsor.tensor import Tensor, numeric_array
from funsor.terms import Number, Variable
from funsor.testing import assert_close, check_funsor, randn


def test_eager_subs_variable():
    v = Variable('v', Reals[3])
    point = Tensor(randn(3))
    d = Delta('foo', v)
    assert d(v=point) is Delta('foo', point)


@pytest.mark.parametrize('log_density', [0, 1.234])
def test_eager_subs_ground(log_density):
    point1 = Tensor(randn(3))
    point2 = Tensor(randn(3))
    d = Delta('foo', point1, log_density)
    check_funsor(d(foo=point1), {}, Real, numeric_array(float(log_density)))
    check_funsor(d(foo=point2), {}, Real, numeric_array(float('-inf')))


def test_add_delta_funsor():
    x = Variable('x', Reals[3])
    y = Variable('y', Reals[3])
    d = Delta('x', y)

    expr = -(1 + x ** 2).log()
    assert d + expr is d + expr(x=y)
    assert expr + d is expr(x=y) + d


def test_reduce():
    point = Tensor(randn(3))
    d = Delta('foo', point)
    assert d.reduce(ops.logaddexp, frozenset(['foo'])) is Number(0)


@pytest.mark.parametrize('log_density', [0, 1.234])
def test_reduce_density(log_density):
    point = Tensor(randn(3))
    d = Delta('foo', point, log_density)
    # Note that log_density affects ground substitution but does not affect reduction.
    assert d.reduce(ops.logaddexp, frozenset(['foo'])) is Number(0)


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)], ids=str)
def test_transform_exp(shape):
    point = Tensor(ops.abs(randn(shape)))
    x = Variable('x', Reals[shape])
    actual = Delta('y', point)(y=ops.exp(x))
    expected = Delta('x', point.log(), point.log().sum())
    assert_close(actual, expected)


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)], ids=str)
def test_transform_log(shape):
    point = Tensor(randn(shape))
    x = Variable('x', Reals[shape])
    actual = Delta('y', point)(y=ops.log(x))
    expected = Delta('x', point.exp(), -point.sum())
    assert_close(actual, expected)
