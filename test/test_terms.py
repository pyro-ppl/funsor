from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.domains import Domain, ints, reals
from funsor.testing import check_funsor

np.seterr(all='ignore')


@pytest.mark.parametrize('x', ["foo", list(), tuple(), set(), dict()])
def test_to_funsor_undefined(x):
    with pytest.raises(ValueError):
        funsor.to_funsor(x)


def test_cons_hash():
    assert funsor.Variable('x', ints(3)) is funsor.Variable('x', ints(3))
    assert funsor.Variable('x', reals()) is funsor.Variable('x', reals())
    assert funsor.Variable('x', reals()) is not funsor.Variable('x', ints(3))
    assert funsor.Number(0, 3) is funsor.Number(0, 3)
    assert funsor.Number(0.) is funsor.Number(0.)
    assert funsor.Number(0.) is not funsor.Number(0, 3)


@pytest.mark.parametrize('expr', [
    "funsor.Variable('x', ints(3))",
    "funsor.Variable('x', reals())",
    "funsor.Number(0.)",
    "funsor.Number(1, dtype=10)",
    "-funsor.Variable('x', reals())",
    "funsor.Variable('x', reals()) + funsor.Variable('y', reals())",
    "funsor.Variable('x', reals())(x=funsor.Number(0.))",
])
def test_reinterpret(expr):
    x = eval(expr)
    assert funsor.reinterpret(x) is x


@pytest.mark.parametrize('domain', [ints(3), reals()])
def test_variable(domain):
    x = funsor.Variable('x', domain)
    check_funsor(x, {'x': domain}, domain)
    assert funsor.Variable('x', domain) is x
    assert x('x') is x
    y = funsor.Variable('y', domain)
    assert x('y') is y
    assert x(x='y') is y
    assert x(x=y) is y
    x4 = funsor.Variable('x', ints(4))
    assert x4 is not x
    assert x4('x') is x4
    assert x(x=x4) is x4
    assert x(y=x4) is x

    xp1 = x + 1.
    assert xp1(x=2.) == 3.


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(ops, symbol)(x)


@pytest.mark.parametrize('data', [0, 0.5, 1])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(symbol, data):
    dtype = 'real'
    if symbol == '~':
        data = bool(data)
        dtype = 2
    expected_data = unary_eval(symbol, data)

    x = funsor.Number(data, dtype)
    actual = unary_eval(symbol, x)
    check_funsor(actual, {}, Domain((), dtype), expected_data)


BINARY_OPS = [
    '+', '-', '*', '/', '**', '==', '!=', '<', '<=', '>', '>=',
    'min', 'max',
]
BOOLEAN_OPS = ['&', '|', '^']


def binary_eval(symbol, x, y):
    if symbol == 'min':
        return funsor.ops.min(x, y)
    if symbol == 'max':
        return funsor.ops.max(x, y)
    return eval('x {} y'.format(symbol))


@pytest.mark.parametrize('data1', [0, 0.2, 1])
@pytest.mark.parametrize('data2', [0, 0.8, 1])
@pytest.mark.parametrize('symbol', BINARY_OPS + BOOLEAN_OPS)
def test_binary(symbol, data1, data2):
    dtype = 'real'
    if symbol in BOOLEAN_OPS:
        dtype = 2
        data1 = bool(data1)
        data2 = bool(data2)
    try:
        expected_data = binary_eval(symbol, data1, data2)
    except ZeroDivisionError:
        return

    x1 = funsor.Number(data1, dtype)
    x2 = funsor.Number(data2, dtype)
    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, {}, Domain((), dtype), expected_data)
