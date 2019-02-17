from __future__ import absolute_import, division, print_function

import itertools

import numpy as np
import pytest
import torch

import funsor
import funsor.ops as ops
from funsor.handlers import Memoize
from funsor.testing import check_funsor

np.seterr(all='ignore')


def test_materialize():

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    g = funsor.materialize(f)

    assert g.dims == f.dims
    assert g.shape == f.shape
    for key in itertools.product(*map(range, g.shape)):
        assert f[key] == g[key]

    x = f.reduce(ops.add, frozenset(['i', 'j']))
    assert not x.dims
    assert not x.shape
    y = funsor.materialize(x)
    assert not y.dims
    assert not y.shape
    assert isinstance(y, funsor.Tensor)


def test_to_funsor():
    assert isinstance(funsor.to_funsor(2), funsor.Number)
    assert isinstance(funsor.to_funsor(2.), funsor.Number)


@pytest.mark.parametrize('x', ["foo", list(), tuple(), set(), dict()])
def test_to_funsor_undefined(x):
    with pytest.raises(ValueError):
        funsor.to_funsor(x)


def test_cons_hash():
    with Memoize():
        assert funsor.Variable('x', 3) is funsor.Variable('x', 3)
        assert funsor.Variable('x', 'real') is funsor.Variable('x', 'real')
        assert funsor.Variable('x', 'real') is not funsor.Variable('x', 3)
        assert funsor.Number(0) is funsor.Number(0)
        assert funsor.Number(0.) is funsor.Number(0.)
        assert funsor.Number(0.) is not funsor.Number(0)

        @funsor.of_shape('real', 2, 2)
        def f1(x, i, j):
            return (x ** i + j).sum('i')

        @funsor.of_shape('real', 2, 2)
        def f2(x, i, j):
            return (x ** i + j).sum('i')

        assert f1 is f2


@pytest.mark.parametrize('expr', [
    "funsor.Variable('x', 3)",
    "funsor.Variable('x', 'real')",
    "funsor.Number(0.)",
    "funsor.Number(1)",
    "-funsor.Variable('x', 'real')",
    "funsor.Variable('x', 'real') + funsor.Variable('y', 'real')",
    "funsor.Variable('x', 'real')(x=funsor.Number(0))",
])
def test_eval(expr):
    with Memoize():
        x = eval(expr)
        assert funsor.engine.eval(x) is x


@pytest.mark.parametrize('size', [3, 'real'])
def test_variable(size):
    with Memoize():
        x = funsor.Variable('x', size)
        check_funsor(x, ('x',), (size,))
        assert funsor.Variable('x', size) is x
        assert x['x'] is x
        assert x('x') is x
        y = funsor.Variable('y', size)
        assert x['y'] is y
        assert x('y') is y
        assert x(x='y') is y
        assert x(x=y) is y
        x4 = funsor.Variable('x', 4)
        assert x4 is not x
        assert x4['x'] is x4
        assert x(x=x4) is x4
        assert x(y=x4) is x

        xp1 = x + 1
        assert xp1(x=2) == 3


@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
@pytest.mark.parametrize('materialize_h', [False, True])
def test_mm(materialize_f, materialize_g, materialize_h):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = funsor.materialize(f)
        assert isinstance(f, funsor.Tensor)
    check_funsor(f, ('i', 'j'), (3, 4))

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_g:
        g = funsor.materialize(g)
        assert isinstance(g, funsor.Tensor)
    check_funsor(g, ('j', 'k'), (4, 5))

    h = (f * g).sum('j')
    if materialize_h:
        h = funsor.materialize(h)
        assert isinstance(h, funsor.Tensor)
    check_funsor(h, ('i', 'k'), (3, 5))
    for i in range(3):
        for k in range(5):
            assert funsor.materialize(h[i, k]) == sum(f[i, j] * g[j, k] for j in range(4))


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(ops, symbol)(x)


@pytest.mark.parametrize('data', [0, 0.5, 1])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(symbol, data):
    if symbol == '~':
        data = bool(data)
    expected_data = unary_eval(symbol, data)

    x = funsor.Number(data)
    actual = unary_eval(symbol, x)
    check_funsor(actual, (), (), expected_data)


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
    if symbol in BOOLEAN_OPS:
        data1 = bool(data1)
        data2 = bool(data2)
    try:
        expected_data = binary_eval(symbol, data1, data2)
    except ZeroDivisionError:
        return

    x1 = funsor.Number(data1)
    x2 = funsor.Number(data2)
    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, (), (), expected_data)


def test_branch():
    with Memoize():
        x = funsor.Variable('x', 'real')
        y = funsor.Number(0.5)
        z = funsor.Tensor(('i', 'j'), torch.randn(10, 5))
        xyz = funsor.Branch('c', (x, y, z))
        assert xyz.dims == ('c', 'x', 'i', 'j')
        assert xyz.shape == (3, 'real', 10, 5)

        assert xyz[0] is x
        assert xyz[1] is y
        assert xyz[2] is z
        assert xyz[funsor.Number(0)] is x
        assert xyz[funsor.Number(1)] is y
        assert xyz[funsor.Number(2)] is z
        assert xyz[funsor.Tensor((), torch.tensor(0))] is x
        assert xyz[funsor.Tensor((), torch.tensor(1))] is y
        assert xyz[funsor.Tensor((), torch.tensor(2))] is z
        assert tuple(xyz) == (x, y, z)

        assert xyz(x=0) is funsor.Branch('c', (x(0), y, z))
        assert xyz(i=1).dims == funsor.Branch('c', (x, y, z(i=1))).dims
        assert xyz(i=1).shape == funsor.Branch('c', (x, y, z(i=1))).shape


def test_of_shape():

    @funsor.of_shape(3)
    def f(i):
        return 0

    check_funsor(f, ('i',), (3,))

    @funsor.of_shape('real', 'real')
    def g(x, y):
        return y - x ** 2

    check_funsor(g, ('x', 'y'), ('real', 'real'))
