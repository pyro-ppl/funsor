# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import pytest

import funsor
from funsor import Number, Variable, bint
from funsor.domains import Domain
from funsor.interpreter import _USE_TCO, interpretation
from funsor.numpy import Array, align_arrays
from funsor.terms import lazy
from funsor.testing import assert_equiv, check_funsor, random_array


@xfail_with_tco
@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
def test_advanced_indexing_lazy(output_shape):
    x = Array(np.random.normal(size=(2, 3, 4) + output_shape), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]))
    u = Variable('u', bint(2))
    v = Variable('v', bint(3))
    with interpretation(lazy):
        i = Number(1, 2) - u
        j = Number(2, 3) - v
        k = u + v

    expected_data = np.empty((2, 3) + output_shape)
    i_data = funsor.numpy.materialize(i).data.astype(np.int64)
    j_data = funsor.numpy.materialize(j).data.astype(np.int64)
    k_data = funsor.numpy.materialize(k).data.astype(np.int64)
    for u in range(2):
        for v in range(3):
            expected_data[u, v] = x.data[i_data[u], j_data[v], k_data[u, v]]
    expected = Array(expected_data, OrderedDict([
        ('u', bint(2)),
        ('v', bint(3)),
    ]))

    assert_equiv(expected, x(i, j, k))
    assert_equiv(expected, x(i=i, j=j, k=k))

    assert_equiv(expected, x(i=i, j=j)(k=k))
    assert_equiv(expected, x(j=j, k=k)(i=i))
    assert_equiv(expected, x(k=k, i=i)(j=j))

    assert_equiv(expected, x(i=i)(j=j, k=k))
    assert_equiv(expected, x(j=j)(k=k, i=i))
    assert_equiv(expected, x(k=k)(i=i, j=j))

    assert_equiv(expected, x(i=i)(j=j)(k=k))
    assert_equiv(expected, x(i=i)(k=k)(j=j))
    assert_equiv(expected, x(j=j)(i=i)(k=k))
    assert_equiv(expected, x(j=j)(k=k)(i=i))
    assert_equiv(expected, x(k=k)(i=i)(j=j))
    assert_equiv(expected, x(k=k)(j=j)(i=i))


@xfail_with_tco
def test_align():
    x = Array(np.random.randn(2, 3, 4), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]))
    y = x.align(('j', 'k', 'i'))
    assert isinstance(y, Array)
    assert tuple(y.inputs) == ('j', 'k', 'i')
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert x(i=i, j=j, k=k) == y(i=i, j=j, k=k)


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    if isinstance(x, np.ndarray):
        return getattr(funsor.ops, symbol)(x)
    return getattr(x, symbol)()


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b')])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p', 'sigmoid',
])
def test_unary(symbol, dims):
    sizes = {'a': 3, 'b': 4}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    dtype = 'real'
    data = np.array(np.random.rand(*shape) + 0.5)
    if symbol == '~':
        data = data.astype(bool)
        dtype = 2
    expected_data = unary_eval(symbol, data)

    x = Array(data, inputs, dtype)
    actual = unary_eval(symbol, x)
    # FIXME: this raises AttributeErorr: 'Unary' object has no attribute 'data'
    check_funsor(actual, inputs, funsor.Domain((), dtype), expected_data)


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


@pytest.mark.parametrize('dims2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')])
@pytest.mark.parametrize('dims1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS + BOOLEAN_OPS)
def test_binary_funsor_funsor(symbol, dims1, dims2):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    inputs1 = OrderedDict((d, bint(sizes[d])) for d in dims1)
    inputs2 = OrderedDict((d, bint(sizes[d])) for d in dims2)
    data1 = np.array(np.random.rand(*shape1)) + 0.5
    data2 = np.array(np.random.rand(*shape2)) + 0.5
    dtype = 'real'
    if symbol in BOOLEAN_OPS:
        dtype = 2
        data1 = data1.astype(bool)
        data2 = data2.astype(bool)
    x1 = Array(data1, inputs1, dtype)
    x2 = Array(data2, inputs2, dtype)
    inputs, aligned = align_arrays(x1, x2)
    expected_data = binary_eval(symbol, aligned[0], aligned[1])

    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, inputs, Domain((), dtype), expected_data)
