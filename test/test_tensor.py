# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
import io
import itertools
import pickle
from collections import OrderedDict
from typing import get_type_hints

import numpy as np
import pytest

import funsor
import funsor.ops as ops
from funsor.domains import Array, Bint, Real, Product, Reals, find_domain
from funsor.interpreter import interpretation
from funsor.tensor import REDUCE_OP_TO_NUMERIC, Einsum, Tensor, align_tensors, numeric_array, stack, tensordot
from funsor.terms import Cat, Lambda, Number, Slice, Stack, Variable, lazy
from funsor.testing import assert_close, assert_equiv, check_funsor, empty, rand, randn, random_tensor, zeros
from funsor.util import get_backend


@pytest.mark.parametrize('output_shape', [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize('inputs', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')], ids=str)
def test_quote(output_shape, inputs):
    if get_backend() == "torch":
        import torch  # noqa: F401

    sizes = {'a': 4, 'b': 5, 'c': 6}
    inputs = OrderedDict((k, Bint[sizes[k]]) for k in inputs)
    x = random_tensor(inputs, Reals[output_shape])
    s = funsor.quote(x)
    assert isinstance(s, str)
    assert_close(eval(s), x)


@pytest.mark.parametrize('shape', [(), (4,), (3, 2)])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64', 'uint8', 'bool'])
def test_to_funsor(shape, dtype):
    t = ops.astype(randn(shape), dtype)
    f = funsor.to_funsor(t)
    assert isinstance(f, Tensor)
    assert funsor.to_funsor(t, Reals[shape]) is f
    with pytest.raises(ValueError):
        funsor.to_funsor(t, Reals[(5,) + shape])


def test_to_data():
    data = zeros((3, 3))
    x = Tensor(data)
    assert funsor.to_data(x) is data


def test_to_data_error():
    data = zeros((3, 3))
    x = Tensor(data, OrderedDict(i=Bint[3]))
    with pytest.raises(ValueError):
        funsor.to_data(x)


def test_cons_hash():
    x = randn((3, 3))
    assert Tensor(x) is Tensor(x)


def test_copy():
    data = randn(3, 2)
    x = Tensor(data)
    assert copy.copy(x) is x


def test_deepcopy():
    data = randn(3, 2)
    x = Tensor(data)

    y = copy.deepcopy(x)
    assert_close(x, y)
    assert y is not x
    assert y.data is not x.data

    memo = {id(data): data}
    z = copy.deepcopy(x, memo)
    assert z is x


def test_indexing():
    data = randn((4, 5))
    inputs = OrderedDict([('i', Bint[4]),
                          ('j', Bint[5])])
    x = Tensor(data, inputs)
    check_funsor(x, inputs, Real, data)

    assert x() is x
    assert x(k=3) is x
    check_funsor(x(1), {'j': Bint[5]}, Real, data[1])
    check_funsor(x(1, 2), {}, Real, data[1, 2])
    check_funsor(x(1, 2, k=3), {}, Real, data[1, 2])
    check_funsor(x(1, j=2), {}, Real, data[1, 2])
    check_funsor(x(1, j=2, k=3), (), Real, data[1, 2])
    check_funsor(x(1, k=3), {'j': Bint[5]}, Real, data[1])
    check_funsor(x(i=1), {'j': Bint[5]}, Real, data[1])
    check_funsor(x(i=1, j=2), (), Real, data[1, 2])
    check_funsor(x(i=1, j=2, k=3), (), Real, data[1, 2])
    check_funsor(x(i=1, k=3), {'j': Bint[5]}, Real, data[1])
    check_funsor(x(j=2), {'i': Bint[4]}, Real, data[:, 2])
    check_funsor(x(j=2, k=3), {'i': Bint[4]}, Real, data[:, 2])


def test_advanced_indexing_shape():
    I, J, M, N = 4, 4, 2, 3
    x = Tensor(randn((I, J)), OrderedDict([
        ('i', Bint[I]),
        ('j', Bint[J]),
    ]))
    m = Tensor(numeric_array([2, 3]), OrderedDict([('m', Bint[M])]), I)
    n = Tensor(numeric_array([0, 1, 1]), OrderedDict([('n', Bint[N])]), J)
    assert x.data.shape == (I, J)

    check_funsor(x(i=m), {'j': Bint[J], 'm': Bint[M]}, Real)
    check_funsor(x(i=m, j=n), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(i=m, j=n, k=m), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(i=m, k=m), {'j': Bint[J], 'm': Bint[M]}, Real)
    check_funsor(x(i=n), {'j': Bint[J], 'n': Bint[N]}, Real)
    check_funsor(x(i=n, k=m), {'j': Bint[J], 'n': Bint[N]}, Real)
    check_funsor(x(j=m), {'i': Bint[I], 'm': Bint[M]}, Real)
    check_funsor(x(j=m, i=n), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(j=m, i=n, k=m), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(j=m, k=m), {'i': Bint[I], 'm': Bint[M]}, Real)
    check_funsor(x(j=n), {'i': Bint[I], 'n': Bint[N]}, Real)
    check_funsor(x(j=n, k=m), {'i': Bint[I], 'n': Bint[N]}, Real)
    check_funsor(x(m), {'j': Bint[J], 'm': Bint[M]}, Real)
    check_funsor(x(m, j=n), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(m, j=n, k=m), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(m, k=m), {'j': Bint[J], 'm': Bint[M]}, Real)
    check_funsor(x(m, n), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(m, n, k=m), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(n), {'j': Bint[J], 'n': Bint[N]}, Real)
    check_funsor(x(n, k=m), {'j': Bint[J], 'n': Bint[N]}, Real)
    check_funsor(x(n, m), {'m': Bint[M], 'n': Bint[N]}, Real)
    check_funsor(x(n, m, k=m), {'m': Bint[M], 'n': Bint[N]}, Real)


def test_slice_simple():
    t = randn((3, 4, 5))
    f = Tensor(t)["i", "j"]
    assert_close(f, f(i=Slice("i", 3)))
    assert_close(f, f(j=Slice("j", 4)))
    assert_close(f, f(i=Slice("i", 3), j=Slice("j", 4)))
    assert_close(f, f(i=Slice("i", 3), j="j"))
    assert_close(f, f(i="i", j=Slice("j", 4)))


@pytest.mark.parametrize("stop", [0, 1, 2, 10])
def test_slice_1(stop):
    t = randn((10, 2))
    actual = Tensor(t)["i"](i=Slice("j", stop, dtype=10))
    expected = Tensor(t[:stop])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize("start", [0, 1, 2, 10])
@pytest.mark.parametrize("stop", [0, 1, 2, 10])
@pytest.mark.parametrize("step", [1, 2, 5, 10])
def test_slice_2(start, stop, step):
    t = randn((10, 2))
    actual = Tensor(t)["i"](i=Slice("j", start, stop, step, dtype=10))
    expected = Tensor(t[start: stop: step])["j"]
    assert_close(actual, expected)


def test_arange_simple():
    t = randn((3, 4, 5))
    f = Tensor(t)["i", "j"]
    assert_close(f, f(i=f.new_arange("i", 3)))
    assert_close(f, f(j=f.new_arange("j", 4)))
    assert_close(f, f(i=f.new_arange("i", 3), j=f.new_arange("j", 4)))
    assert_close(f, f(i=f.new_arange("i", 3), j="j"))
    assert_close(f, f(i="i", j=f.new_arange("j", 4)))


@pytest.mark.parametrize("stop", [0, 1, 2, 10])
def test_arange_1(stop):
    t = randn((10, 2))
    f = Tensor(t)["i"]
    actual = f(i=f.new_arange("j", stop, dtype=10))
    expected = Tensor(t[:stop])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize("start", [0, 1, 2, 10])
@pytest.mark.parametrize("stop", [0, 1, 2, 10])
@pytest.mark.parametrize("step", [1, 2, 5, 10])
def test_arange_2(start, stop, step):
    t = randn((10, 2))
    f = Tensor(t)["i"]
    actual = f(i=f.new_arange("j", start, stop, step, dtype=10))
    expected = Tensor(t[start: stop: step])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
def test_advanced_indexing_tensor(output_shape):
    #      u   v
    #     / \ / \
    #    i   j   k
    #     \  |  /
    #      \ | /
    #        x
    output = Reals[output_shape]
    x = random_tensor(OrderedDict([
        ('i', Bint[2]),
        ('j', Bint[3]),
        ('k', Bint[4]),
    ]), output)
    i = random_tensor(OrderedDict([
        ('u', Bint[5]),
    ]), Bint[2])
    j = random_tensor(OrderedDict([
        ('v', Bint[6]),
        ('u', Bint[5]),
    ]), Bint[3])
    k = random_tensor(OrderedDict([
        ('v', Bint[6]),
    ]), Bint[4])

    expected_data = empty((5, 6) + output_shape)
    for u in range(5):
        for v in range(6):
            expected_data[u, v] = x.data[i.data[u], j.data[v, u], k.data[v]]
    expected = Tensor(expected_data, OrderedDict([
        ('u', Bint[5]),
        ('v', Bint[6]),
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


@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
def test_advanced_indexing_lazy(output_shape):
    x = Tensor(randn((2, 3, 4) + output_shape), OrderedDict([
        ('i', Bint[2]),
        ('j', Bint[3]),
        ('k', Bint[4]),
    ]))
    u = Variable('u', Bint[2])
    v = Variable('v', Bint[3])
    with interpretation(lazy):
        i = Number(1, 2) - u
        j = Number(2, 3) - v
        k = u + v

    expected_data = empty((2, 3) + output_shape)
    i_data = x.materialize(i).data
    j_data = x.materialize(j).data
    k_data = x.materialize(k).data
    for u in range(2):
        for v in range(3):
            expected_data[u, v] = x.data[i_data[u], j_data[v], k_data[u, v]]
    expected = Tensor(expected_data, OrderedDict([
        ('u', Bint[2]),
        ('v', Bint[3]),
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


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(x, symbol)()


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b')])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'atanh', 'sqrt', 'exp', 'log', 'log1p', 'sigmoid', 'tanh',
])
def test_unary(symbol, dims):
    sizes = {'a': 3, 'b': 4}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, Bint[sizes[d]]) for d in dims)
    dtype = 'real'
    data = rand(shape) + 0.5
    if symbol == '~':
        data = ops.astype(data, 'uint8')
        dtype = 2
    if symbol == 'atanh':
        data = ops.clamp(data, -0.99, 0.99)
    if get_backend() != "torch" and symbol in ["abs", "atanh", "sqrt", "exp", "log", "log1p", "sigmoid", "tanh"]:
        expected_data = getattr(ops, symbol)(data)
    else:
        expected_data = unary_eval(symbol, data)

    x = Tensor(data, inputs, dtype)
    actual = unary_eval(symbol, x)
    check_funsor(actual, inputs, Array[dtype, ()], expected_data)


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
    inputs1 = OrderedDict((d, Bint[sizes[d]]) for d in dims1)
    inputs2 = OrderedDict((d, Bint[sizes[d]]) for d in dims2)
    data1 = rand(shape1) + 0.5
    data2 = rand(shape2) + 0.5
    dtype = 'real'
    if symbol in BOOLEAN_OPS:
        dtype = 2
        data1 = ops.astype(data1, 'uint8')
        data2 = ops.astype(data2, 'uint8')
    x1 = Tensor(data1, inputs1, dtype)
    x2 = Tensor(data2, inputs2, dtype)
    inputs, aligned = align_tensors(x1, x2)
    expected_data = binary_eval(symbol, aligned[0], aligned[1])

    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, inputs, Array[dtype, ()], expected_data)


@pytest.mark.parametrize('output_shape2', [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize('output_shape1', [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize('inputs2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')], ids=str)
@pytest.mark.parametrize('inputs1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')], ids=str)
def test_binary_broadcast(inputs1, inputs2, output_shape1, output_shape2):
    sizes = {'a': 4, 'b': 5, 'c': 6}
    inputs1 = OrderedDict((k, Bint[sizes[k]]) for k in inputs1)
    inputs2 = OrderedDict((k, Bint[sizes[k]]) for k in inputs2)
    x1 = random_tensor(inputs1, Reals[output_shape1])
    x2 = random_tensor(inputs1, Reals[output_shape2])

    actual = x1 + x2
    assert actual.output == find_domain(ops.add, x1.output, x2.output)

    block = {'a': 1, 'b': 2, 'c': 3}
    actual_block = actual(**block)
    expected_block = Tensor(x1(**block).data + x2(**block).data)
    assert_close(actual_block, expected_block)


@pytest.mark.parametrize('output_shape2', [(2,), (2, 5), (4, 2, 5)], ids=str)
@pytest.mark.parametrize('output_shape1', [(2,), (3, 2), (4, 3, 2)], ids=str)
@pytest.mark.parametrize('inputs2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')], ids=str)
@pytest.mark.parametrize('inputs1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')], ids=str)
def test_matmul(inputs1, inputs2, output_shape1, output_shape2):
    sizes = {'a': 6, 'b': 7, 'c': 8}
    inputs1 = OrderedDict((k, Bint[sizes[k]]) for k in inputs1)
    inputs2 = OrderedDict((k, Bint[sizes[k]]) for k in inputs2)
    x1 = random_tensor(inputs1, Reals[output_shape1])
    x2 = random_tensor(inputs1, Reals[output_shape2])

    actual = x1 @ x2
    assert actual.output == find_domain(ops.matmul, x1.output, x2.output)

    block = {'a': 1, 'b': 2, 'c': 3}
    actual_block = actual(**block)
    expected_block = Tensor(x1(**block).data @ x2(**block).data)
    assert_close(actual_block, expected_block, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_funsor_scalar(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, Bint[sizes[d]]) for d in dims)
    data1 = rand(shape) + 0.5
    expected_data = binary_eval(symbol, data1, scalar)

    x1 = Tensor(data1, inputs)
    actual = binary_eval(symbol, x1, scalar)
    check_funsor(actual, inputs, Real, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_scalar_funsor(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, Bint[sizes[d]]) for d in dims)
    data1 = rand(shape) + 0.5
    expected_data = binary_eval(symbol, scalar, data1)

    x1 = Tensor(data1, inputs)
    actual = binary_eval(symbol, scalar, x1)
    check_funsor(actual, inputs, Real, expected_data)


@pytest.mark.parametrize("batch_shape", [(), (5,), (4, 3)])
@pytest.mark.parametrize("old_shape,new_shape", [
    ((), ()),
    ((), (1,)),
    ((2,), (2, 1)),
    ((2,), (1, 2)),
    ((6,), (2, 3)),
    ((6,), (2, 1, 3)),
    ((2, 3, 2), (3, 2, 2)),
    ((2, 3, 2), (2, 2, 3)),
])
def test_reshape(batch_shape, old_shape, new_shape):
    inputs = OrderedDict(zip("abc", map(Bint.__getitem__, batch_shape)))
    old = random_tensor(inputs, Reals[old_shape])
    assert old.reshape(old.shape) is old

    new = old.reshape(new_shape)
    assert new.inputs == inputs
    assert new.shape == new_shape
    assert new.dtype == old.dtype

    old2 = new.reshape(old_shape)
    assert_close(old2, old)


def test_getitem_number_0_inputs():
    data = randn((5, 4, 3, 2))
    x = Tensor(data)
    assert_close(x[2], Tensor(data[2]))
    assert_close(x[:, 1], Tensor(data[:, 1]))
    assert_close(x[2, 1], Tensor(data[2, 1]))
    assert_close(x[2, :, 1], Tensor(data[2, :, 1]))
    assert_close(x[3, ...], Tensor(data[3, ...]))
    assert_close(x[3, 2, ...], Tensor(data[3, 2, ...]))
    assert_close(x[..., 1], Tensor(data[..., 1]))
    assert_close(x[..., 2, 1], Tensor(data[..., 2, 1]))
    assert_close(x[3, ..., 1], Tensor(data[3, ..., 1]))


def test_getitem_number_1_inputs():
    data = randn((3, 5, 4, 3, 2))
    inputs = OrderedDict([('i', Bint[3])])
    x = Tensor(data, inputs)
    assert_close(x[2], Tensor(data[:, 2], inputs))
    assert_close(x[:, 1], Tensor(data[:, :, 1], inputs))
    assert_close(x[2, 1], Tensor(data[:, 2, 1], inputs))
    assert_close(x[2, :, 1], Tensor(data[:, 2, :, 1], inputs))
    assert_close(x[3, ...], Tensor(data[:, 3, ...], inputs))
    assert_close(x[3, 2, ...], Tensor(data[:, 3, 2, ...], inputs))
    assert_close(x[..., 1], Tensor(data[..., 1], inputs))
    assert_close(x[..., 2, 1], Tensor(data[..., 2, 1], inputs))
    assert_close(x[3, ..., 1], Tensor(data[:, 3, ..., 1], inputs))


def test_getitem_number_2_inputs():
    data = randn((3, 4, 5, 4, 3, 2))
    inputs = OrderedDict([('i', Bint[3]), ('j', Bint[4])])
    x = Tensor(data, inputs)
    assert_close(x[2], Tensor(data[:, :, 2], inputs))
    assert_close(x[:, 1], Tensor(data[:, :, :, 1], inputs))
    assert_close(x[2, 1], Tensor(data[:, :, 2, 1], inputs))
    assert_close(x[2, :, 1], Tensor(data[:, :, 2, :, 1], inputs))
    assert_close(x[3, ...], Tensor(data[:, :, 3, ...], inputs))
    assert_close(x[3, 2, ...], Tensor(data[:, :, 3, 2, ...], inputs))
    assert_close(x[..., 1], Tensor(data[..., 1], inputs))
    assert_close(x[..., 2, 1], Tensor(data[..., 2, 1], inputs))
    assert_close(x[3, ..., 1], Tensor(data[:, :, 3, ..., 1], inputs))


def test_getitem_variable():
    data = randn((5, 4, 3, 2))
    x = Tensor(data)
    i = Variable('i', Bint[5])
    j = Variable('j', Bint[4])
    assert x[i] is Tensor(data, OrderedDict([('i', Bint[5])]))
    assert x[i, j] is Tensor(data, OrderedDict([('i', Bint[5]), ('j', Bint[4])]))


def test_getitem_string():
    data = randn((5, 4, 3, 2))
    x = Tensor(data)
    assert x['i'] is Tensor(data, OrderedDict([('i', Bint[5])]))
    assert x['i', 'j'] is Tensor(data, OrderedDict([('i', Bint[5]), ('j', Bint[4])]))


def test_getitem_tensor():
    data = randn((5, 4, 3, 2))
    x = Tensor(data)
    i = Variable('i', Bint[5])
    j = Variable('j', Bint[4])
    k = Variable('k', Bint[3])
    m = Variable('m', Bint[2])

    y = random_tensor(OrderedDict(), Bint[5])
    assert_close(x[i](i=y), x[y])

    y = random_tensor(OrderedDict(), Bint[4])
    assert_close(x[:, j](j=y), x[:, y])

    y = random_tensor(OrderedDict(), Bint[3])
    assert_close(x[:, :, k](k=y), x[:, :, y])

    y = random_tensor(OrderedDict(), Bint[2])
    assert_close(x[:, :, :, m](m=y), x[:, :, :, y])

    y = random_tensor(OrderedDict([('i', i.output)]),
                      Bint[j.dtype])
    assert_close(x[i, j](j=y), x[i, y])

    y = random_tensor(OrderedDict([('i', i.output), ('j', j.output)]),
                      Bint[k.dtype])
    assert_close(x[i, j, k](k=y), x[i, j, y])


def test_lambda_getitem():
    data = randn((2,))
    x = Tensor(data)
    y = Tensor(data, OrderedDict(i=Bint[2]))
    i = Variable('i', Bint[2])
    assert x[i] is y
    assert Lambda(i, y) is x


REDUCE_OPS = [
    ops.add,
    ops.mul,
    ops.and_,
    ops.or_,
    ops.logaddexp,
    ops.sample,
    ops.min,
    ops.max,
]


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op', REDUCE_OPS, ids=str)
def test_reduce_all(dims, op):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, Bint[sizes[d]]) for d in dims)
    data = rand(shape) + 0.5
    if op in [ops.and_, ops.or_]:
        data = ops.astype(data, 'uint8')
    expected_data = REDUCE_OP_TO_NUMERIC[op](data, None)

    x = Tensor(data, inputs)
    actual = x.reduce(op)
    check_funsor(actual, {}, Real, expected_data)


@pytest.mark.parametrize('dims,reduced_vars', [
    (dims, reduced_vars)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for num_reduced in range(len(dims) + 2)
    for reduced_vars in itertools.combinations(dims, num_reduced)
])
@pytest.mark.parametrize('op', REDUCE_OPS)
def test_reduce_subset(dims, reduced_vars, op):
    reduced_vars = frozenset(reduced_vars)
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, Bint[sizes[d]]) for d in dims)
    data = rand(shape) + 0.5
    dtype = 'real'
    if op in [ops.and_, ops.or_]:
        data = ops.astype(data, 'uint8')
        dtype = 2
    x = Tensor(data, inputs, dtype)
    actual = x.reduce(op, reduced_vars)
    expected_inputs = OrderedDict(
        (d, Bint[sizes[d]]) for d in dims if d not in reduced_vars)

    reduced_vars &= frozenset(dims)
    if not reduced_vars:
        assert actual is x
    else:
        if reduced_vars == frozenset(dims):
            data = REDUCE_OP_TO_NUMERIC[op](data, None)
        else:
            for pos in reversed(sorted(map(dims.index, reduced_vars))):
                data = REDUCE_OP_TO_NUMERIC[op](data, pos)
        check_funsor(actual, expected_inputs, Array[dtype, ()])
        assert_close(actual, Tensor(data, expected_inputs, dtype),
                     atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('event_shape', [(), (4,), (2, 3)])
@pytest.mark.parametrize('op', REDUCE_OPS, ids=str)
def test_reduce_event(op, event_shape, dims):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    batch_shape = tuple(sizes[d] for d in dims)
    shape = batch_shape + event_shape
    inputs = OrderedDict((d, Bint[sizes[d]]) for d in dims)
    numeric_op = REDUCE_OP_TO_NUMERIC[op]
    data = rand(shape) + 0.5
    dtype = 'real'
    if op in [ops.and_, ops.or_]:
        data = ops.astype(data, 'uint8')
    expected_data = numeric_op(data.reshape(batch_shape + (-1,)), -1)

    x = Tensor(data, inputs, dtype=dtype)
    op_name = numeric_op.__name__[1:] if op in [ops.min, ops.max] else numeric_op.__name__
    actual = getattr(x, op_name)()
    check_funsor(actual, inputs, Array[dtype, ()], expected_data)


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)])
def test_all_equal(shape):
    inputs = OrderedDict()
    data1 = rand(shape) + 0.5
    data2 = rand(shape) + 0.5
    dtype = 'real'

    x1 = Tensor(data1, inputs, dtype=dtype)
    x2 = Tensor(data2, inputs, dtype=dtype)
    assert (x1 == x1).all()
    assert (x2 == x2).all()
    assert not (x1 == x2).all()
    assert not (x1 != x1).any()
    assert not (x2 != x2).any()
    assert (x1 != x2).any()


def test_function_hint_matmul():
    @funsor.function
    def matmul(x: Reals[3, 4], y: Reals[4, 5]) -> Reals[3, 5]:
        return x @ y

    assert get_type_hints(matmul) == get_type_hints(matmul.fn)

    check_funsor(matmul, {'x': Reals[3, 4], 'y': Reals[4, 5]}, Reals[3, 5])

    x = Tensor(randn((3, 4)))
    y = Tensor(randn((4, 5)))
    actual = matmul(x, y)
    expected_data = x.data @ y.data
    check_funsor(actual, {}, Reals[3, 5], expected_data)


def test_function_matmul():
    @funsor.function(Reals[3, 4], Reals[4, 5], Reals[3, 5])
    def matmul(x, y):
        return x @ y

    check_funsor(matmul, {'x': Reals[3, 4], 'y': Reals[4, 5]}, Reals[3, 5])

    x = Tensor(randn((3, 4)))
    y = Tensor(randn((4, 5)))
    actual = matmul(x, y)
    expected_data = x.data @ y.data
    check_funsor(actual, {}, Reals[3, 5], expected_data)


def test_function_lazy_matmul():

    @funsor.function(Reals[3, 4], Reals[4, 5], Reals[3, 5])
    def matmul(x, y):
        return x @ y

    x_lazy = Variable('x', Reals[3, 4])
    y = Tensor(randn((4, 5)))
    actual_lazy = matmul(x_lazy, y)
    check_funsor(actual_lazy, {'x': Reals[3, 4]}, Reals[3, 5])
    assert isinstance(actual_lazy, funsor.tensor.Function)

    x = Tensor(randn((3, 4)))
    actual = actual_lazy(x=x)
    expected_data = x.data @ y.data
    check_funsor(actual, {}, Reals[3, 5], expected_data)


def _numeric_max_and_argmax(x):
    if get_backend() == "torch":
        import torch

        return torch.max(x, dim=-1)
    else:
        return np.max(x, axis=-1), np.argmax(x, axis=-1)


def test_function_nested_eager_hint():

    @funsor.function
    def max_and_argmax(x: Reals[8]) -> Product[Real, Bint[8]]:
        return tuple(_numeric_max_and_argmax(x))

    expected = {"x": Reals[8], "return": Product[Real, Bint[8]]}
    assert get_type_hints(max_and_argmax) == expected

    inputs = OrderedDict([('i', Bint[2]), ('j', Bint[3])])
    x = Tensor(randn((2, 3, 8)), inputs)
    m, a = _numeric_max_and_argmax(x.data)
    expected_max = Tensor(m, inputs, 'real')
    expected_argmax = Tensor(a, inputs, 8)

    actual_max, actual_argmax = max_and_argmax(x)
    assert_close(actual_max, expected_max)
    assert_close(actual_argmax, expected_argmax)


def test_function_nested_eager():

    @funsor.function(Reals[8], (Real, Bint[8]))
    def max_and_argmax(x):
        return tuple(_numeric_max_and_argmax(x))

    inputs = OrderedDict([('i', Bint[2]), ('j', Bint[3])])
    x = Tensor(randn((2, 3, 8)), inputs)
    m, a = _numeric_max_and_argmax(x.data)
    expected_max = Tensor(m, inputs, 'real')
    expected_argmax = Tensor(a, inputs, 8)

    actual_max, actual_argmax = max_and_argmax(x)
    assert_close(actual_max, expected_max)
    assert_close(actual_argmax, expected_argmax)


def test_function_nested_lazy():

    @funsor.function(Reals[8], (Real, Bint[8]))
    def max_and_argmax(x):
        return tuple(_numeric_max_and_argmax(x))

    x_lazy = Variable('x', Reals[8])
    lazy_max, lazy_argmax = max_and_argmax(x_lazy)
    assert isinstance(lazy_max, funsor.tensor.Function)
    assert isinstance(lazy_argmax, funsor.tensor.Function)
    check_funsor(lazy_max, {'x': Reals[8]}, Real)
    check_funsor(lazy_argmax, {'x': Reals[8]}, Bint[8])

    inputs = OrderedDict([('i', Bint[2]), ('j', Bint[3])])
    y = Tensor(randn((2, 3, 8)), inputs)
    actual_max = lazy_max(x=y)
    actual_argmax = lazy_argmax(x=y)
    expected_max, expected_argmax = max_and_argmax(y)
    assert_close(actual_max, expected_max)
    assert_close(actual_argmax, expected_argmax)


def test_function_of_numeric_array():
    backend = get_backend()
    if backend == "torch":
        import torch

        matmul = torch.matmul
    elif backend == "jax":
        import jax

        matmul = jax.numpy.matmul
    else:
        matmul = np.matmul
    x = randn((4, 3))
    y = randn((3, 2))
    f = funsor.function(Reals[4, 3], Reals[3, 2], Reals[4, 2])(matmul)
    actual = f(x, y)
    expected = f(Tensor(x), Tensor(y))
    assert_close(actual, expected)


def test_align():
    x = Tensor(randn((2, 3, 4)), OrderedDict([
        ('i', Bint[2]),
        ('j', Bint[3]),
        ('k', Bint[4]),
    ]))
    y = x.align(('j', 'k', 'i'))
    assert isinstance(y, Tensor)
    assert tuple(y.inputs) == ('j', 'k', 'i')
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert x(i=i, j=j, k=k) == y(i=i, j=j, k=k)


EINSUM_EXAMPLES = [
    'a->a',
    'a,a->a',
    'a,b->',
    'a,b->a',
    'a,b->b',
    'a,b->ab',
    'a,b->ba',
    'ab,ba->',
    'ab,ba->a',
    'ab,ba->b',
    'ab,ba->ab',
    'ab,ba->ba',
    'ab,bc->ac',
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
def test_einsum(equation):
    sizes = dict(a=2, b=3, c=4)
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    tensors = [randn(tuple(sizes[d] for d in dims)) for dims in inputs]
    funsors = [Tensor(x) for x in tensors]
    expected = Tensor(ops.einsum(equation, *tensors))
    actual = Einsum(equation, tuple(funsors))
    assert_close(actual, expected, atol=1e-5, rtol=None)


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('batch1', [''])
@pytest.mark.parametrize('batch2', [''])
def test_batched_einsum(equation, batch1, batch2):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')

    sizes = dict(a=2, b=3, c=4, i=5, j=6)
    batch1 = OrderedDict([(k, Bint[sizes[k]]) for k in batch1])
    batch2 = OrderedDict([(k, Bint[sizes[k]]) for k in batch2])
    funsors = [random_tensor(batch, Reals[tuple(sizes[d] for d in dims)])
               for batch, dims in zip([batch1, batch2], inputs)]
    actual = Einsum(equation, tuple(funsors))

    _equation = ','.join('...' + i for i in inputs) + '->...' + output
    inputs, tensors = align_tensors(*funsors)
    batch = tuple(v.size for v in inputs.values())
    tensors = [ops.expand(x, batch + f.shape) for (x, f) in zip(tensors, funsors)]
    expected = Tensor(ops.einsum(_equation, *tensors), inputs)
    assert_close(actual, expected, atol=1e-5, rtol=None)


def _numeric_tensordot(x, y, dim):
    if get_backend() == "torch":
        import torch

        return torch.tensordot(x, y, dim)
    else:
        return np.tensordot(x, y, axes=dim)


@pytest.mark.parametrize('y_shape', [(), (4,), (4, 5)], ids=str)
@pytest.mark.parametrize('xy_shape', [(), (6,), (6, 7)], ids=str)
@pytest.mark.parametrize('x_shape', [(), (2,), (2, 3)], ids=str)
def test_tensor_tensordot(x_shape, xy_shape, y_shape):
    x = randn(x_shape + xy_shape)
    y = randn(xy_shape + y_shape)
    dim = len(xy_shape)
    actual = tensordot(Tensor(x), Tensor(y), dim)
    expected = Tensor(_numeric_tensordot(x, y, dim))
    assert_close(actual, expected, atol=1e-5, rtol=None)


@pytest.mark.parametrize('n', [1, 2, 5])
@pytest.mark.parametrize('shape,dim', [
    ((), 0),
    ((), -1),
    ((1,), 0),
    ((1,), 1),
    ((1,), -1),
    ((1,), -2),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), 2),
    ((2, 3), -1),
    ((2, 3), -2),
    ((2, 3), -3),
], ids=str)
def test_tensor_stack(n, shape, dim):
    tensors = [randn(shape) for _ in range(n)]
    actual = stack(tuple(Tensor(t) for t in tensors), dim=dim)
    expected = Tensor(ops.stack(dim, *tensors))
    assert_close(actual, expected)


@pytest.mark.parametrize('output', [Bint[2], Real, Reals[4], Reals[2, 3]], ids=str)
def test_funsor_stack(output):
    x = random_tensor(OrderedDict([
        ('i', Bint[2]),
    ]), output)
    y = random_tensor(OrderedDict([
        ('j', Bint[3]),
    ]), output)
    z = random_tensor(OrderedDict([
        ('i', Bint[2]),
        ('k', Bint[4]),
    ]), output)

    xy = Stack('t', (x, y))
    assert isinstance(xy, Tensor)
    assert xy.inputs == OrderedDict([
        ('t', Bint[2]),
        ('i', Bint[2]),
        ('j', Bint[3]),
    ])
    assert xy.output == output
    for j in range(3):
        assert_close(xy(t=0, j=j), x)
    for i in range(2):
        assert_close(xy(t=1, i=i), y)

    xyz = Stack('t', (x, y, z))
    assert isinstance(xyz, Tensor)
    assert xyz.inputs == OrderedDict([
        ('t', Bint[3]),
        ('i', Bint[2]),
        ('j', Bint[3]),
        ('k', Bint[4]),
    ])
    assert xy.output == output
    for j in range(3):
        for k in range(4):
            assert_close(xyz(t=0, j=j, k=k), x)
    for i in range(2):
        for k in range(4):
            assert_close(xyz(t=1, i=i, k=k), y)
    for j in range(3):
        assert_close(xyz(t=2, j=j), z)


@pytest.mark.parametrize('output', [Bint[2], Real, Reals[4], Reals[2, 3]], ids=str)
def test_cat_simple(output):
    x = random_tensor(OrderedDict([
        ('i', Bint[2]),
    ]), output)
    y = random_tensor(OrderedDict([
        ('i', Bint[3]),
        ('j', Bint[4]),
    ]), output)
    z = random_tensor(OrderedDict([
        ('i', Bint[5]),
        ('k', Bint[6]),
    ]), output)

    assert Cat('i', (x,)) is x
    assert Cat('i', (y,)) is y
    assert Cat('i', (z,)) is z

    xy = Cat('i', (x, y))
    assert isinstance(xy, Tensor)
    assert xy.inputs == OrderedDict([
        ('i', Bint[2 + 3]),
        ('j', Bint[4]),
    ])
    assert xy.output == output

    xyz = Cat('i', (x, y, z))
    assert isinstance(xyz, Tensor)
    assert xyz.inputs == OrderedDict([
        ('i', Bint[2 + 3 + 5]),
        ('j', Bint[4]),
        ('k', Bint[6]),
    ])
    assert xy.output == output


@pytest.mark.parametrize("expand_shape", [(4, 3, 2), (4, -1, 2), (4, 3, -1), (4, -1, -1)])
def test_ops_expand(expand_shape):
    x = randn((3, 2))
    actual = ops.expand(x, expand_shape)
    assert actual.shape == (4, 3, 2)


def test_tensor_to_funsor_ambiguous_output():
    x = randn((2, 1))
    f = funsor.to_funsor(x, output=None, dim_to_name=OrderedDict({-2: 'a'}))
    f2 = funsor.to_funsor(x, output=Real, dim_to_name=OrderedDict({-2: 'a'}))
    assert f.inputs == f2.inputs == OrderedDict(a=Bint[2])
    assert f.output.shape == () == f2.output.shape


@pytest.mark.skipif(get_backend() != "torch", reason="torch-specific regression")
def test_log_correct_dtype():
    import torch
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    x = Tensor(torch.rand(3, dtype=torch.get_default_dtype()))
    try:
        assert (x == x).all().log().data.dtype is x.data.dtype
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.skipif(get_backend() != "numpy", reason="backend-specific")
def test_pickle():
    x = Tensor(randn(2, 3))
    f = io.BytesIO()
    pickle.dump(x, f)
    f.seek(0)
    y = pickle.load(f)
    assert_close(x, y)


@pytest.mark.skipif(get_backend() != "torch", reason="backend-specific")
def test_torch_save():
    import torch
    x = Tensor(randn(2, 3))
    f = io.BytesIO()
    torch.save(x, f)
    f.seek(0)
    y = torch.load(f)
    assert_close(x, y)


@pytest.mark.skipif(get_backend() != "torch", reason="backend-specific")
def test_detach():
    import torch
    try:
        from pyro.distributions.util import detach
    except ImportError:
        pytest.skip("detach() is not available")
    x = Tensor(torch.randn(2, 3, requires_grad=True))
    y = detach(x)
    assert_close(x, y)
    assert x.data.requires_grad
    assert not y.data.requires_grad


def test_diagonal_rename():
    x = Tensor(randn(2, 2, 3), OrderedDict(a=funsor.Bint[2], b=funsor.Bint[2], c=funsor.Bint[3]), 'real')
    d = Variable("d", funsor.Bint[2])
    dt = x.materialize(d)
    yt = x(a=dt, b=dt)
    y = x(a=d, b=d)
    assert_close(y, yt)


def test_empty_tensor_possible():
    funsor.to_funsor(randn(3, 0), dim_to_name=OrderedDict([(-1, "a"), (-2, "b")]))
