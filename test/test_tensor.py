# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
from collections import OrderedDict

import numpy as np
import pytest
import torch

import funsor
import funsor.ops as ops
from funsor.domains import Domain, bint, find_domain, reals
from funsor.interpreter import interpretation
from funsor.terms import Cat, Lambda, Number, Slice, Stack, Variable, lazy
from funsor.testing import assert_close, assert_equiv, astype, check_funsor, rand, randn, random_tensor
from funsor.tensor import REDUCE_OP_TO_NUMERIC, Einsum, Tensor, align_tensors, stack, tensordot


@pytest.mark.parametrize('output_shape', [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize('inputs', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')], ids=str)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_quote(output_shape, inputs, backend):
    sizes = {'a': 4, 'b': 5, 'c': 6}
    inputs = OrderedDict((k, bint(sizes[k])) for k in inputs)
    x = random_tensor(inputs, reals(*output_shape), backend)
    s = funsor.quote(x)
    assert isinstance(s, str)
    assert_close(eval(s), x)


@pytest.mark.parametrize('shape', [(), (4,), (3, 2)])
@pytest.mark.parametrize('dtype', [torch.float, torch.long, torch.uint8, torch.bool,
                                   np.float32, np.float64, np.int32, np.int64, np.uint8])
def test_to_funsor(shape, dtype):
    backend = "torch" if isinstance(dtype, torch.dtype) else "numpy"
    t = astype(randn(shape, backend), dtype)
    f = funsor.to_funsor(t)
    assert isinstance(f, Tensor)
    assert funsor.to_funsor(t, reals(*shape)) is f
    with pytest.raises(ValueError):
        funsor.to_funsor(t, reals(5, *shape))


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_to_data(backend):
    zeros = torch.zeros if backend == "torch" else np.zeros
    data = zeros((3, 3))
    x = Tensor(data)
    assert funsor.to_data(x) is data


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_to_data_error(backend):
    zeros = torch.zeros if backend == "torch" else np.zeros
    data = zeros((3, 3))
    x = Tensor(data, OrderedDict(i=bint(3)))
    with pytest.raises(ValueError):
        funsor.to_data(x)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_cons_hash(backend):
    x = randn((3, 3), backend)
    assert Tensor(x) is Tensor(x)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_indexing(backend):
    data = randn((4, 5), backend)
    inputs = OrderedDict([('i', bint(4)),
                          ('j', bint(5))])
    x = Tensor(data, inputs)
    check_funsor(x, inputs, reals(), data)

    assert x() is x
    assert x(k=3) is x
    check_funsor(x(1), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(1, 2), {}, reals(), data[1, 2])
    check_funsor(x(1, 2, k=3), {}, reals(), data[1, 2])
    check_funsor(x(1, j=2), {}, reals(), data[1, 2])
    check_funsor(x(1, j=2, k=3), (), reals(), data[1, 2])
    check_funsor(x(1, k=3), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(i=1), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(i=1, j=2), (), reals(), data[1, 2])
    check_funsor(x(i=1, j=2, k=3), (), reals(), data[1, 2])
    check_funsor(x(i=1, k=3), {'j': bint(5)}, reals(), data[1])
    check_funsor(x(j=2), {'i': bint(4)}, reals(), data[:, 2])
    check_funsor(x(j=2, k=3), {'i': bint(4)}, reals(), data[:, 2])


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_advanced_indexing_shape(backend):
    tensor = torch.tensor if backend == "torch" else np.array
    I, J, M, N = 4, 4, 2, 3
    x = Tensor(randn((I, J), backend), OrderedDict([
        ('i', bint(I)),
        ('j', bint(J)),
    ]))
    m = Tensor(tensor([2, 3]), OrderedDict([('m', bint(M))]), I)
    n = Tensor(tensor([0, 1, 1]), OrderedDict([('n', bint(N))]), J)
    assert x.data.shape == (I, J)

    check_funsor(x(i=m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(i=m, j=n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(i=m, j=n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(i=m, k=m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(i=n), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(i=n, k=m), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(j=m), {'i': bint(I), 'm': bint(M)}, reals())
    check_funsor(x(j=m, i=n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(j=m, i=n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(j=m, k=m), {'i': bint(I), 'm': bint(M)}, reals())
    check_funsor(x(j=n), {'i': bint(I), 'n': bint(N)}, reals())
    check_funsor(x(j=n, k=m), {'i': bint(I), 'n': bint(N)}, reals())
    check_funsor(x(m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(m, j=n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(m, j=n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(m, k=m), {'j': bint(J), 'm': bint(M)}, reals())
    check_funsor(x(m, n), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(m, n, k=m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(n), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(n, k=m), {'j': bint(J), 'n': bint(N)}, reals())
    check_funsor(x(n, m), {'m': bint(M), 'n': bint(N)}, reals())
    check_funsor(x(n, m, k=m), {'m': bint(M), 'n': bint(N)}, reals())


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_slice_simple(backend):
    t = randn((3, 4, 5), backend)
    f = Tensor(t)["i", "j"]
    assert_close(f, f(i=Slice("i", 3)))
    assert_close(f, f(j=Slice("j", 4)))
    assert_close(f, f(i=Slice("i", 3), j=Slice("j", 4)))
    assert_close(f, f(i=Slice("i", 3), j="j"))
    assert_close(f, f(i="i", j=Slice("j", 4)))


@pytest.mark.parametrize("stop", [0, 1, 2, 10])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_slice_1(stop, backend):
    t = randn((10, 2), backend)
    actual = Tensor(t)["i"](i=Slice("j", stop, dtype=10))
    expected = Tensor(t[:stop])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize("start", [0, 1, 2, 10])
@pytest.mark.parametrize("stop", [0, 1, 2, 10])
@pytest.mark.parametrize("step", [1, 2, 5, 10])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_slice_2(start, stop, step, backend):
    t = randn((10, 2), backend)
    actual = Tensor(t)["i"](i=Slice("j", start, stop, step, dtype=10))
    expected = Tensor(t[start: stop: step])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_arange_simple(backend):
    t = randn((3, 4, 5), backend)
    f = Tensor(t)["i", "j"]
    assert_close(f, f(i=f.new_arange("i", 3)))
    assert_close(f, f(j=f.new_arange("j", 4)))
    assert_close(f, f(i=f.new_arange("i", 3), j=f.new_arange("j", 4)))
    assert_close(f, f(i=f.new_arange("i", 3), j="j"))
    assert_close(f, f(i="i", j=f.new_arange("j", 4)))


@pytest.mark.parametrize("stop", [0, 1, 2, 10])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_arange_1(stop, backend):
    t = randn((10, 2), backend)
    f = Tensor(t)["i"]
    actual = f(i=f.new_arange("j", stop, dtype=10))
    expected = Tensor(t[:stop])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize("start", [0, 1, 2, 10])
@pytest.mark.parametrize("stop", [0, 1, 2, 10])
@pytest.mark.parametrize("step", [1, 2, 5, 10])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_arange_2(start, stop, step, backend):
    t = randn((10, 2), backend)
    f = Tensor(t)["i"]
    actual = f(i=f.new_arange("j", start, stop, step, dtype=10))
    expected = Tensor(t[start: stop: step])["j"]
    assert_close(actual, expected)


@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_advanced_indexing_tensor(output_shape, backend):
    empty = torch.empty if backend == "torch" else np.empty
    #      u   v
    #     / \ / \
    #    i   j   k
    #     \  |  /
    #      \ | /
    #        x
    output = reals(*output_shape)
    x = random_tensor(OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]), output, backend)
    i = random_tensor(OrderedDict([
        ('u', bint(5)),
    ]), bint(2), backend)
    j = random_tensor(OrderedDict([
        ('v', bint(6)),
        ('u', bint(5)),
    ]), bint(3), backend)
    k = random_tensor(OrderedDict([
        ('v', bint(6)),
    ]), bint(4), backend)

    expected_data = empty((5, 6) + output_shape)
    for u in range(5):
        for v in range(6):
            expected_data[u, v] = x.data[i.data[u], j.data[v, u], k.data[v]]
    expected = Tensor(expected_data, OrderedDict([
        ('u', bint(5)),
        ('v', bint(6)),
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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_advanced_indexing_lazy(output_shape, backend):
    empty = torch.empty if backend == "torch" else np.empty
    x = Tensor(randn((2, 3, 4) + output_shape, backend), OrderedDict([
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

    expected_data = empty((2, 3) + output_shape)
    i_data = x.materialize(i).data
    j_data = x.materialize(j).data
    k_data = x.materialize(k).data
    for u in range(2):
        for v in range(3):
            expected_data[u, v] = x.data[i_data[u], j_data[v], k_data[u, v]]
    expected = Tensor(expected_data, OrderedDict([
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


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(x, symbol)()


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b')])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p', 'sigmoid',
])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_unary(symbol, dims, backend):
    sizes = {'a': 3, 'b': 4}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    dtype = 'real'
    data = rand(shape) + 0.5
    if symbol == '~':
        data = astype(data, 'uint8')
        dtype = 2
    expected_data = unary_eval(symbol, data)

    x = Tensor(data, inputs, dtype)
    actual = unary_eval(symbol, x)
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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_binary_funsor_funsor(symbol, dims1, dims2, backend):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    inputs1 = OrderedDict((d, bint(sizes[d])) for d in dims1)
    inputs2 = OrderedDict((d, bint(sizes[d])) for d in dims2)
    data1 = rand(shape1, backend) + 0.5
    data2 = rand(shape2, backend) + 0.5
    dtype = 'real'
    if symbol in BOOLEAN_OPS:
        dtype = 2
        data1 = astype(data1, 'uint8')
        data2 = astype(data2, 'uint8')
    x1 = Tensor(data1, inputs1, dtype)
    x2 = Tensor(data2, inputs2, dtype)
    inputs, aligned = align_tensors(x1, x2)
    expected_data = binary_eval(symbol, aligned[0], aligned[1])

    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, inputs, Domain((), dtype), expected_data)


@pytest.mark.parametrize('output_shape2', [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize('output_shape1', [(), (2,), (3, 2)], ids=str)
@pytest.mark.parametrize('inputs2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')], ids=str)
@pytest.mark.parametrize('inputs1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')], ids=str)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_binary_broadcast(inputs1, inputs2, output_shape1, output_shape2, backend):
    sizes = {'a': 4, 'b': 5, 'c': 6}
    inputs1 = OrderedDict((k, bint(sizes[k])) for k in inputs1)
    inputs2 = OrderedDict((k, bint(sizes[k])) for k in inputs2)
    x1 = random_tensor(inputs1, reals(*output_shape1), backend)
    x2 = random_tensor(inputs1, reals(*output_shape2), backend)

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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_matmul(inputs1, inputs2, output_shape1, output_shape2, backend):
    sizes = {'a': 6, 'b': 7, 'c': 8}
    inputs1 = OrderedDict((k, bint(sizes[k])) for k in inputs1)
    inputs2 = OrderedDict((k, bint(sizes[k])) for k in inputs2)
    x1 = random_tensor(inputs1, reals(*output_shape1), backend)
    x2 = random_tensor(inputs1, reals(*output_shape2), backend)

    actual = x1 @ x2
    assert actual.output == find_domain(ops.matmul, x1.output, x2.output)

    block = {'a': 1, 'b': 2, 'c': 3}
    actual_block = actual(**block)
    expected_block = Tensor(x1(**block).data @ x2(**block).data)
    assert_close(actual_block, expected_block, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_binary_funsor_scalar(symbol, dims, scalar, backend):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data1 = rand(shape, backend) + 0.5
    expected_data = binary_eval(symbol, data1, scalar)

    x1 = Tensor(data1, inputs)
    actual = binary_eval(symbol, x1, scalar)
    check_funsor(actual, inputs, reals(), expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_binary_scalar_funsor(symbol, dims, scalar, backend):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data1 = rand(shape, backend) + 0.5
    expected_data = binary_eval(symbol, scalar, data1)

    x1 = Tensor(data1, inputs)
    actual = binary_eval(symbol, scalar, x1)
    check_funsor(actual, inputs, reals(), expected_data)


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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_reshape(batch_shape, old_shape, new_shape, backend):
    inputs = OrderedDict(zip("abc", map(bint, batch_shape)))
    old = random_tensor(inputs, reals(*old_shape), backend)
    assert old.reshape(old.shape) is old

    new = old.reshape(new_shape)
    assert new.inputs == inputs
    assert new.shape == new_shape
    assert new.dtype == old.dtype

    old2 = new.reshape(old_shape)
    assert_close(old2, old)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_getitem_number_0_inputs(backend):
    data = randn((5, 4, 3, 2), backend)
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


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_getitem_number_1_inputs(backend):
    data = randn((3, 5, 4, 3, 2), backend)
    inputs = OrderedDict([('i', bint(3))])
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


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_getitem_number_2_inputs(backend):
    data = randn((3, 4, 5, 4, 3, 2), backend)
    inputs = OrderedDict([('i', bint(3)), ('j', bint(4))])
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


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_getitem_variable(backend):
    data = randn((5, 4, 3, 2), backend)
    x = Tensor(data)
    i = Variable('i', bint(5))
    j = Variable('j', bint(4))
    assert x[i] is Tensor(data, OrderedDict([('i', bint(5))]))
    assert x[i, j] is Tensor(data, OrderedDict([('i', bint(5)), ('j', bint(4))]))


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_getitem_string(backend):
    data = randn((5, 4, 3, 2), backend)
    x = Tensor(data)
    assert x['i'] is Tensor(data, OrderedDict([('i', bint(5))]))
    assert x['i', 'j'] is Tensor(data, OrderedDict([('i', bint(5)), ('j', bint(4))]))


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_getitem_tensor(backend):
    data = randn((5, 4, 3, 2), backend)
    x = Tensor(data)
    i = Variable('i', bint(5))
    j = Variable('j', bint(4))
    k = Variable('k', bint(3))
    m = Variable('m', bint(2))

    y = random_tensor(OrderedDict(), bint(5), backend)
    assert_close(x[i](i=y), x[y])

    y = random_tensor(OrderedDict(), bint(4), backend)
    assert_close(x[:, j](j=y), x[:, y])

    y = random_tensor(OrderedDict(), bint(3), backend)
    assert_close(x[:, :, k](k=y), x[:, :, y])

    y = random_tensor(OrderedDict(), bint(2), backend)
    assert_close(x[:, :, :, m](m=y), x[:, :, :, y])

    y = random_tensor(OrderedDict([('i', i.output)]),
                      bint(j.dtype))
    assert_close(x[i, j](j=y), x[i, y])

    y = random_tensor(OrderedDict([('i', i.output), ('j', j.output)]),
                      bint(k.dtype))
    assert_close(x[i, j, k](k=y), x[i, j, y])


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_lambda_getitem(backend):
    data = randn((2,), backend)
    x = Tensor(data)
    y = Tensor(data, OrderedDict(i=bint(2)))
    i = Variable('i', bint(2))
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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_reduce_all(dims, op, backend):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data = rand(shape, backend) + 0.5
    if op in [ops.and_, ops.or_]:
        data = astype(data, 'uint8')
    expected_data = REDUCE_OP_TO_NUMERIC[op](data, None)

    x = Tensor(data, inputs)
    actual = x.reduce(op)
    check_funsor(actual, {}, reals(), expected_data)


@pytest.mark.parametrize('dims,reduced_vars', [
    (dims, reduced_vars)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for num_reduced in range(len(dims) + 2)
    for reduced_vars in itertools.combinations(dims, num_reduced)
])
@pytest.mark.parametrize('op', REDUCE_OPS)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_reduce_subset(dims, reduced_vars, op, backend):
    reduced_vars = frozenset(reduced_vars)
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    data = rand(shape, backend) + 0.5
    dtype = 'real'
    if op in [ops.and_, ops.or_]:
        data = astype(data, 'uint8')
        dtype = 2
    x = Tensor(data, inputs, dtype)
    actual = x.reduce(op, reduced_vars)
    expected_inputs = OrderedDict(
        (d, bint(sizes[d])) for d in dims if d not in reduced_vars)

    reduced_vars &= frozenset(dims)
    if not reduced_vars:
        assert actual is x
    else:
        if reduced_vars == frozenset(dims):
            data = REDUCE_OP_TO_NUMERIC[op](data, None)
        else:
            for pos in reversed(sorted(map(dims.index, reduced_vars))):
                data = REDUCE_OP_TO_NUMERIC[op](data, pos)
        check_funsor(actual, expected_inputs, Domain((), dtype))
        assert_close(actual, Tensor(data, expected_inputs, dtype),
                     atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('event_shape', [(), (4,), (2, 3)])
@pytest.mark.parametrize('op', REDUCE_OPS, ids=str)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_reduce_event(op, event_shape, dims, backend):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    batch_shape = tuple(sizes[d] for d in dims)
    shape = batch_shape + event_shape
    inputs = OrderedDict((d, bint(sizes[d])) for d in dims)
    numeric_op = REDUCE_OP_TO_NUMERIC[op]
    data = rand(shape, backend) + 0.5
    dtype = 'real'
    if op in [ops.and_, ops.or_]:
        data = astype(data, 'uint8')
    expected_data = numeric_op(data.reshape(batch_shape + (-1,)), -1)

    x = Tensor(data, inputs, dtype=dtype)
    op_name = numeric_op.__name__[1:] if op in [ops.min, ops.max] else numeric_op.__name__
    actual = getattr(x, op_name)()
    check_funsor(actual, inputs, Domain((), dtype), expected_data)


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_all_equal(shape, backend):
    inputs = OrderedDict()
    data1 = rand(shape, backend) + 0.5
    data2 = rand(shape, backend) + 0.5
    dtype = 'real'

    x1 = Tensor(data1, inputs, dtype=dtype)
    x2 = Tensor(data2, inputs, dtype=dtype)
    assert (x1 == x1).all()
    assert (x2 == x2).all()
    assert not (x1 == x2).all()
    assert not (x1 != x1).any()
    assert not (x2 != x2).any()
    assert (x1 != x2).any()


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_function_matmul(backend):
    _numeric_matmul = torch.matmul if backend == "torch" else np.matmul

    @funsor.function(reals(3, 4), reals(4, 5), reals(3, 5))
    def matmul(x, y):
        return _numeric_matmul(x, y)

    check_funsor(matmul, {'x': reals(3, 4), 'y': reals(4, 5)}, reals(3, 5))

    x = Tensor(randn((3, 4), backend))
    y = Tensor(randn((4, 5), backend))
    actual = matmul(x, y)
    expected_data = _numeric_matmul(x.data, y.data)
    check_funsor(actual, {}, reals(3, 5), expected_data)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_function_lazy_matmul(backend):
    _numeric_matmul = torch.matmul if backend == "torch" else np.matmul

    @funsor.function(reals(3, 4), reals(4, 5), reals(3, 5))
    def matmul(x, y):
        return _numeric_matmul(x, y)

    x_lazy = Variable('x', reals(3, 4))
    y = Tensor(torch.randn(4, 5))
    actual_lazy = matmul(x_lazy, y)
    check_funsor(actual_lazy, {'x': reals(3, 4)}, reals(3, 5))
    assert isinstance(actual_lazy, funsor.tensor.Function)

    x = Tensor(torch.randn(3, 4))
    actual = actual_lazy(x=x)
    expected_data = _numeric_matmul(x.data, y.data)
    check_funsor(actual, {}, reals(3, 5), expected_data)


def _numeric_max_and_argmax(x):
    if torch.is_tensor(x):
        return torch.max(x, dim=-1)
    else:
        assert isinstance(x, np.ndarray)
        return np.max(x, axis=-1), np.argmax(x, axis=-1)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_function_nested_eager(backend):

    @funsor.function(reals(8), (reals(), bint(8)))
    def max_and_argmax(x):
        return tuple(_numeric_max_and_argmax(x))

    inputs = OrderedDict([('i', bint(2)), ('j', bint(3))])
    x = Tensor(randn((2, 3, 8), backend), inputs)
    m, a = _numeric_max_and_argmax(x.data)
    expected_max = Tensor(m, inputs, 'real')
    expected_argmax = Tensor(a, inputs, 8)

    actual_max, actual_argmax = max_and_argmax(x)
    assert_close(actual_max, expected_max)
    assert_close(actual_argmax, expected_argmax)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_function_nested_lazy(backend):

    @funsor.function(reals(8), (reals(), bint(8)))
    def max_and_argmax(x):
        return tuple(_numeric_max_and_argmax(x))

    x_lazy = Variable('x', reals(8))
    lazy_max, lazy_argmax = max_and_argmax(x_lazy)
    assert isinstance(lazy_max, funsor.tensor.Function)
    assert isinstance(lazy_argmax, funsor.tensor.Function)
    check_funsor(lazy_max, {'x': reals(8)}, reals())
    check_funsor(lazy_argmax, {'x': reals(8)}, bint(8))

    inputs = OrderedDict([('i', bint(2)), ('j', bint(3))])
    y = Tensor(randn((2, 3, 8), backend), inputs)
    actual_max = lazy_max(x=y)
    actual_argmax = lazy_argmax(x=y)
    expected_max, expected_argmax = max_and_argmax(y)
    assert_close(actual_max, expected_max)
    assert_close(actual_argmax, expected_argmax)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_function_of_numeric_array(backend):
    _numeric_matmul = torch.matmul if backend == "torch" else np.matmul
    x = randn((4, 3), backend)
    y = randn((3, 2), backend)
    f = funsor.function(reals(4, 3), reals(3, 2), reals(4, 2))(_numeric_matmul)
    actual = f(x, y)
    expected = f(Tensor(x), Tensor(y))
    assert_close(actual, expected)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_align(backend):
    x = Tensor(randn((2, 3, 4), backend), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_einsum(equation, backend):
    einsum = torch.einsum if backend == "torch" else np.einsum
    sizes = dict(a=2, b=3, c=4)
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    tensors = [randn(tuple(sizes[d] for d in dims), backend) for dims in inputs]
    funsors = [Tensor(x) for x in tensors]
    expected = Tensor(einsum(equation, *tensors))
    actual = Einsum(equation, tuple(funsors))
    assert_close(actual, expected, atol=1e-5, rtol=None)


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('batch1', [''])
@pytest.mark.parametrize('batch2', [''])
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_batched_einsum(equation, batch1, batch2, backend):
    inputs, output = equation.split('->')
    inputs = inputs.split(',')

    sizes = dict(a=2, b=3, c=4, i=5, j=6)
    batch1 = OrderedDict([(k, bint(sizes[k])) for k in batch1])
    batch2 = OrderedDict([(k, bint(sizes[k])) for k in batch2])
    funsors = [random_tensor(batch, reals(*(sizes[d] for d in dims)), backend)
               for batch, dims in zip([batch1, batch2], inputs)]
    actual = Einsum(equation, tuple(funsors))

    _equation = ','.join('...' + i for i in inputs) + '->...' + output
    inputs, tensors = align_tensors(*funsors)
    batch = tuple(v.size for v in inputs.values())
    tensors = [ops.expand(x, batch + f.shape) for (x, f) in zip(tensors, funsors)]
    expected = Tensor(ops.einsum(_equation, *tensors), inputs)
    assert_close(actual, expected, atol=1e-5, rtol=None)


def _numeric_tensordot(x, y, dim):
    if torch.is_tensor(x):
        return torch.tensordot(x, y, dim)
    else:
        return np.tensordot(x, y, axes=dim)


@pytest.mark.parametrize('y_shape', [(), (4,), (4, 5)], ids=str)
@pytest.mark.parametrize('xy_shape', [(), (6,), (6, 7)], ids=str)
@pytest.mark.parametrize('x_shape', [(), (2,), (2, 3)], ids=str)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_tensor_tensordot(x_shape, xy_shape, y_shape, backend):
    x = randn(x_shape + xy_shape, backend)
    y = randn(xy_shape + y_shape, backend)
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
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_tensor_stack(n, shape, dim, backend):
    tensors = [randn(shape, backend) for _ in range(n)]
    actual = stack(tuple(Tensor(t) for t in tensors), dim=dim)
    expected = Tensor(ops.stack(dim, *tensors))
    assert_close(actual, expected)


@pytest.mark.parametrize('output', [bint(2), reals(), reals(4), reals(2, 3)], ids=str)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_funsor_stack(output, backend):
    x = random_tensor(OrderedDict([
        ('i', bint(2)),
    ]), output, backend)
    y = random_tensor(OrderedDict([
        ('j', bint(3)),
    ]), output, backend)
    z = random_tensor(OrderedDict([
        ('i', bint(2)),
        ('k', bint(4)),
    ]), output, backend)

    xy = Stack('t', (x, y))
    assert isinstance(xy, Tensor)
    assert xy.inputs == OrderedDict([
        ('t', bint(2)),
        ('i', bint(2)),
        ('j', bint(3)),
    ])
    assert xy.output == output
    for j in range(3):
        assert_close(xy(t=0, j=j), x)
    for i in range(2):
        assert_close(xy(t=1, i=i), y)

    xyz = Stack('t', (x, y, z))
    assert isinstance(xyz, Tensor)
    assert xyz.inputs == OrderedDict([
        ('t', bint(3)),
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
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


@pytest.mark.parametrize('output', [bint(2), reals(), reals(4), reals(2, 3)], ids=str)
@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_cat_simple(output, backend):
    x = random_tensor(OrderedDict([
        ('i', bint(2)),
    ]), output, backend)
    y = random_tensor(OrderedDict([
        ('i', bint(3)),
        ('j', bint(4)),
    ]), output, backend)
    z = random_tensor(OrderedDict([
        ('i', bint(5)),
        ('k', bint(6)),
    ]), output, backend)

    assert Cat('i', (x,)) is x
    assert Cat('i', (y,)) is y
    assert Cat('i', (z,)) is z

    xy = Cat('i', (x, y))
    assert isinstance(xy, Tensor)
    assert xy.inputs == OrderedDict([
        ('i', bint(2 + 3)),
        ('j', bint(4)),
    ])
    assert xy.output == output

    xyz = Cat('i', (x, y, z))
    assert isinstance(xyz, Tensor)
    assert xyz.inputs == OrderedDict([
        ('i', bint(2 + 3 + 5)),
        ('j', bint(4)),
        ('k', bint(6)),
    ])
    assert xy.output == output
