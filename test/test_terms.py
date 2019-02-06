from __future__ import absolute_import, division, print_function

import itertools

import pytest
import torch

import funsor
from funsor.terms import align_tensors


def check_funsor(x, dims, shape, data=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, funsor.Funsor)
    assert set(x.dims) == set(dims)
    if shape is not None:
        assert dict(zip(x.dims, x.shape)) == dict(zip(dims, shape))
    if data is not None:
        if x.dims != dims:
            data = data.permute(tuple(dims.index(d) for d in x.dims))
        assert (x.data == data).all()


def test_materialize():

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    g = f.materialize()

    assert g.dims == f.dims
    assert g.shape == f.shape
    for key in itertools.product(*map(range, g.shape)):
        assert f[key] == g[key]


def test_to_funsor():
    assert isinstance(funsor.to_funsor(2), funsor.Number)
    assert isinstance(funsor.to_funsor(2.), funsor.Number)
    assert isinstance(funsor.to_funsor(torch.tensor(2)), funsor.Tensor)
    assert isinstance(funsor.to_funsor(torch.tensor(2.)), funsor.Tensor)


def test_cons_hash():
    assert funsor.Variable('x', 3) is funsor.Variable('x', 3)
    assert funsor.Variable('x', 'real') is funsor.Variable('x', 'real')
    assert funsor.Variable('x', 'real') is not funsor.Variable('x', 3)
    assert funsor.Number(0) is funsor.Number(0)
    assert funsor.Number(0.) is funsor.Number(0.)
    assert funsor.Number(0.) is not funsor.Number(0)

    x = torch.randn(3, 3)
    assert funsor.Tensor(('i', 'j'), x) is funsor.Tensor(('i', 'j'), x)

    @funsor.of_shape('real', 2, 2)
    def f1(x, i, j):
        return (x ** i + j).sum('i')

    @funsor.of_shape('real', 2, 2)
    def f2(x, i, j):
        return (x ** i + j).sum('i')

    assert f1 is f2


@pytest.mark.parametrize('materialize_f', [False, True])
@pytest.mark.parametrize('materialize_g', [False, True])
@pytest.mark.parametrize('materialize_h', [False, True])
def test_mm(materialize_f, materialize_g, materialize_h):

    @funsor.of_shape(3, 4)
    def f(i, j):
        return i + j

    if materialize_f:
        f = f.materialize()
        assert isinstance(f, funsor.Tensor)
    check_funsor(f, ('i', 'j'), (3, 4))

    @funsor.of_shape(4, 5)
    def g(j, k):
        return j + k

    if materialize_g:
        g = g.materialize()
        assert isinstance(g, funsor.Tensor)
    check_funsor(g, ('j', 'k'), (4, 5))

    h = (f * g).sum('j')
    if materialize_h:
        h = h.materialize()
        assert isinstance(h, funsor.Tensor)
    check_funsor(h, ('i', 'k'), (3, 5))
    for i in range(3):
        for k in range(5):
            assert h[i, k].materialize() == sum(f[i, j] * g[j, k] for j in range(4))


@pytest.mark.parametrize('size', [3, 'real', 'density'])
def test_variable(size):
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


def test_indexing():
    data = torch.randn(4, 5)
    x = funsor.Tensor(('i', 'j'), data)
    check_funsor(x, ('i', 'j'), (4, 5), data)

    assert x() is x
    assert x(k=3) is x
    check_funsor(x(1), ['j'], [5], data[1])
    check_funsor(x(1, 2), (), (), data[1, 2])
    check_funsor(x(1, 2, k=3), (), (), data[1, 2])
    check_funsor(x(1, j=2), (), (), data[1, 2])
    check_funsor(x(1, j=2, k=3), (), (), data[1, 2])
    check_funsor(x(1, k=3), ['j'], [5], data[1])
    check_funsor(x(i=1), ('j',), (5,), data[1])
    check_funsor(x(i=1, j=2), (), (), data[1, 2])
    check_funsor(x(i=1, j=2, k=3), (), (), data[1, 2])
    check_funsor(x(i=1, k=3), ('j',), (5,), data[1])
    check_funsor(x(j=2), ('i',), (4,), data[:, 2])
    check_funsor(x(j=2, k=3), ('i',), (4,), data[:, 2])

    assert x[:] is x
    assert x[:, :] is x
    check_funsor(x[0, 0], (), (), data[0, 0])
    check_funsor(x[0], ('j',), (5,), data[0])
    check_funsor(x[:, 0], ('i',), (4,), data[:, 0])


def test_advanced_indexing():
    I, J, M, N = 4, 5, 2, 3
    x = funsor.Tensor(('i', 'j'), torch.randn(4, 5))
    m = funsor.Tensor(('m',), torch.tensor([2, 3]))
    n = funsor.Tensor(('n',), torch.tensor([0, 1, 1]))

    assert x.shape == (4, 5)

    check_funsor(x(i=m), ('j', 'm'), (J, M))
    check_funsor(x(i=m, j=n), ('m', 'n'), (M, N))
    check_funsor(x(i=m, j=n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(i=m, k=m), ('j', 'm'), (J, M))
    check_funsor(x(i=n), ('j', 'n'), (J, N))
    check_funsor(x(i=n, k=m), ('j', 'n'), (J, N))
    check_funsor(x(j=m), ('i', 'm'), (I, M))
    check_funsor(x(j=m, i=n), ('m', 'n'), (M, N))
    check_funsor(x(j=m, i=n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(j=m, k=m), ('i', 'm'), (I, M))
    check_funsor(x(j=n), ('i', 'n'), (I, N))
    check_funsor(x(j=n, k=m), ('i', 'n'), (I, N))
    check_funsor(x(m), ('j', 'm'), (J, M), x.data[m.data].t())
    check_funsor(x(m, j=n), ('m', 'n'), (M, N))
    check_funsor(x(m, j=n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(m, k=m), ('j', 'm'), (J, M), x.data[m.data].t())
    check_funsor(x(m, n), ('m', 'n'), (M, N))
    check_funsor(x(m, n, k=m), ('m', 'n'), (M, N))
    check_funsor(x(n), ('j', 'n'), (J, N), x.data[n.data].t())
    check_funsor(x(n, k=m), ('j', 'n'), (J, N), x.data[n.data].t())
    check_funsor(x(n, m), ('m', 'n'), (M, N))
    check_funsor(x(n, m, k=m), ('m', 'n'), (M, N))

    check_funsor(x[m], ('j', 'm'), (J, M), x.data[m.data].t())
    check_funsor(x[n], ('j', 'n'), (J, N), x.data[n.data].t())
    check_funsor(x[:, m], ('i', 'm'), (I, M))
    check_funsor(x[:, n], ('i', 'n'), (I, N))
    check_funsor(x[m, n], ('m', 'n'), (M, N))
    check_funsor(x[n, m], ('m', 'n'), (M, N))


def test_ellipsis():
    data = torch.randn(3, 4, 5)
    x = funsor.Tensor(('i', 'j', 'k'), data)
    check_funsor(x, ('i', 'j', 'k'), (3, 4, 5))

    assert x[...] is x
    check_funsor(x[..., 1, 2, 3], (), (), data[1, 2, 3])
    check_funsor(x[..., 2, 3], ('i',), (3,), data[..., 2, 3])
    check_funsor(x[..., 3], ('i', 'j'), (3, 4), data[..., 3])
    check_funsor(x[1, ..., 2, 3], (), (), data[1, 2, 3])
    check_funsor(x[1, ..., 3], ('j',), (4,), data[1, ..., 3])
    check_funsor(x[1, ...], ('j', 'k'), (4, 5), data[1])
    check_funsor(x[1, 2, ..., 3], (), (), data[1, 2, 3])
    check_funsor(x[1, 2, ...], ('k',), (5,), data[1, 2])
    check_funsor(x[1, 2, 3, ...], (), (), data[1, 2, 3])


def unary_eval(symbol, x):
    if symbol in ['~', '-']:
        return eval('{} x'.format(symbol))
    return getattr(x, symbol)()


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)])
@pytest.mark.parametrize('symbol', [
    '~', '-', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(symbol, shape):
    data = torch.rand(shape) + 0.5
    if symbol == '~':
        data = data.byte()
    expected_data = unary_eval(symbol, data)
    dims = tuple('abc'[:len(shape)])

    x = funsor.Tensor(dims, data)
    actual = unary_eval(symbol, x)
    check_funsor(actual, dims, shape, expected_data)


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
    dims = tuple(sorted(set(dims1 + dims2)))
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    data1 = torch.rand(shape1) + 0.5
    data2 = torch.rand(shape2) + 0.5
    if symbol in BOOLEAN_OPS:
        data1 = data1.byte()
        data2 = data2.byte()
    dims, aligned = align_tensors(funsor.Tensor(dims1, data1),
                                  funsor.Tensor(dims2, data2))
    expected_data = binary_eval(symbol, aligned[0], aligned[1])

    x1 = funsor.Tensor(dims1, data1)
    x2 = funsor.Tensor(dims2, data2)
    actual = binary_eval(symbol, x1, x2)
    check_funsor(actual, dims, expected_data.shape, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_funsor_scalar(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = binary_eval(symbol, data1, scalar)

    x1 = funsor.Tensor(dims, data1)
    actual = binary_eval(symbol, x1, scalar)
    check_funsor(actual, dims, shape, expected_data)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('symbol', BINARY_OPS)
def test_binary_scalar_funsor(symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data1 = torch.rand(shape) + 0.5
    expected_data = binary_eval(symbol, scalar, data1)

    x1 = funsor.Tensor(dims, data1)
    actual = binary_eval(symbol, scalar, x1)
    check_funsor(actual, dims, shape, expected_data)


REDUCE_OPS = ['sum', 'prod', 'logsumexp', 'all', 'any', 'min', 'max']


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name', REDUCE_OPS)
def test_reduce_all(dims, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        data = data.byte()
    if op_name == 'logsumexp':
        # work around missing torch.Tensor.logsumexp()
        expected_data = data.reshape(-1).logsumexp(0)
    else:
        expected_data = getattr(data, op_name)()

    x = funsor.Tensor(dims, data)
    actual = getattr(x, op_name)()
    check_funsor(actual, (), (), expected_data)


@pytest.mark.parametrize('dims,dims_reduced', [
    (dims, dims_reduced)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for num_reduced in range(len(dims) + 2)
    for dims_reduced in itertools.combinations(dims + ('z',), num_reduced)
])
@pytest.mark.parametrize('op_name', REDUCE_OPS)
def test_reduce_subset(dims, dims_reduced, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        data = data.byte()
    x = funsor.Tensor(dims, data)
    actual = getattr(x, op_name)(dims_reduced)

    dims_reduced = set(dims_reduced) & set(dims)
    if not dims_reduced:
        assert actual is x
    else:
        if dims_reduced == set(dims):
            if op_name == 'logsumexp':
                # work around missing torch.Tensor.logsumexp()
                data = data.reshape(-1).logsumexp(0)
            else:
                data = getattr(data, op_name)()
        else:
            for pos in reversed(sorted(map(dims.index, dims_reduced))):
                if op_name in ('min', 'max'):
                    data = getattr(data, op_name)(pos)[0]
                else:
                    data = getattr(data, op_name)(pos)
        dims = tuple(d for d in dims if d not in dims_reduced)
        shape = data.shape
        check_funsor(actual, dims, data.shape, data)


def test_of_shape():

    @funsor.of_shape(3)
    def f(i):
        return 0

    check_funsor(f, ('i',), (3,))

    @funsor.of_shape('real', 'real')
    def g(x, y):
        return y - x ** 2

    check_funsor(g, ('x', 'y'), ('real', 'real'))
