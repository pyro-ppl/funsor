from __future__ import absolute_import, division, print_function

import itertools

import pytest
import torch

import funsor


def check_funsor(x, dims, shape=None, tensor=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, funsor.Funsor)
    assert set(x.dims) == set(dims)
    if shape is not None:
        assert dict(zip(x.dims, x.shape)) == dict(zip(dims, shape))
    if tensor is not None:
        if x.dims != dims:
            tensor = tensor.permute(tuple(dims.index(d) for d in x.dims))
        assert (x.tensor == tensor).all()


@pytest.mark.parametrize('vectorize', [True, False])
def test_materialize(vectorize):

    @funsor.lazy(3, 4)
    def f(i, j):
        return i + j

    g = f.materialize(vectorize=vectorize)

    assert g.dims == f.dims
    assert g.shape == f.shape
    for key in itertools.product(*map(range, g.shape)):
        assert f[key] == g[key]


def test_contract():

    @funsor.lazy(3, 4)
    def f(i, j):
        return i + j

    assert f.dims == ("i", "j")
    assert f.shape == (3, 4)

    @funsor.lazy(4, 5)
    def g(j, k):
        return j + k

    assert g.dims == ("j", "k")
    assert g.shape == (4, 5)

    h = funsor.contract(("i", "k"), f, g)
    assert h.dims == ("i", "k")
    assert h.shape == (3, 5)
    for i in range(3):
        for k in range(5):
            assert h[i, k] == sum(f[i, j] * g[j, k] for j in range(4))


def test_indexing():
    tensor = torch.randn(4, 5)
    x = funsor.TorchFunsor(('i', 'j'), tensor)
    check_funsor(x, ('i', 'j'), (4, 5), tensor)

    assert x() is x
    check_funsor(x(1), ['j'], [5], tensor[1])
    check_funsor(x(1, 2), (), (), tensor[1, 2])
    check_funsor(x(i=1), ('j',), (5,), tensor[1])
    check_funsor(x(j=2), ('i',), (4,), tensor[:, 2])
    check_funsor(x(1, j=2), (), (), tensor[1, 2])
    check_funsor(x(i=1, j=2), (), (), tensor[1, 2])

    assert x[0].shape == (5,)
    assert x[0, 0].shape == ()
    assert x[:, 0].shape == (4,)
    assert x[:] is x
    assert x[:, :] is x


def test_advanced_indexing():
    I, J, M, N = 4, 5, 2, 3
    x = funsor.TorchFunsor(('i', 'j'), torch.randn(4, 5))
    m = funsor.TorchFunsor(('m',), torch.tensor([2, 3]))
    n = funsor.TorchFunsor(('n',), torch.tensor([0, 1, 1]))

    assert x.shape == (4, 5)

    check_funsor(x(m), ('j', 'm'), (J, M), x.tensor[m.tensor].t())
    check_funsor(x(n), ('j', 'n'), (J, N), x.tensor[n.tensor].t())
    check_funsor(x(m, n), ('m', 'n'), (M, N))
    check_funsor(x(n, m), ('m', 'n'), (M, N))
    check_funsor(x(i=m), ('j', 'm'), (J, M))
    check_funsor(x(i=n), ('j', 'n'), (J, N))
    check_funsor(x(j=m), ('i', 'm'), (I, M))
    check_funsor(x(j=n), ('i', 'n'), (I, N))
    check_funsor(x(i=m, j=n), ('m', 'n'), (M, N))
    check_funsor(x(j=m, i=n), ('m', 'n'), (M, N))
    check_funsor(x(m, j=n), ('m', 'n'), (M, N))

    check_funsor(x[m], ('j', 'm'), (J, M), x.tensor[m.tensor].t())
    check_funsor(x[n], ('j', 'n'), (J, N), x.tensor[n.tensor].t())
    check_funsor(x[:, m], ('i', 'm'), (I, M))
    check_funsor(x[:, n], ('i', 'n'), (I, N))
    check_funsor(x[m, n], ('m', 'n'), (M, N))
    check_funsor(x[n, m], ('m', 'n'), (M, N))


def test_ellipsis():
    tensor = torch.randn(3, 4, 5)
    x = funsor.TorchFunsor(('i', 'j', 'k'), tensor)
    check_funsor(x, ('i', 'j', 'k'), (3, 4, 5))

    assert x[...] is x

    check_funsor(x[..., 1, 2, 3], (), (), tensor[1, 2, 3])
    check_funsor(x[..., 2, 3], ('i',), (3,), tensor[..., 2, 3])
    check_funsor(x[..., 3], ('i', 'j'), (3, 4), tensor[..., 3])
    check_funsor(x[1, ..., 2, 3], (), (), tensor[1, 2, 3])
    check_funsor(x[1, ..., 3], ('j',), (4,), tensor[1, ..., 3])
    check_funsor(x[1, ...], ('j', 'k'), (4, 5), tensor[1])
    check_funsor(x[1, 2, ..., 3], (), (), tensor[1, 2, 3])
    check_funsor(x[1, 2, ...], ('k',), (5,), tensor[1, 2])
    check_funsor(x[1, 2, 3, ...], (), (), tensor[1, 2, 3])


@pytest.mark.parametrize('shape', [(), (4,), (2, 3)])
@pytest.mark.parametrize('op_name', [
    '__neg__', 'abs', 'sqrt', 'exp', 'log', 'log1p',
])
def test_unary(op_name, shape):
    tensor = torch.rand(shape) + 0.5
    expected_tensor = getattr(tensor, op_name)()
    dims = tuple('abc'[:len(shape)])

    x = funsor.TorchFunsor(dims, tensor)
    actual = getattr(x, op_name)()
    check_funsor(actual, dims, shape, expected_tensor)


BINARY_OPS = [
    ('__add__', '+'),
    ('__sub__', '-'),
    ('__mul__', '*'),
    ('__div__', '/'),
    ('__pow__', '**'),
    ('__eq__',  '=='),
    ('__ne__',  '!='),
]

BOOLEAN_OPS = [
    ('__and__', '&'),
    ('__or__',  '|'),
    ('__xor__', '^'),
]


@pytest.mark.parametrize('dims2', [(), ('a',), ('b', 'a'), ('b', 'c', 'a')])
@pytest.mark.parametrize('dims1', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name,symbol', BINARY_OPS + BOOLEAN_OPS)
def test_binary_funsor_funsor(op_name, symbol, dims1, dims2):
    dims = tuple(sorted(set(dims1 + dims2)))
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    shape1 = tuple(sizes[d] for d in dims1)
    shape2 = tuple(sizes[d] for d in dims2)
    tensor1 = torch.rand(shape1) + 0.5
    tensor2 = torch.rand(shape2) + 0.5
    if (op_name, symbol) in BOOLEAN_OPS:
        tensor1 = tensor1.byte()
        tensor2 = tensor2.byte()
    dims, tensors = funsor._align_tensors((dims1, tensor1),
                                          (dims2, tensor2))
    expected_tensor = eval('tensors[0] {} tensors[1]'.format(symbol))

    x1 = funsor.TorchFunsor(dims1, tensor1)  # noqa F841
    x2 = funsor.TorchFunsor(dims2, tensor2)  # noqa F841
    actual = eval('x1 {} x2'.format(symbol))
    check_funsor(actual, dims, shape, expected_tensor)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name,symbol', BINARY_OPS)
def test_binary_funsor_scalar(op_name, symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    tensor1 = torch.rand(shape) + 0.5
    expected_tensor = eval('tensor1 {} scalar'.format(symbol))

    x1 = funsor.TorchFunsor(dims, tensor1)  # noqa F841
    actual = eval('x1 {} scalar'.format(symbol))
    check_funsor(actual, dims, shape, expected_tensor)


@pytest.mark.parametrize('scalar', [0.5])
@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name,symbol', BINARY_OPS)
def test_binary_scalar_funsor(op_name, symbol, dims, scalar):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    tensor1 = torch.rand(shape) + 0.5
    expected_tensor = eval('scalar {} tensor1'.format(symbol))

    x1 = funsor.TorchFunsor(dims, tensor1)  # noqa F841
    actual = eval('scalar {} x1'.format(symbol))
    check_funsor(actual, dims, shape, expected_tensor)


@pytest.mark.parametrize('dims', [(), ('a',), ('a', 'b'), ('b', 'a', 'c')])
@pytest.mark.parametrize('op_name', ['sum', 'prod', 'all', 'any'])
def test_reduce_all(dims, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    tensor = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        tensor = tensor.byte()
    expected_tensor = getattr(tensor, op_name)()

    x = funsor.TorchFunsor(dims, tensor)
    actual = getattr(x, op_name)()
    check_funsor(actual, (), (), expected_tensor)


@pytest.mark.parametrize('dims,dim', [
    (dims, dim)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for dim in dims
])
@pytest.mark.parametrize('op_name', ['sum', 'prod', 'logsumexp', 'all', 'any'])
def test_reduce_one(dims, dim, op_name):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    tensor = torch.rand(shape) + 0.5
    if op_name in ['all', 'any']:
        tensor = tensor.byte()
    pos = dims.index(dim)
    expected_tensor = getattr(tensor, op_name)(pos)
    expected_dims = dims[:pos] + dims[1 + pos:]
    expected_shape = expected_tensor.shape

    x = funsor.TorchFunsor(dims, tensor)
    actual = getattr(x, op_name)(dim)
    check_funsor(actual, expected_dims, expected_shape, expected_tensor)
