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
        if x.shape != tensor.shape:
            raise NotImplementedError('TODO')
        assert (x.tensor == tensor).all()


def test_materialize():

    @funsor.lazy(3, 4)
    def f(i, j):
        return i + j

    g = f.materialize()

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


def test_index_1():
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


def test_advanced_index_1():
    I, J, M, N = 4, 5, 2, 3
    x = funsor.TorchFunsor(('i', 'j'), torch.randn(4, 5))
    m = funsor.TorchFunsor(('m',), torch.tensor([2, 3]))
    n = funsor.TorchFunsor(('n',), torch.tensor([0, 1, 1]))

    assert x.shape == (4, 5)

    check_funsor(x(m), ('j', 'm'), (J, M))
    check_funsor(x(n), ('j', 'n'), (J, N))
    check_funsor(x(m, n), ('m', 'n'), (M, N))
    check_funsor(x(n, m), ('m', 'n'), (M, N))
    check_funsor(x(i=m), ('j', 'm'), (J, M))
    check_funsor(x(i=n), ('j', 'n'), (J, N))
    check_funsor(x(j=m), ('i', 'm'), (I, M))
    check_funsor(x(j=n), ('i', 'n'), (I, N))
    check_funsor(x(i=m, j=n), ('m', 'n'), (M, N))
    check_funsor(x(j=m, i=n), ('m', 'n'), (M, N))
    check_funsor(x(m, j=n), ('m', 'n'), (M, N))

    check_funsor(x[m], ('j', 'm'), (J, M))
    check_funsor(x[n], ('j', 'n'), (J, N))
    check_funsor(x[:, m], ('i', 'm'), (I, M))
    check_funsor(x[:, n], ('i', 'n'), (I, N))
    check_funsor(x[m, n], ('m', 'n'), (M, N))
    check_funsor(x[n, m], ('m', 'n'), (M, N))


@pytest.mark.xfail(reason='TODO')
def test_ellipsis():
    x = funsor.TorchFunsor(('i', 'j', 'k'), torch.randn(3, 4, 5))
    check_funsor(x, ('i', 'j', 'k'), (3, 4, 5))

    assert x[...] is x

    check_funsor(x[..., 0, 0, 0], (), ())
    check_funsor(x[..., 0, 0], ('i',), (3,))
    check_funsor(x[..., 0], ('i', 'k'), (3, 4))
    check_funsor(x[0, ..., 0, 0], (), ())
    check_funsor(x[0, ..., 0], ('j',), (4,))
    check_funsor(x[0, ...], ('j', 'k'), (4, 5))
    check_funsor(x[0, 0, ..., 0], (), ())
    check_funsor(x[0, 0, ...], ('k',), (5,))
    check_funsor(x[0, 0, 0, ...], (), ())
