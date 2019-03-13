from collections import OrderedDict

import pytest

import funsor
from funsor import bint, reals, Variable, Number
from funsor.numpy import Array
import numpy as np

from funsor.testing import check_funsor, assert_equiv, random_array


@pytest.mark.parametrize('shape', [(), (4,), (3, 2)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64, np.uint8])
def test_to_funsor(shape, dtype):
    t = np.random.normal(size=shape).astype(dtype)
    f = funsor.to_funsor(t)
    assert isinstance(f, Array)


def test_cons_hash():
    x = np.random.randn(3, 3)
    assert Array(x) is Array(x)


def test_indexing():
    data = np.random.normal(size=(4, 5))
    inputs = OrderedDict([('i', bint(4)),
                          ('j', bint(5))])
    x = Array(data, inputs)
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


def test_advanced_indexing_shape():
    I, J, M, N = 4, 4, 2, 3
    x = Array(np.random.normal(size=(I, J)), OrderedDict([
        ('i', bint(I)),
        ('j', bint(J)),
    ]))
    m = Array(np.array([2, 3]), OrderedDict([('m', bint(M))]), I)
    n = Array(np.array([0, 1, 1]), OrderedDict([('n', bint(N))]), J)
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


@pytest.mark.parametrize('output_shape', [(), (7,), (3, 2)])
def test_advanced_indexing_array(output_shape):
    #      u   v
    #     / \ / \
    #    i   j   k
    #     \  |  /
    #      \ | /
    #        x
    output = reals(*output_shape)
    x = random_array(OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]), output)
    i = random_array(OrderedDict([
        ('u', bint(5)),
    ]), bint(2))
    j = random_array(OrderedDict([
        ('v', bint(6)),
        ('u', bint(5)),
    ]), bint(3))
    k = random_array(OrderedDict([
        ('v', bint(6)),
    ]), bint(4))

    expected_data = np.empty((5, 6) + output_shape)
    for u in range(5):
        for v in range(6):
            expected_data[u, v] = x.data[i.data[u], j.data[v, u], k.data[v]]
    expected = Array(expected_data, OrderedDict([
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
def test_advanced_indexing_lazy(output_shape):
    x = Array(np.random.normal(size=(2, 3, 4) + output_shape), OrderedDict([
        ('i', bint(2)),
        ('j', bint(3)),
        ('k', bint(4)),
    ]))
    u = Variable('u', bint(2))
    v = Variable('v', bint(3))
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
