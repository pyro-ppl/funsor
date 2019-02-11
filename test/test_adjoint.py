from __future__ import absolute_import, division, print_function

import pytest
import torch

import funsor
import funsor.ops as ops
from funsor.adjoint import backward


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


ARGREDUCE_OPS = [ops.min, ops.max, ops.sample]


@pytest.mark.xfail(reason='backward has incorrect signature')
@pytest.mark.parametrize('dims,dim', [
    (dims, dim)
    for dims in [('a',), ('a', 'b'), ('b', 'a', 'c')]
    for dim in dims
])
@pytest.mark.parametrize('op', ARGREDUCE_OPS)
def test_backward_one(op, dims, dim):
    sizes = {'a': 3, 'b': 4, 'c': 5}
    shape = tuple(sizes[d] for d in dims)
    data = torch.rand(shape) + 0.5
    x = funsor.Tensor(dims, data)
    actual = backward(op, x, frozenset([dim]))
    assert isinstance(actual, dict)
    assert set(actual.keys()) == {dim}

    pos = dims.index(dim)
    if op in (ops.min, ops.max):
        data = getattr(data, op.__name__)(pos)[0]
        shape = data.shape
    else:
        shape = data.shape[:pos] + data.shape[1+pos:]
        data = None
    dims = tuple(d for d in dims if d != dim)
    check_funsor(actual, dims, shape, data)
