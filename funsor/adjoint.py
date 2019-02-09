from __future__ import absolute_import, division, print_function

import torch

import funsor.ops as ops
from funsor.terms import Funsor, Tensor


class adjoints(object):
    """
    Handler enabling compute graphs to be computed so that adjoint quantities
    can later be computed.
    """
    def __enter__(self):
        raise NotImplementedError(
            'TODO(eb8680) Install effect handler to record compute graph.')

    def __exit__(self, *args):
        raise NotImplementedError(
            'TODO(eb8680) Unistall effect handler to record compute graph.')


def backward(op, forward_result, dims):
    """
    Compute backward operation, after having computed a ``forward_result``
    ``with adjoints()``.

    :param callable op: A reduction operation.
    :param set dims: An optional dim or set of dims to reduce.
    :return: A dict mapping a subset of input dims to funsors possibly
        depending on remaining dims.
    :rtype: dict
    """
    if isinstance(dims, str):
        dims = (dims,)
    assert set(dims) <= set(forward_result.dims)
    if not dims:
        return {}
    raise NotImplementedError('TODO')


def argreduce(op, term, dims):
    """
    Reduce along a subset of dimensions,
    keeping track of the values of those dimensions.

    :param callable op: A reduction operation.
    :param set dims: An optional dim or set of dims to reduce.
    :return: A tuple ``(args, remaining)`` where ``args`` is a
        dict mapping a subset of input dims to funsors possibly depending
        on remaining dims, and ``remaining`` is a funsor depending on
        remaing dims.
    :rtype: tuple
    """
    assert callable(op)
    assert isinstance(term, Funsor)
    assert all(isinstance(dim, str) for dim in dims)
    with adjoints():
        remaining = term.reduce(op, dims)
    args = backward(op, remaining, dims)
    return args, remaining


# TODO Move this into a Backward definition.
def argreduce_one(op, term, dim):
    assert dim in term.dims

    if op in (ops.min, ops.max):
        pos = term.dims.index(dim)
        value = getattr(term.data, op.__name__)(pos)[0]
        dims = term.dims[:pos] + term.dims[1 + pos:]
        return Tensor(dims, value)

    if op is ops.sample:
        pos = term.dims.index(dim)
        probs = (term.data - term.data.max(pos, keepdim=True)[0]).exp()
        probs = probs.transpose(pos, -1)
        value = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
        value = value.reshape(probs.shape[:-1] + (1,))
        value = value.transpose(pos, -1).squeeze(pos)
        dims = term.dims[:pos] + term.dims[1 + pos:]
        return Tensor(dims, value)

    raise NotImplementedError


__all__ = [
    'adjoints',
    'backward',
]
