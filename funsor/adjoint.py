from __future__ import absolute_import, division, print_function

import torch

import funsor.ops as ops
from funsor.registry import KeyedRegistry
from funsor.terms import Funsor
from funsor.torch import Tensor


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


backward = KeyedRegistry()


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


################################################################################
# Backward Implementations
################################################################################

@backward.register(ops.min, Tensor)
def _min_tensor(term, dims):
    if len(dims) != 1:
        raise NotImplementedError('TODO')
    dim = next(iter(dims))

    pos = term.dims.index(dim)
    value = term.data.min(pos)[0]
    dims = term.dims[:pos] + term.dims[1 + pos:]
    return {dim: Tensor(dims, value)}


@backward.register(ops.max, Tensor)
def _max_tensor(term, dims):
    if len(dims) != 1:
        raise NotImplementedError('TODO')
    dim = next(iter(dims))

    pos = term.dims.index(dim)
    value = term.data.max(pos)[0]
    dims = term.dims[:pos] + term.dims[1 + pos:]
    return {dim: Tensor(dims, value)}


@backward.register(ops.sample, Tensor)
def _sample_tensor(term, dims):
    if len(dims) > 1:
        raise NotImplementedError('TODO')
    dim = next(iter(dims))

    pos = term.dims.index(dim)
    probs = (term.data - term.data.max(pos, keepdim=True)[0]).exp()
    probs = probs.transpose(pos, -1)
    value = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
    value = value.reshape(probs.shape[:-1] + (1,))
    value = value.transpose(pos, -1).squeeze(pos)
    dims = term.dims[:pos] + term.dims[1 + pos:]
    return {dim: Tensor(dims, value)}


__all__ = [
    'adjoints',
    'argreduce',
    'backward',
]
