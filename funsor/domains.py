# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import operator
from collections import namedtuple
from functools import reduce

import funsor.ops as ops
from funsor.util import broadcast_shape, get_tracing_state, lazy_property, quote


class Domain(namedtuple('Domain', ['shape', 'dtype'])):
    """
    An object representing the type and shape of a :class:`Funsor` input or
    output.
    """
    def __new__(cls, shape, dtype):
        assert isinstance(shape, tuple)
        if get_tracing_state():
            shape = tuple(map(int, shape))
        assert all(isinstance(size, int) for size in shape), shape
        if isinstance(dtype, int):
            assert not shape
        elif isinstance(dtype, str):
            assert dtype == 'real'
        else:
            raise ValueError(repr(dtype))
        return super(Domain, cls).__new__(cls, shape, dtype)

    def __repr__(self):
        shape = tuple(self.shape)
        if isinstance(self.dtype, int):
            if not shape:
                return 'bint({})'.format(self.dtype)
            return 'bint({}, {})'.format(self.dtype, shape)
        if not shape:
            return 'reals()'
        return 'reals{}'.format(shape)

    def __iter__(self):
        if isinstance(self.dtype, int) and not self.shape:
            from funsor.terms import Number
            return (Number(i, self.dtype) for i in range(self.dtype))
        raise NotImplementedError

    @lazy_property
    def num_elements(self):
        return reduce(operator.mul, self.shape, 1)

    @property
    def size(self):
        assert isinstance(self.dtype, int)
        return self.dtype


@quote.register(Domain)
def _(arg, indent, out):
    out.append((indent, repr(arg)))


def reals(*shape):
    """
    Construct a real domain of given shape.
    """
    return Domain(shape, 'real')


def bint(size):
    """
    Construct a bounded integer domain of scalar shape.
    """
    if get_tracing_state():
        size = int(size)
    assert isinstance(size, int) and size >= 0
    return Domain((), size)


def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    assert callable(op), op
    assert all(isinstance(arg, Domain) for arg in domains)
    if len(domains) == 1:
        dtype = domains[0].dtype
        shape = domains[0].shape
        if op is ops.log or op is ops.exp:
            dtype = 'real'
        elif isinstance(op, ops.ReshapeOp):
            shape = op.shape
        elif isinstance(op, ops.AssociativeOp):
            shape = ()
        return Domain(shape, dtype)

    lhs, rhs = domains
    if isinstance(op, ops.GetitemOp):
        dtype = lhs.dtype
        shape = lhs.shape[:op.offset] + lhs.shape[1 + op.offset:]
        return Domain(shape, dtype)
    elif op == ops.matmul:
        assert lhs.shape and rhs.shape
        if len(rhs.shape) == 1:
            assert lhs.shape[-1] == rhs.shape[-1]
            shape = lhs.shape[:-1]
        elif len(lhs.shape) == 1:
            assert lhs.shape[-1] == rhs.shape[-2]
            shape = rhs.shape[:-2] + rhs.shape[-1:]
        else:
            assert lhs.shape[-1] == rhs.shape[-2]
            shape = broadcast_shape(lhs.shape[:-1], rhs.shape[:-2] + (1,)) + rhs.shape[-1:]
        return Domain(shape, 'real')

    if lhs.dtype == 'real' or rhs.dtype == 'real':
        dtype = 'real'
    elif op in (ops.add, ops.mul, ops.pow, ops.max, ops.min):
        dtype = op(lhs.dtype - 1, rhs.dtype - 1) + 1
    elif op in (ops.and_, ops.or_, ops.xor):
        dtype = 2
    elif lhs.dtype == rhs.dtype:
        dtype = lhs.dtype
    else:
        raise NotImplementedError('TODO')

    if lhs.shape == rhs.shape:
        shape = lhs.shape
    else:
        shape = broadcast_shape(lhs.shape, rhs.shape)
    return Domain(shape, dtype)


__all__ = [
    'Domain',
    'find_domain',
    'bint',
    'reals',
]
