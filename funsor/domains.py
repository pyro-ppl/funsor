from __future__ import absolute_import, division, print_function

import operator
from collections import namedtuple

from pyro.distributions.util import broadcast_shape
from six import integer_types
from six.moves import reduce

import funsor.ops as ops
from funsor.util import lazy_property


class Domain(namedtuple('Domain', ['shape', 'dtype'])):
    """
    An object representing the type and shape of a :class:`Funsor` input or
    output.
    """
    def __new__(cls, shape, dtype):
        assert isinstance(shape, tuple)
        assert all(isinstance(size, integer_types) for size in shape)
        if isinstance(dtype, integer_types):
            assert not shape
        elif isinstance(dtype, str):
            assert dtype == 'real'
        else:
            raise ValueError(repr(dtype))
        return super(Domain, cls).__new__(cls, shape, dtype)

    def __repr__(self):
        shape = tuple(self.shape)
        if isinstance(self.dtype, integer_types):
            if not shape:
                return 'bint({})'.format(self.dtype)
            return 'bint({}, {})'.format(self.dtype, shape)
        if not shape:
            return 'reals()'
        return 'reals{}'.format(shape)

    def __iter__(self):
        if isinstance(self.dtype, integer_types) and not self.shape:
            from funsor.terms import Number
            return (Number(i, self.dtype) for i in range(self.dtype))
        raise NotImplementedError

    @lazy_property
    def num_elements(self):
        return reduce(operator.mul, self.shape, 1)


def reals(*shape):
    """
    Construct a real domain of given shape.
    """
    return Domain(shape, 'real')


def bint(size):
    """
    Construct a bounded integer domain of scalar shape.
    """
    assert isinstance(size, integer_types) and size >= 0
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
        return Domain(shape, dtype)

    lhs, rhs = domains
    if isinstance(op, ops.GetitemOp):
        dtype = lhs.dtype
        shape = lhs.shape[:op.offset] + lhs.shape[1 + op.offset:]
        return Domain(shape, dtype)

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
