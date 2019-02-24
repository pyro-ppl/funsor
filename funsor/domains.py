from __future__ import absolute_import, division, print_function

from collections import namedtuple

from pyro.distributions.util import broadcast_shape
from six import integer_types

import funsor.ops as ops
from funsor.util import lazy_property


class Domain(namedtuple('Domain', ['shape', 'dtype'])):
    """
    An object representing the type and shape of a :class:`Funsor` input or
    output.
    """
    def __new__(cls, shape, dtype):
        assert isinstance(shape, tuple)
        assert isinstance(dtype, integer_types) or (isinstance(dtype, str) and dtype == 'real')
        return super(Domain, cls).__new__(cls, shape, dtype)

    def __repr__(self):
        shape = tuple(self.shape)
        if isinstance(self.dtype, integer_types):
            if not shape:
                return 'ints({})'.format(self.dtype)
            return 'ints({}, {})'.format(self.dtype, shape)
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
        result = 1
        for size in self.shape:
            result *= size
        return result


def reals(*shape):
    """
    Construct a real domain of given shape.
    """
    return Domain(shape, 'real')


def ints(size, shape=()):
    """
    Construct a bounded integer domain of given shape.
    """
    assert isinstance(size, integer_types) and size >= 0
    return Domain(shape, size)


def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    assert callable(op), op
    assert all(isinstance(arg, Domain) for arg in domains)
    if len(domains) == 1:
        return domains[0]

    lhs, rhs = domains
    if op is ops.getitem:
        dtype = lhs.dtype
        shape = lhs.shape[rhs.num_elements:]
        return Domain(shape, dtype)

    if lhs.dtype == 'real' or rhs.dtype == 'real':
        dtype = "real"
    else:
        dtype = op(lhs.dtype, rhs.dtype)

    if lhs.shape == rhs.shape:
        shape = lhs.shape
    else:
        shape = broadcast_shape(lhs.shape, rhs.shape)
    return Domain(shape, dtype)


__all__ = [
    'Domain',
    'find_domain',
    'ints',
    'reals',
]
