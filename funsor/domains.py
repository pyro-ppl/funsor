from __future__ import absolute_import, division, print_function

from collections import namedtuple


class Domain(namedtuple('Domain', ['shape', 'dtype'])):
    """
    An object representing the type and shape of a :class:`Funsor` input or
    output.
    """
    def __new__(cls, shape, dtype):
        assert isinstance(shape, tuple)
        assert isinstance(dtype, int) or (isinstance(dtype, str) and dtype == 'real')
        return super(Domain, cls).__new__(cls, shape, dtype)

    def __repr__(self):
        if isinstance(self.dtype, int):
            if not self.shape:
                return 'ints({})'.format(self.dtype)
            return 'ints({}, {})'.format(self.dtype, self.shape)
        if not self.shape:
            return 'reals()'
        return 'reals{}'.format(self.shape)

    def __iter__(self):
        if isinstance(self.dtype, int) and not self.shape:
            return range(self.dtype)
        raise NotImplementedError


def reals(*shape):
    """
    Construct a real domain of given shape.
    """
    return Domain(shape, 'real')


def ints(size, shape=()):
    """
    Construct a bounded integer domain of given shape.
    """
    assert isinstance(size, int) and size >= 0
    return Domain(shape, size)


def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    assert callable(op), op
    assert all(isinstance(arg, Domain) for arg in domains)
    return domains[0]  # FIXME broadcast here


__all__ = [
    'Domain',
    'find_domain',
    'ints',
    'reals',
]
