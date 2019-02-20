from __future__ import absolute_import, division, print_function

from collections import namedtuple


class Domain(namedtuple('Domain', ['shape', 'dtype'])):
    """
    An object representing the type and shape of a :class:`Funsor` input or
    output.
    """
    def __init__(self, shape, dtype):
        assert isinstance(shape, tuple)
        assert isinstance(dtype, int) or (isinstance(dtype, str) and dtype == 'real')
        super(Domain, self).__init__(shape, dtype)

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


def find_domain(op, *args):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``args``.
    :param callable op: An operation.
    :param Domain \*args: One or more input domains.
    """
    assert callable(op)
    assert all(isinstance(arg, Domain) for arg in args)
    return args[0]  # FIXME broadcast here


__all__ = [
    'Domain',
    'find_domain',
    'ints',
    'reals',
]
