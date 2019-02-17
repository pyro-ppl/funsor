from __future__ import absolute_import, division, print_function

from funsor.handlers import OpRegistry


class Eager(OpRegistry):
    pass


class Simplify(OpRegistry):
    pass


__all__ = [
    'Eager',
    'Simplify',
]
