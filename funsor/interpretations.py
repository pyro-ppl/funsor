r"""
Funsor interpretations
----------------------

Funsor provides three basic interpretations.

- ``reflect`` is completely lazy, even with respect to substitution.
- ``lazy`` substitutes eagerly but performs ops lazily.
- ``eager`` does everything eagerly.

"""

from __future__ import absolute_import, division, print_function

from contextlib2 import contextmanager

from funsor.registry import KeyedRegistry


def reflect(cls, *args):
    return None  # defer to default implementation


lazy = KeyedRegistry()
eager = KeyedRegistry()

_INTERPRETATION = eager  # Use eager interpretation by default.


def get_interpretation():
    return _INTERPRETATION


def set_interpretation(new):
    assert callable(new)
    global _INTERPRETATION
    _INTERPRETATION = new


@contextmanager
def interpretation(new):
    assert callable(new)
    global _INTERPRETATION
    old = _INTERPRETATION
    try:
        _INTERPRETATION = new
        yield
    finally:
        _INTERPRETATION = old


__all__ = [
    'eager',
    'lazy',
    'reflect',
    'set_interpretation',
    'get_interpretation',
    'interpretation',
]
