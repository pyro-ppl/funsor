r"""
Funsor interpretations
----------------------

Funsor provides three basic interpretations.

- ``reflect`` is completely lazy, even with respect to substitution.
- ``lazy`` substitutes eagerly but performs ops lazily.
- ``eager`` does everything eagerly.

"""

from __future__ import absolute_import, division, print_function

import types
from collections import OrderedDict

import torch
from contextlib2 import contextmanager

from funsor.domains import Domain
from funsor.registry import KeyedRegistry
from funsor.six import singledispatch


def reflect(cls, *args):
    return cls(*args)


_lazy = KeyedRegistry(default=lambda *args: None)
_eager = KeyedRegistry(default=lambda *args: None)


def lazy(cls, *args):
    result = _lazy(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


def eager(cls, *args):
    result = _eager(cls, *args)
    if result is None:
        result = reflect(cls, *args)
    return result


lazy.register = _lazy.register
eager.register = _eager.register

_INTERPRETATION = eager  # Use eager interpretation by default.


def interpret(cls, *args):
    return _INTERPRETATION(cls, *args)


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


@singledispatch
def reinterpret(x):
    r"""
    Overloaded reinterpretation deferred expression.
    Default semantics: do nothing (reflect)

    This handles a limited class of expressions, raising
    ``NotImplementedError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor`s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    raise ValueError(type(x))


# We need to register this later in terms.py after declaring Funsor.
# reinterpret.register(Funsor)
def reinterpret_funsor(x):
    return _INTERPRETATION(type(x), *map(reinterpret, x._ast_values))


@reinterpret.register(str)
@reinterpret.register(int)
@reinterpret.register(float)
@reinterpret.register(type)
@reinterpret.register(types.FunctionType)
@reinterpret.register(types.BuiltinFunctionType)
@reinterpret.register(torch.Tensor)
@reinterpret.register(Domain)
def _reinterpret_ground(x):
    return x


@reinterpret.register(tuple)
def _reinterpret_tuple(x):
    return tuple(map(reinterpret, x))


@reinterpret.register(frozenset)
def _reinterpret_frozenset(x):
    return frozenset(map(reinterpret, x))


@reinterpret.register(dict)
def _reinterpret_dict(x):
    return {key: reinterpret(value) for key, value in x.items()}


@reinterpret.register(OrderedDict)
def _reinterpret_ordereddict(x):
    return OrderedDict((key, reinterpret(value)) for key, value in x.items())


__all__ = [
    'eager',
    'interpret',
    'interpretation',
    'lazy',
    'reflect',
    'reinterpret',
    'set_interpretation',
]
