from __future__ import absolute_import, division, print_function

import functools
import inspect
import os
import re
import types
from collections import OrderedDict

import torch
from contextlib2 import contextmanager

from funsor.domains import Domain
from funsor.ops import Op
from funsor.registry import KeyedRegistry
from funsor.six import singledispatch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEBUG = int(os.environ.get("FUNSOR_DEBUG", 0))
_STACK_SIZE = 0

_INTERPRETATION = None  # To be set later in funsor.terms


if _DEBUG:
    def interpret(cls, *args):
        global _STACK_SIZE
        indent = '  ' * _STACK_SIZE
        typenames = [cls.__name__] + [type(arg).__name__ for arg in args]
        print(indent + ' '.join(typenames))

        _STACK_SIZE += 1
        try:
            result = _INTERPRETATION(cls, *args)
        finally:
            _STACK_SIZE -= 1

        if _DEBUG > 1:
            result_str = re.sub('\n', '\n          ' + indent, str(result))
        else:
            result_str = type(result).__name__
        print(indent + '-> ' + result_str)
        return result
else:
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
    Overloaded reinterpretation of a deferred expression.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

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
@reinterpret.register(Op)
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


if _DEBUG:
    class DebugLogged(object):
        def __init__(self, fn):
            self.fn = fn
            while isinstance(fn, functools.partial):
                fn = fn.func
            path = inspect.getabsfile(fn)
            lineno = inspect.getsourcelines(fn)[1]
            self._message = "{} file://{} {}".format(fn.__name__, path, lineno)

        def __call__(self, *args, **kwargs):
            print('  ' * _STACK_SIZE + self._message)
            return self.fn(*args, **kwargs)

    def debug_logged(fn):
        if isinstance(fn, DebugLogged):
            return fn
        return DebugLogged(fn)
else:
    def debug_logged(fn):
        return fn


def dispatched_interpretation(fn):
    """
    Decorator to create a dispatched interpretation function.
    """
    registry = KeyedRegistry(default=lambda *args: None)
    if _DEBUG:
        fn.register = lambda *args: lambda fn: registry.register(*args)(debug_logged(fn))
    else:
        fn.register = registry.register
    fn.dispatch = registry.__call__
    return fn


__all__ = [
    'dispatched_interpretation',
    'interpret',
    'interpretation',
    'reinterpret',
    'set_interpretation',
]
