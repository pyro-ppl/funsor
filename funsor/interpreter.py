import functools
import inspect
import os
import re
import types
from collections import OrderedDict
from functools import singledispatch

import numpy
import torch
from contextlib2 import contextmanager

from funsor.domains import Domain
from funsor.ops import Op
from funsor.registry import KeyedRegistry

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEBUG = int(os.environ.get("FUNSOR_DEBUG", 0))
_STACK_SIZE = 0

_INTERPRETATION = None  # To be set later in funsor.terms
_USE_TCO = int(os.environ.get("FUNSOR_USE_TCO", 0))

_GENSYM_COUNTER = 0


def _get_name(cls):
    return getattr(cls, '__origin__', cls).__name__


if _DEBUG:
    def interpret(cls, *args):
        global _STACK_SIZE
        indent = '  ' * _STACK_SIZE
        typenames = [_get_name(cls)] + [_get_name(type(arg)) for arg in args]
        print(indent + ' '.join(typenames))

        _STACK_SIZE += 1
        try:
            result = _INTERPRETATION(cls, *args)
        finally:
            _STACK_SIZE -= 1

        if _DEBUG > 1:
            result_str = re.sub('\n', '\n          ' + indent, str(result))
        else:
            result_str = _get_name(type(result))
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
def recursion_reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.
    This interpreter uses the Python stack and is subject to the recursion limit.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    raise ValueError(type(x))


# We need to register this later in terms.py after declaring Funsor.
# reinterpret.register(Funsor)
def reinterpret_funsor(x):
    return _INTERPRETATION(type(x), *map(recursion_reinterpret, x._ast_values))


@recursion_reinterpret.register(str)
@recursion_reinterpret.register(int)
@recursion_reinterpret.register(float)
@recursion_reinterpret.register(type)
@recursion_reinterpret.register(functools.partial)
@recursion_reinterpret.register(types.FunctionType)
@recursion_reinterpret.register(types.BuiltinFunctionType)
@recursion_reinterpret.register(numpy.ndarray)
@recursion_reinterpret.register(torch.Tensor)
@recursion_reinterpret.register(torch.nn.Module)
@recursion_reinterpret.register(Domain)
@recursion_reinterpret.register(Op)
def recursion_reinterpret_ground(x):
    return x


@recursion_reinterpret.register(tuple)
def recursion_reinterpret_tuple(x):
    return tuple(map(recursion_reinterpret, x))


@recursion_reinterpret.register(frozenset)
def recursion_reinterpret_frozenset(x):
    return frozenset(map(recursion_reinterpret, x))


@recursion_reinterpret.register(dict)
def recursion_reinterpret_dict(x):
    return {key: recursion_reinterpret(value) for key, value in x.items()}


@recursion_reinterpret.register(OrderedDict)
def recursion_reinterpret_ordereddict(x):
    return OrderedDict((key, recursion_reinterpret(value)) for key, value in x.items())


@singledispatch
def children(x):
    raise ValueError(type(x))


# has to be registered in terms.py
def children_funsor(x):
    return x._ast_values


@children.register(tuple)
@children.register(frozenset)
def _children_tuple(x):
    return x


@children.register(dict)
@children.register(OrderedDict)
def _children_tuple(x):
    return x.values()


@children.register(str)
@children.register(int)
@children.register(float)
@children.register(type)
@children.register(functools.partial)
@children.register(types.FunctionType)
@children.register(types.BuiltinFunctionType)
@children.register(numpy.ndarray)
@children.register(torch.Tensor)
@children.register(torch.nn.Module)
@children.register(Domain)
@children.register(Op)
def _children_ground(x):
    return ()


def is_atom(x):
    if isinstance(x, (tuple, frozenset)) and not isinstance(x, Domain):
        return len(x) == 0 or all(is_atom(c) for c in x)
    return isinstance(x, (
        int,
        str,
        float,
        type,
        functools.partial,
        types.FunctionType,
        types.BuiltinFunctionType,
        torch.Tensor,
        torch.nn.Module,
        numpy.ndarray,
        Domain,
        Op
    ))


def gensym(x=None):
    global _GENSYM_COUNTER
    _GENSYM_COUNTER += 1
    sym = _GENSYM_COUNTER
    if x is not None:
        if isinstance(x, str):
            return x + "_" + str(sym)
        return id(x)
    return "V" + str(sym)


def stack_reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.
    This interpreter uses an explicit stack and no recursion but is much slower.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    x_name = gensym(x)
    node_vars = {x_name: x}
    node_names = {x: x_name}
    env = {}
    stack = [(x_name, x)]
    parent_to_children = OrderedDict()
    child_to_parents = OrderedDict()
    while stack:
        h_name, h = stack.pop(0)
        parent_to_children[h_name] = []
        for c in children(h):
            if c in node_names:
                c_name = node_names[c]
            else:
                c_name = gensym(c)
                node_names[c] = c_name
                node_vars[c_name] = c
                stack.append((c_name, c))
            parent_to_children.setdefault(h_name, []).append(c_name)
            child_to_parents.setdefault(c_name, []).append(h_name)

    children_counts = OrderedDict((k, len(v)) for k, v in parent_to_children.items())
    leaves = [name for name, count in children_counts.items() if count == 0]
    while leaves:
        h_name = leaves.pop(0)
        if h_name in child_to_parents:
            for parent in child_to_parents[h_name]:
                children_counts[parent] -= 1
                if children_counts[parent] == 0:
                    leaves.append(parent)

        h = node_vars[h_name]
        if is_atom(h):
            env[h_name] = h
        elif isinstance(h, (tuple, frozenset)):
            env[h_name] = type(h)(
                env[c_name] for c_name in parent_to_children[h_name])
        else:
            env[h_name] = _INTERPRETATION(
                type(h), *(env[c_name] for c_name in parent_to_children[h_name]))

    return env[x_name]


def reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    if _USE_TCO:
        return stack_reinterpret(x)
    else:
        return recursion_reinterpret(x)


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
