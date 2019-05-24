from __future__ import absolute_import, division, print_function

import functools
import inspect
import os
import re
import types
import uuid

import torch
from collections import OrderedDict
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
def children(x):
    raise ValueError(type(x))


def children_funsor(h):
    return h._ast_values


@children.register(tuple)
@children.register(frozenset)
def _children_tuple(h):
    return h


@children.register(str)
@children.register(int)
@children.register(float)
@children.register(type)
@children.register(types.FunctionType)
@children.register(types.BuiltinFunctionType)
@children.register(torch.Tensor)
@children.register(Domain)
@children.register(Op)
def _children_ground(x):
    return ()


def is_ground(x):
    if isinstance(x, (tuple, frozenset)) and not isinstance(x, Domain):
        return len(x) == 0 or all(is_ground(c) for c in x)
    return isinstance(x, (
        int,
        str,
        float,
        type,
        types.FunctionType,
        types.BuiltinFunctionType,
        torch.Tensor,
        Domain,
        Op
    ))


def gensym():
    return "V" + str(uuid.uuid4().hex)


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
    node_vars = {}
    env = {}
    x_name = gensym()
    stack = [(x_name, x)]
    parent_to_children = OrderedDict()
    child_to_parent = OrderedDict()
    while stack:
        h_name, h = stack.pop(0)
        node_vars[h_name] = h
        parent_to_children[h_name] = []
        for c in children(h):
            c_name = gensym()
            stack.append((c_name, c))
            parent_to_children[h_name].append(c_name)
            child_to_parent[c_name] = h_name

    children_counts = OrderedDict((k, len(v)) for k, v in parent_to_children.items())
    leaves = [h_name for h_name, count in children_counts.items() if count == 0]
    while leaves:
        h_name = leaves.pop(0)
        if h_name in child_to_parent:
            parent = child_to_parent[h_name]
            children_counts[parent] -= 1
            if children_counts[parent] == 0:
                leaves.append(parent)

        h = node_vars[h_name]
        if is_ground(h):
            env[h_name] = h
        elif isinstance(h, (tuple, frozenset)):
            env[h_name] = type(h)(
                env[c_name] for c_name in parent_to_children[h_name])
        else:
            env[h_name] = _INTERPRETATION(
                type(h), *(env[c_name] for c_name in parent_to_children[h_name]))

    return env[x_name]


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
    'children',
    'dispatched_interpretation',
    'interpret',
    'interpretation',
    'reinterpret',
    'set_interpretation',
]
