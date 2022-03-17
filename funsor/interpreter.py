# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
import types
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import singledispatch

import numpy as np

from funsor.domains import ArrayType
from funsor.ops import Op, is_numeric_array
from funsor.util import is_jax_compiled_function, is_nn_module

from . import instrument

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_TCO = int(os.environ.get("FUNSOR_USE_TCO", 0))
_TYPECHECK = int(os.environ.get("FUNSOR_TYPECHECK", 0))
_STACK = []  # To be populated later in funsor.terms
_GENSYM_COUNTER = 0


class PatternMissingError(NotImplementedError):
    def __str__(self):
        return f"{super().__str__()}\nThis is most likely due to a missing pattern."


def get_interpretation():
    return _STACK[-1]


def push_interpretation(new):
    assert callable(new)
    _STACK.append(new)


def pop_interpretation():
    return _STACK.pop()


class Interpreter:
    @property
    def __call__(self):
        return _STACK[-1].interpret


if instrument.DEBUG:

    def _classname(cls):
        return getattr(cls, "classname", cls.__name__)

    def interpret(cls, *args):
        indent = instrument.get_indent()
        if instrument.DEBUG > 1:
            typenames = [_classname(cls)] + [_classname(type(arg)) for arg in args]
        else:
            typenames = [cls.__name__] + [type(arg).__name__ for arg in args]
        print(indent + " ".join(typenames))

        instrument.STACK_SIZE += 1
        try:
            result = _STACK[-1].interpret(cls, *args)
        finally:
            instrument.STACK_SIZE -= 1

        if instrument.DEBUG > 1:
            result_str = re.sub("\n", "\n          " + indent, str(result))
        else:
            result_str = type(result).__name__
        print(indent + "-> " + result_str)
        return result

elif _TYPECHECK:

    def interpret(cls, *args):
        reflect = _STACK[0]
        interpretation = _STACK[-1]
        if interpretation is not reflect:
            reflect.interpret(cls, *args)  # for checking only
        return interpretation.interpret(cls, *args)

else:
    interpret = Interpreter()


def interpretation(new):
    warnings.warn(
        "'with interpretation(x)' should be replaced by 'with x'", DeprecationWarning
    )
    return new


_ground_types = (
    str,
    int,
    float,
    type,
    functools.partial,
    types.FunctionType,
    types.BuiltinFunctionType,
    ArrayType,
    Op,
    np.generic,
    np.ndarray,
    np.ufunc,
)


@singledispatch
def children(x):
    if is_atom(x):
        return ()
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


def is_atom(x):
    if isinstance(x, (tuple, frozenset)):
        return all(is_atom(c) for c in x)
    return (
        isinstance(x, _ground_types)
        or is_numeric_array(x)
        or is_nn_module(x)
        or is_jax_compiled_function(x)
    )


def gensym(x=None):
    global _GENSYM_COUNTER
    _GENSYM_COUNTER += 1
    sym = _GENSYM_COUNTER
    if x is not None:
        if isinstance(x, str):
            return x + "_" + str(sym)
        return id(x)
    return "V" + str(sym)


def anf(x, stop=is_atom):
    stack = deque([x])
    child_to_parents = defaultdict(list)
    children_counts = defaultdict(int)
    leaves = deque()
    while stack:
        h = stack.popleft()
        for c in children(h):
            if stop(c):
                continue
            if c not in child_to_parents:
                stack.append(c)
            child_to_parents[c].append(h)
            children_counts[h] += 1
        if children_counts[h] == 0:
            leaves.append(h)

    env = OrderedDict(((x, x),))
    while leaves:
        h = leaves.popleft()
        for parent in child_to_parents[h]:
            children_counts[parent] -= 1
            if children_counts[parent] == 0:
                leaves.append(parent)
        env[h] = h

    env.move_to_end(x)
    return env


def stack_reinterpret(x):
    r"""
    Overloaded reinterpretation of a deferred expression.
    This interpreter does not use the Python stack and
    therefore works with arbitrarily large expressions.

    This handles a limited class of expressions, raising
    ``ValueError`` in unhandled cases.

    :param x: An input, typically involving deferred
        :class:`~funsor.terms.Funsor` s.
    :type x: A funsor or data structure holding funsors.
    :return: A reinterpreted version of the input.
    :raises: ValueError
    """
    if is_atom(x):
        return x

    interpret = _STACK[-1].interpret
    env = anf(x)
    for key, value in env.items():
        if isinstance(value, (tuple, frozenset)):  # TODO absorb this into interpret
            env[key] = type(value)(c if is_atom(c) else env[c] for c in children(value))
        else:
            env[key] = interpret(
                type(value), *(c if is_atom(c) else env[c] for c in children(value))
            )
    return env[x]


@instrument.debug_logged
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
    if is_atom(x):
        return x
    elif isinstance(x, (tuple, frozenset)):
        return type(x)(map(recursion_reinterpret, children(x)))
    else:
        return _STACK[-1].interpret(type(x), *map(recursion_reinterpret, children(x)))


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


__all__ = [
    "PatternMissingError",
    "interpret",
    "interpretation",
    "pop_interpretation",
    "push_interpretation",
    "reinterpret",
]
