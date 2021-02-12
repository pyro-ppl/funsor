# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
import types
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ContextDecorator
from functools import singledispatch
from timeit import default_timer

import numpy as np

from funsor.domains import ArrayType
from funsor.instrument import debug_logged
from funsor.ops import Op, is_numeric_array
from funsor.registry import KeyedRegistry
from funsor.util import is_nn_module

from . import instrument

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_USE_TCO = int(os.environ.get("FUNSOR_USE_TCO", 0))
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


else:
    interpret = Interpreter()


def interpretation(new):
    warnings.warn(
        "'with interpretation(x)' should be replaced by 'with x'",
        DeprecationWarning,
    )
    assert isinstance(new, Interpretation)
    return new


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
@debug_logged
def reinterpret_funsor(x):
    interpret = _STACK[-1].interpret
    return interpret(type(x), *map(recursion_reinterpret, x._ast_values))


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


for t in _ground_types:

    @recursion_reinterpret.register(t)
    def recursion_reinterpret_ground(x):
        return x


@recursion_reinterpret.register(tuple)
@debug_logged
def recursion_reinterpret_tuple(x):
    return tuple(map(recursion_reinterpret, x))


@recursion_reinterpret.register(frozenset)
@debug_logged
def recursion_reinterpret_frozenset(x):
    return frozenset(map(recursion_reinterpret, x))


@recursion_reinterpret.register(dict)
@debug_logged
def recursion_reinterpret_dict(x):
    return {key: recursion_reinterpret(value) for key, value in x.items()}


@recursion_reinterpret.register(OrderedDict)
@debug_logged
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


for t in _ground_types:

    @children.register(t)
    def _children_ground(x):
        return ()


def is_atom(x):
    if isinstance(x, (tuple, frozenset)):
        return all(is_atom(c) for c in x)
    return isinstance(x, _ground_types) or is_numeric_array(x) or is_nn_module(x)


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
    interpret = _STACK[-1].interpret
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
            env[h_name] = type(h)(env[c_name] for c_name in parent_to_children[h_name])
        else:
            env[h_name] = interpret(
                type(h), *(env[c_name] for c_name in parent_to_children[h_name])
            )

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


################################################################################
# Interpretation class hierarchy.


class Interpretation(ContextDecorator, ABC):
    """
    Base class for Funsor interpretations.

    Instances may be used as context managers or decorators.
    """

    is_total = False

    def __enter__(self):
        # TODO consider requiring totality:
        # assert self.is_total
        new = self
        if not self.is_total:
            new = PrioritizedInterpretation(new, _STACK[-1])
        _STACK.append(new)
        return self

    def __exit__(self, *args):
        _STACK.pop()

    @abstractmethod
    def interpret(self, cls, *args):
        raise NotImplementedError


class SimpleInterpretation(Interpretation):
    def __init__(self, interpret):
        assert callable(interpret)
        super().__init__()
        self.interpret = interpret
        self.__name__ = interpret.__name__

    def update(self, interpret):
        assert callable(interpret)
        self.interpret = interpret
        return self

    def interpret(self, cls, *args):
        raise NotImplementedError


class DispatchedInterpretation(Interpretation):
    def __init__(self, name="dispatched"):
        super().__init__()
        self.__name__ = name
        self.registry = registry = KeyedRegistry(default=lambda *args: None)

        if instrument.DEBUG or instrument.PROFILE:
            self.register = lambda *args: lambda fn: registry.register(*args)(
                debug_logged(fn)
            )
        else:
            self.register = registry.register

        if instrument.PROFILE:
            COUNTERS = instrument.COUNTERS

            def profiled_dispatch(*args):
                name = self.__name__ + ".dispatch"
                start = default_timer()
                result = registry.dispatch(*args)
                COUNTERS["time"][name] += default_timer() - start
                COUNTERS["call"][name] += 1
                COUNTERS["interpretation"][self.__name__] += 1
                return result

            self.dispatch = profiled_dispatch
        else:
            self.dispatch = registry.dispatch

    def interpret(self, cls, *args):
        return self.dispatch(cls, *args)(*args)


class PrioritizedInterpretation(Interpretation):
    def __init__(self, *subinterpreters):
        assert len(subinterpreters) >= 1
        super().__init__()
        self.subinterpreters = subinterpreters
        self.__name__ = "/".join(s.__name__ for s in subinterpreters)
        if isinstance(self.subinterpreters[0], DispatchedInterpretation):
            self.register = self.subinterpreters[0].register
            self.dispatch = self.subinterpreters[0].dispatch

        if __debug__:
            self._volume = sum(getattr(s, "_volume", 1) for s in subinterpreters)
            assert self._volume < 10, "suspicious interpreter overflow"

    @property
    def base(self):
        return self.subinterpreters[0]

    @property
    def is_total(self):
        return any(s.is_total for s in self.subinterpreters)

    def interpret(self, cls, *args):
        for subinterpreter in self.subinterpreters:
            result = subinterpreter.interpret(cls, *args)
            if result is not None:
                return result


class StatefulInterpretationMeta(ABC):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.registry = KeyedRegistry(default=lambda *args: None)
        cls.dispatch = cls.registry.dispatch


class StatefulInterpretation(
    DispatchedInterpretation, metaclass=StatefulInterpretationMeta
):
    """
    Base class for interpreters with instance-dependent state or parameters.

    Example usage::

        class MyInterpretation(StatefulInterpretation):

            def __init__(self, my_param):
                self.my_param = my_param

        @MyInterpretation.register(...)
        def my_impl(interpreter_state, cls, *args):
            my_param = interpreter_state.my_param
            ...

        with interpretation(MyInterpretation(my_param=0.1)):
            ...
    """

    def __init__(self, name="stateful"):
        super().__init__()
        self.__name__ = name

    def interpret(self, cls, *args):
        return self.dispatch(cls, *args)(self, *args)

    if instrument.DEBUG:

        @classmethod
        def register(cls, *args):
            return lambda fn: cls.registry.register(*args)(debug_logged(fn))

    else:

        @classmethod
        def register(cls, *args):
            return cls.registry.register(*args)


################################################################################
# Concrete interpretations.


@SimpleInterpretation
def reflect(cls, *args):
    raise ValueError("Should be overwritten in terms.py")


reflect.is_total = True

normalize_base = DispatchedInterpretation("normalize")
normalize = PrioritizedInterpretation(normalize_base, reflect)

lazy_base = DispatchedInterpretation("lazy")
lazy = PrioritizedInterpretation(lazy_base, reflect)

eager_base = DispatchedInterpretation("eager")
eager = PrioritizedInterpretation(eager_base, normalize_base, reflect)

die = DispatchedInterpretation("die")
eager_or_die = PrioritizedInterpretation(eager_base, die, reflect)

sequential_base = DispatchedInterpretation("sequential")
# XXX does this work with sphinx/help()?
"""
Eagerly execute ops with known implementations; additonally execute
vectorized ops sequentially if no known vectorized implementation exists.
"""
sequential = PrioritizedInterpretation(
    sequential_base, eager_base, normalize_base, reflect
)

moment_matching_base = DispatchedInterpretation("moment_matching")
"""
A moment matching interpretation of :class:`Reduce` expressions. This falls
back to :class:`eager` in other cases.
"""
moment_matching = PrioritizedInterpretation(
    moment_matching_base, eager_base, normalize_base, reflect
)

push_interpretation(eager)  # Use eager interpretation by default.

__all__ = [
    "PatternMissingError",
    "StatefulInterpretation",
    "die",
    "eager",
    "interpret",
    "interpretation",
    "lazy",
    "moment_matching",
    "normalize",
    "pop_interpretation",
    "push_interpretation",
    "reflect",
    "reinterpret",
    "sequential",
]
