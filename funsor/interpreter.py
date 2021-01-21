# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import atexit
import functools
import inspect
import os
import re
import types
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import contextmanager
from functools import singledispatch
from timeit import default_timer

import numpy as np

from funsor.domains import ArrayType
from funsor.ops import Op, is_numeric_array
from funsor.registry import KeyedRegistry
from funsor.util import is_nn_module

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEBUG = int(os.environ.get("FUNSOR_DEBUG", 0))
_PROFILE = int(os.environ.get("FUNSOR_PROFILE", 0))
_STACK_SIZE = 0

_INTERPRETATION = None  # To be set later in funsor.terms
_USE_TCO = int(os.environ.get("FUNSOR_USE_TCO", 0))

_GENSYM_COUNTER = 0


def _indent():
    result = u'    \u2502' * (_STACK_SIZE // 4 + 3)
    return result[:_STACK_SIZE]


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
            global _STACK_SIZE
            print(_indent() + self._message)
            _STACK_SIZE += 1
            try:
                return self.fn(*args, **kwargs)
            finally:
                _STACK_SIZE -= 1

        @property
        def register(self):
            return self.fn.register

    def debug_logged(fn):
        if isinstance(fn, DebugLogged):
            return fn
        return DebugLogged(fn)
elif _PROFILE:

    class ProfileLogged(object):
        def __init__(self, fn):
            self.fn = fn
            while isinstance(fn, functools.partial):
                fn = fn.func
            path = inspect.getabsfile(fn).split("/funsor/")[-1]
            lineno = inspect.getsourcelines(fn)[1]
            self._message = "{} {} {}".format(fn.__name__, path, lineno)

        def __call__(self, *args, **kwargs):
            start = default_timer()
            result = self.fn(*args, **kwargs)
            COUNTERS["time"][self._message] += default_timer() - start
            COUNTERS["call"][self._message] += 1
            return result

        @property
        def register(self):
            return self.fn.register

    def debug_logged(fn):
        if isinstance(fn, ProfileLogged):
            return fn
        return ProfileLogged(fn)
else:
    def debug_logged(fn):
        return fn


COUNTERS = defaultdict(Counter)
if _PROFILE:
    COUNTERS["time"]["total"] -= default_timer()

    @atexit.register
    def print_counters():
        COUNTERS["time"]["total"] += default_timer()
        for name, counter in sorted(COUNTERS.items()):
            if "total" not in counter and len(counter) > 1:
                counter["total"] = sum(counter.values())
            print("-" * 80)
            print(f"     count {name}")
            for key, value in counter.most_common(_PROFILE):
                if isinstance(value, float):
                    print(f"{value: >10f} {key}")
                else:
                    print(f"{value: >10} {key}")
        print("-" * 80)


def _classname(cls):
    return getattr(cls, "classname", cls.__name__)


class Interpreter:
    @property
    def __call__(self):
        return _INTERPRETATION


def debug_interpret(cls, *args):
    global _STACK_SIZE
    indent = _indent()
    if _DEBUG > 1:
        typenames = [_classname(cls)] + [_classname(type(arg)) for arg in args]
    else:
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


interpret = debug_interpret if _DEBUG else Interpreter()


def set_interpretation(new):
    assert callable(new)
    global _INTERPRETATION
    _INTERPRETATION = new


@contextmanager
def interpretation(new):
    assert callable(new)
    global _INTERPRETATION
    old = _INTERPRETATION
    new = InterpreterStack(new, old)
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
@debug_logged
def reinterpret_funsor(x):
    return _INTERPRETATION(type(x), *map(recursion_reinterpret, x._ast_values))


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
        return len(x) == 0 or all(is_atom(c) for c in x)
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


class InterpreterStack(namedtuple("InterpreterStack", ["default", "fallback"])):
    def __call__(self, cls, *args):
        for interpreter in self:
            result = interpreter(cls, *args)
            if result is not None:
                return result


def dispatched_interpretation(fn):
    """
    Decorator to create a dispatched interpretation function.
    """
    registry = KeyedRegistry(default=lambda *args: None)

    if _DEBUG or _PROFILE:
        fn.register = lambda *args: lambda fn: registry.register(*args)(debug_logged(fn))
    else:
        fn.register = registry.register

    if _PROFILE:
        def profiled_dispatch(*args):
            name = fn.__name__ + ".dispatch"
            start = default_timer()
            result = registry.dispatch(*args)
            COUNTERS["time"][name] += default_timer() - start
            COUNTERS["call"][name] += 1
            COUNTERS["interpretation"][fn.__name__] += 1
            return result
        fn.dispatch = profiled_dispatch
    else:
        fn.dispatch = registry.dispatch

    return fn


class StatefulInterpretationMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.registry = KeyedRegistry(default=lambda *args: None)
        cls.dispatch = cls.registry.dispatch


class StatefulInterpretation(metaclass=StatefulInterpretationMeta):
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

    def __call__(self, cls, *args):
        return self.dispatch(cls, *args)(self, *args)

    if _DEBUG:
        @classmethod
        def register(cls, *args):
            return lambda fn: cls.registry.register(*args)(debug_logged(fn))
    else:
        @classmethod
        def register(cls, *args):
            return cls.registry.register(*args)


class PatternMissingError(NotImplementedError):
    def __str__(self):
        return "{}\nThis is most likely due to a missing pattern.".format(super().__str__())


__all__ = [
    'PatternMissingError',
    'StatefulInterpretation',
    'dispatched_interpretation',
    'interpret',
    'interpretation',
    'reinterpret',
    'set_interpretation',
]
