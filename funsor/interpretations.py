# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import ContextDecorator
from timeit import default_timer

from . import instrument
from .interpreter import get_interpretation, pop_interpretation, push_interpretation
from .registry import KeyedRegistry


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
            new = PrioritizedInterpretation(new, get_interpretation())
        push_interpretation(new)
        return self

    def __exit__(self, *args):
        pop_interpretation()

    @abstractmethod
    def interpret(self, cls, *args):
        raise NotImplementedError


class CallableInterpretation(Interpretation):
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
                instrument.debug_logged(fn)
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

        with MyInterpretation(my_param=0.1):
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
            return lambda fn: cls.registry.register(*args)(instrument.debug_logged(fn))

    else:

        @classmethod
        def register(cls, *args):
            return cls.registry.register(*args)


################################################################################
# Concrete interpretations.


@CallableInterpretation
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
    "Interpretation",
    "DispatchedInterpretation",
    "CallableInterpretation",
    "StatefulInterpretation",
    "die",
    "eager",
    "lazy",
    "moment_matching",
    "normalize",
    "reflect",
    "sequential",
]
