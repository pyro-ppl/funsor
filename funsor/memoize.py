# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable

import funsor.interpreter as interpreter
from funsor.util import get_backend


def memoize(cache=None):
    base_interpretation = interpreter.get_interpretation()
    return MemoizeInterpretation(base_interpretation, cache)


class MemoizeInterpretation(interpreter.Interpretation):
    """
    Exploit cons-hashing to do implicit common subexpression elimination
    """

    def __init__(self, base_interpretation, cache=None):
        self.base_interpretation = base_interpretation
        self.cache = {} if cache is None else cache

    @property
    def is_total(self):
        return self.base_interpretation.is_total

    def interpret(self, cls, *args):
        # JAX DeviceArray has .__hash__ method but raise the unhashable error there.
        if get_backend() == "jax":
            import jax

            key = tuple(
                id(arg)
                if isinstance(arg, jax.interpreters.xla.DeviceArray)
                or not isinstance(arg, Hashable)
                else arg
                for arg in args
            )
        else:
            key = tuple(
                id(arg) if not isinstance(arg, Hashable) else arg for arg in args
            )
        value = self.cache.get(key)
        if value is None:
            self.cache[key] = value = self.base_interpretation(cls, *args)
        return value
