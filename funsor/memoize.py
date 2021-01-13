# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable
from contextlib import contextmanager

import funsor.interpreter as interpreter
from funsor.util import get_backend


@contextmanager
def memoize(cache=None):
    """
    Exploit cons-hashing to do implicit common subexpression elimination
    """
    if cache is None:
        cache = {}

    @interpreter.interpretation(interpreter._INTERPRETATION)  # use base
    def memoize_interpretation(cls, *args):
        # JAX DeviceArray has .__hash__ method but raise the unhashable error there.
        if get_backend() == "jax":
            import jax

            key = tuple(id(arg)
                        if isinstance(arg, jax.interpreters.xla.DeviceArray)
                        or not isinstance(arg, Hashable)
                        else arg for arg in args)
        else:
            key = tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)
        if key not in cache:
            cache[key] = cls(*args)
        return cache[key]

    with interpreter.interpretation(memoize_interpretation):
        yield cache
