# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable
from contextlib import contextmanager

import funsor.interpreter as interpreter


@contextmanager
def memoize(cache=None):
    """
    Exploit cons-hashing to do implicit common subexpression elimination
    """
    if cache is None:
        cache = {}

    @interpreter.interpretation(interpreter._INTERPRETATION)  # use base
    def memoize_interpretation(cls, *args):
        key = (cls,) + tuple(id(arg) if ("DeviceArray" in type(arg).__name__) or not isinstance(arg, Hashable)
                             else arg for arg in args)
        if key not in cache:
            cache[key] = cls(*args)
        return cache[key]

    with interpreter.interpretation(memoize_interpretation):
        yield cache
