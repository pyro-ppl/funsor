from collections import Hashable
from contextlib2 import contextmanager

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
        key = (cls,) + tuple(id(arg) if not isinstance(arg, Hashable) else arg for arg in args)
        if key not in cache:
            cache[key] = cls(*args)
        return cache[key]

    try:
        with interpreter.interpretation(memoize_interpretation):
            yield cache
    finally:
        assert cache is not None  # XXX do anything more here?
