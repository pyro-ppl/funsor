from contextlib2 import contextmanager

import funsor.interpreter as interpreter
from funsor.util import force_hashable


@contextmanager
def memoize(cache=None):
    """
    Exploit cons-hashing to do implicit common subexpression elimination
    """
    if cache is None:
        cache = {}

    @interpreter.interpretation(interpreter._INTERPRETATION)  # use base
    def memoize_interpretation(cls, *args):
        key = (cls,) + tuple(map(force_hashable, args))
        if key not in cache:
            cache[key] = cls(*args)
        return cache[key]

    with interpreter.interpretation(memoize_interpretation):
        yield cache
