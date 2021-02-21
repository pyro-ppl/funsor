# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from funsor.interpretations import Interpretation
from funsor.interpreter import get_interpretation


@contextlib.contextmanager
def memoize(cache=None):
    base_interpretation = get_interpretation()
    with MemoizeInterpretation(base_interpretation, cache) as interp:
        yield interp.cache


class MemoizeInterpretation(Interpretation):
    """
    Exploit cons-hashing to do implicit common subexpression elimination
    """

    def __init__(self, base_interpretation, cache=None):
        super().__init__(f"memoize/{base_interpretation.__name__}")
        self.base_interpretation = base_interpretation
        if cache is None:
            cache = {}
        else:
            assert isinstance(cache, dict)
        self.cache = cache

    @property
    def is_total(self):
        return self.base_interpretation.is_total

    def interpret(self, cls, *args):
        key = self.make_hash_key(cls, *args)
        value = self.cache.get(key)
        if value is None:
            self.cache[key] = value = self.base_interpretation.interpret(cls, *args)
        return value


__all__ = ["memoize"]
