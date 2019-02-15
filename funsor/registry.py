from __future__ import absolute_import, division, print_function

from collections import defaultdict
from multipledispatch import Dispatcher


class KeyedRegistry(object):

    def __init__(self):
        self.registry = defaultdict(lambda: Dispatcher('f'))

    def register(self, key, *types):
        return self.registry[key].register(*types)

    def __call__(self, key, *args, **kwargs):
        return self.registry[key](*args, **kwargs)


__all__ = [
    'KeyedRegistry',
]
