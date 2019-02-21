from __future__ import absolute_import, division, print_function

from collections import defaultdict

from multipledispatch import Dispatcher


class KeyedRegistry(object):

    def __init__(self, default=None):
        self.default = default
        self.registry = defaultdict(lambda: Dispatcher('f'))

    def register(self, key, *types):
        if self.default:
            objects = (object,) * len(types)
            if objects != types:
                self.registry[key].register(*objects)(self.default)
        return self.registry[key].register(*types)

    def __call__(self, key, *args, **kwargs):
        if self.default is None or key in self.registry:
            return self.registry[key](*args, **kwargs)
        return self.default(*args, **kwargs)


__all__ = [
    'KeyedRegistry',
]
