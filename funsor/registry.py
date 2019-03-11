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
                self.register_impl(key, self.default, *objects)

        # This decorator supports stacking multiple decorators, which is not
        # supported by multipledipatch (which returns a Dispatch object rather
        # than the original function).
        def decorator(fn):
            self.register_impl(key, fn, *types)
            return fn

        return decorator

    def register_impl(self, key, impl, *types):
        self.registry[key].register(*types)(impl)

    def __call__(self, key, *args, **kwargs):
        if self.default is None or key in self.registry:
            return self.registry[key](*args, **kwargs)
        return self.default(*args, **kwargs)


__all__ = [
    'KeyedRegistry',
]
