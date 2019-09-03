from collections import defaultdict

from multipledispatch import Dispatcher


class KeyedRegistry(object):

    def __init__(self, default=None):
        self.default = default
        self.registry = defaultdict(lambda: Dispatcher('f'))

    def register(self, key, *types):
        key = getattr(key, "__origin__", key)
        register = self.registry[key].register
        if self.default:
            objects = (object,) * len(types)
            if objects != types:
                register(*objects)(self.default)

        # This decorator supports stacking multiple decorators, which is not
        # supported by multipledipatch (which returns a Dispatch object rather
        # than the original function).
        def decorator(fn):
            register(*types)(fn)
            return fn

        return decorator

    def __call__(self, key, *args, **kwargs):
        key = getattr(key, "__origin__", key)
        if self.default is None or key in self.registry:
            return self.registry[key](*args, **kwargs)
        return self.default(*args, **kwargs)


__all__ = [
    'KeyedRegistry',
]
