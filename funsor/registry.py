from __future__ import absolute_import, division, print_function

from collections import defaultdict

from multipledispatch import Dispatcher
from multipledispatch.variadic import Variadic


# class KeyedRegistry(object):
# 
#     def __init__(self, default=None):
#         self.default = default
#         self.registry = defaultdict(lambda: Dispatcher('f'))
# 
#     def register(self, key, *types):
#         register = self.registry[key].register
#         if self.default:
#             objects = (object,) * len(types)
#             if objects != types:
#                 register(*objects)(self.default)
# 
#         # This decorator supports stacking multiple decorators, which is not
#         # supported by multipledipatch (which returns a Dispatch object rather
#         # than the original function).
#         def decorator(fn):
#             register(*types)(fn)
#             return fn
# 
#         return decorator
# 
#     def __call__(self, key, *args, **kwargs):
#         if self.default is None or key in self.registry:
#             return self.registry[key](*args, **kwargs)
#         return self.default(*args, **kwargs)


class KeyedRegistry(Dispatcher):

    def __init__(self, default=None):
        super(KeyedRegistry, self).__init__('f')
        if default is not None:
            self.register(Variadic[object])(default)

    def __call__(self, key, *args, **kwargs):
        types = (key,) + tuple(type(arg) for arg in args)
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    "No signature for {}: {}".format(
                        self.name, ', '.join(t.__name__ for t in types)))
            self._cache[types] = func
        return func(*args, **kwargs)


__all__ = [
    'KeyedRegistry',
]
