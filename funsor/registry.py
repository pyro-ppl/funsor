# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from multipledispatch import Dispatcher
from multipledispatch.conflict import supercedes


class PartialDispatcher(Dispatcher):
    """
    Wrapper to avoid appearance in stack traces.
    """
    def partial_call(self, *args):
        """
        Likde :meth:`__call__` but avoids calling ``func()``.
        """
        types = tuple(map(type, args))
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if func is None:
                raise NotImplementedError(
                    'Could not find signature for %s: <%s>' %
                    (self.name, ', '.join(cls.__name__ for cls in types)))
            self._cache[types] = func
        return func


class PartialDefault:
    def __init__(self, default):
        self.default = default

    @property
    def __call__(self):
        return self.default

    def partial_call(self, *args):
        return self.default


class KeyedRegistry(object):

    def __init__(self, default=None):
        self.default = default if default is None else PartialDefault(default)
        self.registry = defaultdict(lambda: PartialDispatcher('f'))

    def register(self, key, *types):
        key = getattr(key, "__origin__", key)
        register = self.registry[key].register
        if self.default:
            objects = (object,) * len(types)
            try:
                if objects != types and supercedes(types, objects):
                    register(*objects)(self.default)
            except TypeError:
                pass  # mysterious source of ambiguity in Python 3.5 breaks this

        # This decorator supports stacking multiple decorators, which is not
        # supported by multipledipatch (which returns a Dispatch object rather
        # than the original function).
        def decorator(fn):
            register(*types)(fn)
            return fn

        return decorator

    def __contains__(self, key):
        return key in self.registry

    def __getitem__(self, key):
        key = getattr(key, "__origin__", key)
        if self.default is None:
            return self.registry[key]
        return self.registry.get(key, self.default)

    def __call__(self, key, *args):
        return self[key](*args)

    def dispatch(self, key, *args):
        return self[key].partial_call(*args)


__all__ = [
    'KeyedRegistry',
]
