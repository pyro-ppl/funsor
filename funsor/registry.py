# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from funsor.typing import TypingDispatcher, _type_to_typing, deep_type, get_origin


class PartialDispatcher(TypingDispatcher):
    """
    Wrapper to avoid appearance in stack traces.
    """
    def __init__(self, name, default=None):
        self.default = default if default is None else PartialDefault(default)
        super().__init__(name)

    def partial_call(self, *args):
        """
        Likde :meth:`__call__` but avoids calling ``func()``.
        """
        types = tuple(map(deep_type, args))
        types = tuple(map(_type_to_typing, types))
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
        self.registry = defaultdict(lambda: PartialDispatcher('f', default=default))
        self.default = PartialDefault(default) if default is not None else default

    def register(self, key, *types):
        register = self.registry[get_origin(key)].register

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
        key = get_origin(key)
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
