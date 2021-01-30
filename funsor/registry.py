# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from funsor.typing import TypingDispatcher, get_origin


class PartialDispatcher(TypingDispatcher):
    """
    Wrapper to avoid appearance in stack traces.
    """
    def __init__(self, name, default=None):
        self.default = default if default is None else PartialDefault(default)
        super().__init__(name)


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
        # TODO make registry a WeakKeyDictionary
        self.default = default if default is None else PartialDefault(default)
        self.registry = defaultdict(lambda: PartialDispatcher('f', default=default))

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
