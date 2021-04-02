# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from multipledispatch.dispatcher import Dispatcher, expand_tuples

from funsor.typing import Variadic, deep_type, get_origin, get_type_hints, typing_wrap


class PartialDispatcher(Dispatcher):
    """
    Wrapper to avoid appearance in stack traces.
    """

    def __init__(self, default=None, name="PartialDispatcher"):
        self.default = default if default is None else PartialDefault(default)
        super().__init__(name)
        if default is not None:
            self.add(([object],), self.default)

    def add(self, signature, func):

        # Handle annotations
        if not signature:
            annotations = get_type_hints(func)
            annotations.pop("return", None)
            if annotations:
                signature = tuple(annotations.values())

        # Handle some union types by expanding at registration time
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func)
            return

        # Handle variadic types
        signature = (
            Variadic[tuple(tp)] if isinstance(tp, list) else tp for tp in signature
        )

        signature = tuple(map(typing_wrap, signature))
        super().add(signature, func)

    def partial_call(self, *args):
        """
        Likde :meth:`__call__` but avoids calling ``func()``.
        """
        types = tuple(map(typing_wrap, map(deep_type, args)))
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if func is None:
                raise NotImplementedError(
                    "Could not find signature for %s: <%s>"
                    % (self.name, ", ".join(cls.__name__ for cls in types))
                )
            self._cache[types] = func
        return func

    def __call__(self, *args):
        return self.partial_call(*args)(*args)


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
        self.registry = defaultdict(lambda: PartialDispatcher(default=default))

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
        return get_origin(key) in self.registry

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
    "KeyedRegistry",
]
