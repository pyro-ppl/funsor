from __future__ import absolute_import, division, print_function

import inspect

import six

try:
    from functools import singledispatch  # only in python 3
except ImportError:

    class singledispatch(object):
        def __init__(self, default):
            self._default = default
            self._registry = {}
            self._cache = {}

        def __call__(self, arg):
            try:
                fn = self._cache[type(arg)]
            except KeyError:
                fn = self._default
                for cls in inspect.getmro(type(arg)):
                    if cls in self._registry:
                        fn = self._registry[cls]
                        break
                self._cache[type(arg)] = fn
            return fn(arg)

        def register(self, cls):

            def decorator(fn):
                self._registry[cls] = fn
                self._cache.clear()
                return fn

            return decorator


def getargspec(fn):
    """wrapper to remove annoying DeprecationWarning for inspect.getargspec in Py3"""
    if six.PY3:
        args, vargs, kwargs, defaults, _, _, _ = inspect.getfullargspec(fn)
    else:
        args, vargs, kwargs, defaults = inspect.getargspec(fn)
    return args, vargs, kwargs, defaults


__all__ = [
    'getargspec',
    'singledispatch',
]
