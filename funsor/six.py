import inspect
import re

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
                fn = None
                for cls in inspect.getmro(type(arg)):
                    if cls in self._registry:
                        fn = self._registry[cls]
                        break
                if fn is None:
                    for cls in self._registry:
                        if isinstance(arg, cls):
                            fn = self._registry[cls]
                            break
                if fn is None:
                    fn = self._default
                self._cache[type(arg)] = fn
            return fn(arg)

        def register(self, cls):

            def decorator(fn):
                self._registry[cls] = fn
                self._cache.clear()
                return fn

            return decorator


def getargspec(fn):
    """
    Similar to Python 2's :py:func:`inspect.getargspec` but:
    - In Python 3 uses ``getfullargspec`` to avoid ``DeprecationWarning``.
    - For builtin functions like ``torch.matmul``, falls back to attmpting
      to parse the function docstring, assuming torch-style.
    """
    assert callable(fn)
    try:
        if six.PY3:
            args, vargs, kwargs, defaults, _, _, _ = inspect.getfullargspec(fn)
        else:
            args, vargs, kwargs, defaults = inspect.getargspec(fn)
    except TypeError:
        # Fall back to attmpting to parse a PyTorch-style docstring.
        match = re.match(r"\s{}\(([^)]*)\)".format(fn.__name__), fn.__doc__)
        if match is None:
            raise
        parts = match.group(1).split(", ")
        args = [a.split("=")[0] for a in parts]
        if not all(re.match(r"^[^\d\W]\w*\Z", arg) for arg in args):
            raise
        vargs = None
        kwargs = None
        defaults = ()  # Ignore defaults.
    return args, vargs, kwargs, defaults


__all__ = [
    'getargspec',
    'singledispatch',
]
