import functools
import inspect
import re


class lazy_property(object):
    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.fn(obj)
        setattr(obj, self.fn.__name__, value)
        return value


def getargspec(fn):
    """
    Similar to Python 2's :py:func:`inspect.getargspec` but:
    - In Python 3 uses ``getfullargspec`` to avoid ``DeprecationWarning``.
    - For builtin functions like ``torch.matmul``, falls back to attmpting
      to parse the function docstring, assuming torch-style.
    """
    assert callable(fn)
    try:
        args, vargs, kwargs, defaults, _, _, _ = inspect.getfullargspec(fn)
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
