# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

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


def quote(arg):
    """
    Serialize an object to text that can be parsed by Python.

    This is useful to save intermediate funsors to add to tests.
    """
    out = []
    _quote_inplace(arg, 0, out)
    lines = []
    for indent, line in out:
        if indent + len(line) >= 80:
            line += "  # noqa"
        lines.append(' ' * indent + line)
    return '\n'.join(lines)


def pretty(arg, maxlen=40):
    """
    Pretty print an expression. This is useful for debugging.
    """
    out = []
    _quote_inplace(arg, 0, out)
    fill = u'   \u2502' * 100
    lines = []
    for indent, line in out:
        if len(line) > maxlen:
            line = line[:maxlen] + "..."
        lines.append(fill[:indent] + line)
    return '\n'.join(lines)


@functools.singledispatch
def _quote_inplace(arg, indent, out):
    line = re.sub('\n\\s*', ' ', repr(arg))
    out.append((indent, line))


quote.inplace = _quote_inplace
quote.register = _quote_inplace.register


@quote.register(tuple)
def _(arg, indent, out):
    if not arg:
        out.append((indent, "()"))
        return
    for value in arg[:1]:
        temp = []
        quote.inplace(value, indent + 1, temp)
        i, line = temp[0]
        temp[0] = i - 1, "(" + line
        out.extend(temp)
        i, line = out[-1]
        out[-1] = i, line + ','
    for value in arg[1:]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ','
    i, line = out[-1]
    out[-1] = i, line + ')'
