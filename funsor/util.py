# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import pprint
import re
import sys

import numpy as np

_FUNSOR_BACKEND = "numpy"
_JAX_LOADED = False
_JAX_COMPILED_FUNCTION_TYPE = None


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

    # This is needed to pacify sphinx.
    def __call__(self):
        raise ValueError


def getargspec(fn):
    """
    Similar to Python 2's :py:func:`inspect.getargspec` but:
    - In Python 3 uses ``getfullargspec`` to avoid ``DeprecationWarning``.
    - For builtin functions like ``torch.matmul`` or ``numpy.matmul``, falls back to
      attempting to parse the function docstring, assuming torch-style or numpy-style.
    """
    assert callable(fn)
    try:
        args, vargs, kwargs, defaults, _, _, _ = inspect.getfullargspec(fn)
    except TypeError:
        # Fall back to attempting to parse a PyTorch/NumPy-style docstring.
        match = re.match(r"\s*{}\(([^)]*)\)".format(fn.__name__), fn.__doc__)
        if match is None:
            raise
        parts = re.sub(r"[\[\]]", "", match.group(1)).split(", ")
        args = [a.split("=")[0] for a in parts if a not in ["/", "*"]]
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
        lines.append(" " * indent + line)
    return "\n".join(lines)


def pretty(arg, linewidth=75, threshold=1000):
    """
    Pretty print an expression. This is useful for debugging.

    :param int linewidth: Maximum linewidth before summarization.
    :param int threshold: Total number of array elements which trigger
        summarization rather than full repr (default 1000). To always use the full
        repr without summarization, pass sys.maxsize
    :returns: A pretty string.
    :rtype: str
    """
    out = []
    try:
        old = quote.printoptions.copy()
        quote.printoptions["threshold"] = threshold
        _quote_inplace(arg, 0, out)
    finally:
        quote.printoptions.update(old)
    fill = "   \u2502" * 100
    lines = []
    for indent, line in out:
        if len(line) > linewidth:
            line = line[: linewidth - 3] + "..."
        lines.append(fill[:indent] + line)
    return "\n".join(lines)


def _pprint_funsor(pprinter, object, stream, indent, allowance, context, level):
    out = []
    try:
        old = quote.printoptions.copy()
        quote.printoptions["threshold"] = 0  # Omit arrays from pprint.
        _quote_inplace(object, 0, out)
    finally:
        quote.printoptions.update(old)

    # This depends on internals of the pprint.PrettyPrinter class.
    write = stream.write
    for i, (extra_indent, line) in enumerate(out):
        max_width = pprinter._width - indent - extra_indent
        if len(line) > max_width:
            line = line[: max_width - 3] + "..."
        if i > 0:
            write("\n")
            write(" " * (indent + extra_indent))
        write(line)


def register_pprint(cls):
    if hasattr(cls, "__repr__"):
        # This depends on internals of the pprint.PrettyPrinter class.
        pprint.PrettyPrinter._dispatch[cls.__repr__] = _pprint_funsor


@functools.singledispatch
def _quote_inplace(arg, indent, out):
    line = re.sub("\n\\s*", " ", repr(arg))
    out.append((indent, line))


quote.inplace = _quote_inplace
quote.register = _quote_inplace.register
quote.printoptions = {"threshold": sys.maxsize}
quote.reprtypes = ()


def _quote_repr(arg, indent, out):
    out.append((indent, repr(arg)))


def _quote_register_repr(typ):
    quote.register(typ)(_quote_repr)
    quote.reprtypes = tuple({typ}.union(quote.reprtypes))


quote.register_repr = _quote_register_repr
quote.register_repr(int)
quote.register_repr(float)
quote.register_repr(str)


@quote.register(tuple)
def _(arg, indent, out):
    if all(isinstance(value, quote.reprtypes) for value in arg):
        out.append((indent, repr(arg)))
        return

    for value in arg[:1]:
        temp = []
        quote.inplace(value, indent + 1, temp)
        i, line = temp[0]
        temp[0] = i - 1, "(" + line
        out.extend(temp)
        i, line = out[-1]
        out[-1] = i, line + ","
    for value in arg[1:]:
        quote.inplace(value, indent + 1, out)
        i, line = out[-1]
        out[-1] = i, line + ","
    i, line = out[-1]
    out[-1] = i, line + ")"


@quote.register(np.ndarray)
def _quote(arg, indent, out):
    """
    Work around NumPy ndarray not supporting reproducible repr.
    """
    if arg.size >= quote.printoptions["threshold"]:
        data = "..." + " x ".join(str(d) for d in arg.shape) + "..."
    else:
        data = repr(arg.tolist())
    out.append((indent, f"np.array({data}, dtype=np.{arg.dtype})"))


def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.
    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))


def set_backend(backend):
    """
    Set backend for Funsor.

    Currently three backends are supported: "numpy" (default), "torch", and
    "jax". Funsor runs with only one backend at a time.

    .. note: When `jax` backend is set, we cannot revert back to the default
    `numpy` backend because we dispatch to using `jax.numpy` all ops with
    `numpy.ndarray` or `numpy.generic` inputs.

    :param str backend: either "numpy", "torch", or "jax".
    """
    global _FUNSOR_BACKEND, _JAX_LOADED
    if _FUNSOR_BACKEND == backend:
        return

    if backend == "numpy":
        if _JAX_LOADED:
            raise ValueError(
                "Cannot revert back to NumPy backend when JAX backend has been set."
            )
        else:
            _FUNSOR_BACKEND = "numpy"
    elif backend == "torch":
        _FUNSOR_BACKEND = "torch"

        import torch  # noqa: F401

        import funsor.torch  # noqa: F401
    elif backend == "jax":
        _FUNSOR_BACKEND = "jax"
        _JAX_LOADED = True
        global _JAX_COMPILED_FUNCTION_TYPE

        import jax  # noqa: F401

        import funsor.jax  # noqa: F401

        if _JAX_COMPILED_FUNCTION_TYPE is None:
            _JAX_COMPILED_FUNCTION_TYPE = type(jax.jit(lambda: 0))
    else:
        raise ValueError(
            "backend should be either 'numpy', 'torch', or 'jax'"
            ", got {}".format(backend)
        )


def get_backend():
    """
    Get the current backend of Funsor.

    :return: either "numpy", "torch", or "jax".
    :rtype: str
    """
    return _FUNSOR_BACKEND


def get_default_dtype():
    """
    Get the current default floating point.

    :return: floating point dtype.
    :rtype: str
    """
    backend = get_backend()
    if backend == "torch":
        torch = sys.modules.get("torch")
        return str(torch.get_default_dtype()).split(".")[1]
    elif backend == "numpy":
        np = sys.modules.get("numpy")
        return np.dtype(np.float_).name
    elif backend == "jax":
        np = sys.modules.get("jax.numpy")
        return np.dtype(np.float_).name


def get_tracing_state():
    torch = sys.modules.get("torch")
    if torch is not None:
        return torch._C._get_tracing_state()
    return None


def is_nn_module(x):
    torch = sys.modules.get("torch")
    if torch is not None:
        return isinstance(x, torch.nn.Module)
    return False


def is_jax_compiled_function(x):
    jax = sys.modules.get("jax")
    if jax is not None:
        return isinstance(x, _JAX_COMPILED_FUNCTION_TYPE)
    return False


def as_callable(fn):
    """
    Converts nn.Modules ``m`` to ``m.forward``.
    """
    if is_nn_module(fn):
        return fn.forward
    return fn


def methodof(cls, name=None):
    """
    Decorator to set the named method of the given class. Can be stacked.

    Example usage::

       @methodof(MyClass)
       def __call__(self, x):
           return x
    """

    def decorator(fn):
        name_ = name
        if name_ is None:
            fn_ = fn
            while not hasattr(fn_, "__name__"):
                if isinstance(fn_, property):
                    fn_ = fn_.fget
                else:
                    fn_ = fn_.__func__
            name_ = fn_.__name__
        setattr(cls, name_, fn)
        return fn

    return decorator
