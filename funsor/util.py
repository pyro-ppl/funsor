# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import re
import typing
import weakref

import numpy as np
from multipledispatch.variadic import isvariadic


_FUNSOR_BACKEND = "numpy"
_JAX_LOADED = False


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


@quote.register(np.ndarray)
def _quote(arg, indent, out):
    """
    Work around NumPy ndarray not supporting reproducible repr.
    """
    out.append((indent, "np.array({}, dtype=np.{})".format(repr(arg.tolist()), arg.dtype)))


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
    strict = kwargs.pop('strict', False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError('shape mismatch: objects cannot be broadcast to a single shape: {}'.format(
                    ' vs '.join(map(str, shapes))))
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

    if backend == "numpy":
        if _JAX_LOADED:
            raise ValueError("Cannot revert back to NumPy backend when JAX backend has been set.")
        else:
            _FUNSOR_BACKEND = "numpy"
    elif backend == "torch":
        _FUNSOR_BACKEND = "torch"

        import torch  # noqa: F401
        import funsor.torch  # noqa: F401
    elif backend == "jax":
        _FUNSOR_BACKEND = "jax"
        _JAX_LOADED = True

        import jax  # noqa: F401
        import funsor.jax  # noqa: F401
    else:
        raise ValueError("backend should be either 'numpy', 'torch', or 'jax'"
                         ", got {}".format(backend))


def get_backend():
    """
    Get the current backend of Funsor.

    :return: either "numpy", "torch", or "jax".
    :rtype: str
    """
    return _FUNSOR_BACKEND


def get_tracing_state():
    if _FUNSOR_BACKEND == "torch":
        import torch

        return torch._C._get_tracing_state()
    else:
        return None


def is_nn_module(x):
    if _FUNSOR_BACKEND == "torch":
        import torch

        return isinstance(x, torch.nn.Module)
    return False


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
                fn_ = fn_.__func__
            name_ = fn_.__name__
        setattr(cls, name_, fn)
        return fn
    return decorator


class GenericTypeMeta(type):
    """
    Metaclass to support subtyping with parameters for pattern matching, e.g. Number[int, int].
    """
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, "__args__"):
            cls.__args__ = ()
        if cls.__args__:
            base, = bases
            cls.__origin__ = base
        else:
            cls._type_cache = weakref.WeakValueDictionary()

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = (arg_types,)
        assert not any(isvariadic(arg_type) for arg_type in arg_types), "nested variadic types not supported"
        # switch tuple to typing.Tuple
        arg_types = tuple(typing.Tuple if arg_type is tuple else arg_type for arg_type in arg_types)
        if arg_types not in cls._type_cache:
            assert not cls.__args__, "cannot subscript a subscripted type {}".format(cls)
            new_dct = cls.__dict__.copy()
            new_dct.update({"__args__": arg_types})
            # type(cls) to handle GenericTypeMeta subclasses
            cls._type_cache[arg_types] = type(cls)(cls.__name__, (cls,), new_dct)
        return cls._type_cache[arg_types]

    def __subclasscheck__(cls, subcls):  # issubclass(subcls, cls)
        if cls is subcls:
            return True
        if not isinstance(subcls, GenericTypeMeta):
            return super(GenericTypeMeta, getattr(cls, "__origin__", cls)).__subclasscheck__(subcls)

        cls_origin = getattr(cls, "__origin__", cls)
        subcls_origin = getattr(subcls, "__origin__", subcls)
        if not super(GenericTypeMeta, cls_origin).__subclasscheck__(subcls_origin):
            return False

        if cls.__args__:
            if not subcls.__args__:
                return False
            if len(cls.__args__) != len(subcls.__args__):
                return False
            for subcls_param, param in zip(subcls.__args__, cls.__args__):
                if not _issubclass_tuple(subcls_param, param):
                    return False
        return True

    def __repr__(cls):
        return cls.__name__ + (
            "" if not cls.__args__ else
            "[{}]".format(", ".join(repr(t) for t in cls.__args__)))

    @lazy_property
    def classname(cls):
        return repr(cls)


def _issubclass_tuple(subcls, cls):
    """
    utility for structural subtype checking with tuple subexpressions
    """
    # so much boilerplate...
    cls_is_union = hasattr(cls, "__origin__") and (cls.__origin__ or cls) is typing.Union
    if isinstance(cls, tuple) or cls_is_union:
        return any(_issubclass_tuple(subcls, option)
                   for option in (getattr(cls, "__args__", []) if cls_is_union else cls))

    subcls_is_union = hasattr(subcls, "__origin__") and (subcls.__origin__ or subcls) is typing.Union
    if isinstance(subcls, tuple) or subcls_is_union:
        return any(_issubclass_tuple(option, cls)
                   for option in (getattr(subcls, "__args__", []) if subcls_is_union else subcls))

    subcls_is_tuple = hasattr(subcls, "__origin__") and (subcls.__origin__ or subcls) in (tuple, typing.Tuple)
    cls_is_tuple = hasattr(cls, "__origin__") and (cls.__origin__ or cls) in (tuple, typing.Tuple)
    if subcls_is_tuple != cls_is_tuple:
        return False
    if not cls_is_tuple:
        return issubclass(subcls, cls)
    if not cls.__args__:
        return True
    if not subcls.__args__ or len(subcls.__args__) != len(cls.__args__):
        return False

    return all(_issubclass_tuple(a, b) for a, b in zip(subcls.__args__, cls.__args__))
