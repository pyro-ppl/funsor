# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import sys
import typing
import weakref

import typing_extensions
from multipledispatch.variadic import Variadic as _OrigVariadic
from multipledispatch.variadic import isvariadic

#################################
# Runtime type-checking helpers
#################################


@functools.singledispatch
def deep_type(obj):
    """
    An enhanced version of :func:`type` that reconstructs structured :mod:`typing`` types
    for a limited set of immutable data structures, notably ``tuple`` and ``frozenset``.
    Mostly intended for internal use in Funsor interpretation pattern-matching.

    Example::

        assert deep_type((1, ("a",))) is typing.Tuple[int, typing.Tuple[str]]
        assert deep_type(frozenset(["a"])) is typing.FrozenSet[str]
    """
    # compare to pytypes.deep_type(obj)
    return type(obj)


@deep_type.register(tuple)
def _deep_type_tuple(obj):
    return typing.Tuple[tuple(map(deep_type, obj))] if obj else typing.Tuple


@deep_type.register(frozenset)
def _deep_type_frozenset(obj):
    if not obj:
        return typing.FrozenSet
    tp = deep_type(next(iter(obj)))
    for x in obj:
        if not deep_isinstance(x, tp):
            tp = get_origin(tp)
        if not deep_isinstance(x, tp):
            raise NotImplementedError(
                f"TODO handle inhomogeneous frozensets: {str(obj)}"
            )
    return typing.FrozenSet[tp]


_subclasscheck_registry = {}


def register_subclasscheck(cls):
    """
    Decorator for registering a custom ``__subclasscheck__`` method for ``cls``
    which is only ever invoked in :func:`deep_issubclass`.

    This is primarily intended for working with the :mod:`typing` library at runtime.
    Prefer overriding ``__subclasscheck__`` in the usual way with a metaclass
    where possible.
    """

    def _fn(fn):
        _subclasscheck_registry[cls] = fn
        return fn

    return _fn


@register_subclasscheck(typing.Any)
def _subclasscheck_any(cls, subcls):
    return True


@register_subclasscheck(typing.Union)
def _subclasscheck_union(cls, subcls):
    """A basic ``__subclasscheck__`` method for :class:`~typing.Union`."""
    return any(deep_issubclass(subcls, arg) for arg in get_args(cls))


@register_subclasscheck(frozenset)
@register_subclasscheck(typing.FrozenSet)
def _subclasscheck_frozenset(cls, subcls):
    """A basic ``__subclasscheck__`` method for :class:`~typing.FrozenSet`."""

    if not issubclass(get_origin(subcls), frozenset):
        return False

    cls_args, subcls_args = get_args(cls), get_args(subcls)

    if not cls_args:
        return True

    if not subcls_args:
        return cls_args[0] is typing.Any

    return len(subcls_args) == len(cls_args) == 1 and all(
        deep_issubclass(a, b) for a, b in zip(subcls_args, cls_args)
    )


@register_subclasscheck(tuple)
@register_subclasscheck(typing.Tuple)
def _subclasscheck_tuple(cls, subcls):
    """A basic ``__subclasscheck__`` method for :class:`~typing.Tuple`."""

    if not issubclass(get_origin(subcls), get_origin(cls)):
        return False

    cls_args, subcls_args = get_args(cls), get_args(subcls)

    if not cls_args:  # cls is base Tuple
        return True

    if not subcls_args:
        return cls_args[0] is typing.Any

    if cls_args[-1] is Ellipsis:  # cls variadic
        if subcls_args[-1] is Ellipsis:  # both variadic
            return deep_issubclass(subcls_args[0], cls_args[0])
        return all(deep_issubclass(a, cls_args[0]) for a in subcls_args)

    if subcls_args[-1] is Ellipsis:  # only subcls variadic
        # issubclass(Tuple[A, ...], Tuple[X, Y]) == False
        return False

    # neither variadic
    return len(cls_args) == len(subcls_args) and all(
        deep_issubclass(a, b) for a, b in zip(subcls_args, cls_args)
    )


@functools.lru_cache(maxsize=None)
def deep_issubclass(subcls, cls):
    """
    Enhanced version of :func:`issubclass` that can handle structured types,
    including Funsor terms, :class:`~typing.Tuple`, and :class:`~typing.FrozenSet`.

    Does not support more advanced :mod:`typing` features such as
    :class:`~typing.TypeVar`, arbitrary :class:`~typing.Generic` subtypes,
    forward references, or mutable collection types like :class:`~typing.List`.
    Will attempt to fall back to :func:`issubclass` when it encounters a type in
    ``subcls`` or ``cls`` that it does not understand.

    Usage::

        class A: pass
        class B(A): pass

        assert deep_issubclass(typing.Tuple[int, B], typing.Tuple[int, A])
        assert not deep_issubclass(typing.Tuple[int, A], typing.Tuple[int, B])

        assert deep_issubclass(typing.Tuple[A, A], typing.Tuple[A, ...])
        assert not deep_issubclass(typing.Tuple[B], typing.Tuple[A, ...])

    :param subcls: A class that may be a subclass of ``cls``.
    :param cls: A class that may be a parent class of ``subcls``.
    """
    # compare to pytypes.is_subtype(subcls, cls)

    # handle unpacking
    if isinstance(subcls, _RuntimeSubclassCheckMeta):
        try:
            return deep_issubclass(subcls.__args__[0], cls)
        except TypeError as e:
            if e.args[0] == "issubclass() arg 1 must be a class":
                return deep_issubclass(get_origin(subcls.__args__[0]), cls)
            raise

    if get_origin(subcls) is typing.Union:
        return all(deep_issubclass(arg, cls) for arg in get_args(subcls))

    if subcls is typing.Any:
        return cls is typing.Any

    try:
        return _subclasscheck_registry[get_origin(cls)](cls, subcls)
    except KeyError:
        return issubclass(subcls, cls)


def deep_isinstance(obj, cls):
    """
    Enhanced version of :func:`isinstance` that can handle basic structured :mod:`typing` types,
    including Funsor terms and other :class:`~funsor.typing.GenericTypeMeta` instances,
    :class:`~typing.Union`, :class:`~typing.Tuple`, and :class:`~typing.FrozenSet`.

    Does not support :class:`~typing.TypeVar`, arbitrary :class:`~typing.Generic`,
    forward references, or mutable generic collection types like :class:`~typing.List`.
    Will attempt to fall back to :func:`isinstance` when it encounters
    an unsupported type in ``obj`` or ``cls``.

    Usage::

        x = (1, ("a", "b"))
        assert deep_isinstance(x, typing.Tuple[int, tuple])
        assert deep_isinstance(x, typing.Tuple[typing.Any, typing.Tuple[str, ...]])

    :param obj: An object that may be an instance of ``cls``.
    :param cls: A class that may be a parent class of ``obj``.
    """

    # compare to pytypes.is_of_type(obj, cls)
    try:
        return deep_issubclass(deep_type(obj), cls)
    except TypeError:
        return isinstance(obj, cls)


def _type_to_typing(tp):
    if tp is object:
        tp = typing.Any
    return tp


##############################################
# Funsor-compatible typing introspection API
##############################################


def get_args(tp):
    if isinstance(tp, (GenericTypeMeta, type)) or sys.version_info[:2] < (3, 7):
        result = getattr(tp, "__args__", None)
    else:
        result = typing_extensions.get_args(tp)
    return () if result is None else result


def get_origin(tp):
    if isinstance(tp, GenericTypeMeta) or sys.version_info[:2] < (3, 7):
        result = getattr(tp, "__origin__", None)
    else:
        result = typing_extensions.get_origin(tp)
    return tp if result is None else result


reuse_upstream_documentation = False  # upstream docs are not sphinx compatible
if reuse_upstream_documentation and sys.version_info[:2] >= (3, 7):
    get_args = functools.wraps(typing_extensions.get_args)(get_args)
    get_origin = functools.wraps(typing_extensions.get_origin)(get_origin)


@functools.wraps(typing.get_type_hints)
def get_type_hints(obj, globalns=None, localns=None, **kwargs):
    return typing.get_type_hints(obj, globalns=globalns, localns=localns, **kwargs)


######################################################################
# Metaclass for generating parametric types with Tuple-like variance
######################################################################


class GenericTypeMeta(type):
    """
    Metaclass to support subtyping with parameters for pattern matching, e.g. ``Number[int, int]``.
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, "__args__"):
            cls.__args__ = ()
        if cls.__args__:
            (base,) = bases
            cls.__origin__ = base
        else:
            cls._type_cache = weakref.WeakValueDictionary()

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = (arg_types,)
        arg_types = tuple(map(_type_to_typing, arg_types))
        try:
            return cls._type_cache[arg_types]
        except KeyError:
            assert not get_args(cls), "cannot subscript a subscripted type {}".format(
                cls
            )
            assert not any(
                isvariadic(arg_type) for arg_type in arg_types
            ), "nested variadic types not supported"
            new_dct = cls.__dict__.copy()
            new_dct.update({"__args__": arg_types})
            # type(cls) to handle GenericTypeMeta subclasses
            result = type(cls)(cls.__name__, (cls,), new_dct)
            cls._type_cache[arg_types] = result
            return result

    def __subclasscheck__(cls, subcls):  # issubclass(subcls, cls)
        if cls is subcls:
            return True

        cls_origin = get_origin(cls)
        if not isinstance(subcls, GenericTypeMeta):
            return super(GenericTypeMeta, cls_origin).__subclasscheck__(subcls)

        if not super(GenericTypeMeta, cls_origin).__subclasscheck__(get_origin(subcls)):
            return False

        cls_args, subcls_args = get_args(cls), get_args(subcls)
        if len(cls_args) != len(subcls_args):
            return len(cls_args) == 0

        return all(
            deep_issubclass(_type_to_typing(ps), _type_to_typing(pc))
            for ps, pc in zip(subcls_args, cls_args)
        )

    def __repr__(cls):
        return get_origin(cls).__name__ + (
            ""
            if not get_args(cls)
            else "[{}]".format(", ".join(repr(t) for t in get_args(cls)))
        )


##############################################################
# Tools and overrides for typing-compatible multipledispatch
##############################################################


class _RuntimeSubclassCheckMeta(GenericTypeMeta):
    def __call__(cls, tp):
        tp = _type_to_typing(tp)
        return tp if isinstance(tp, GenericTypeMeta) or isvariadic(tp) else cls[tp]

    def __subclasscheck__(cls, subcls):
        return deep_issubclass(subcls, cls.__args__[0])


class typing_wrap(metaclass=_RuntimeSubclassCheckMeta):
    """
    Utility callable for overriding the runtime behavior of :mod:`typing` objects.
    """

    pass


class _DeepVariadicSignatureType(type):
    def __getitem__(cls, key):
        if not isinstance(key, tuple):
            key = (key,)
        return _OrigVariadic[tuple(map(typing_wrap, key))]


class Variadic(metaclass=_DeepVariadicSignatureType):
    """
    A typing-compatible drop-in replacement for :class:`~multipledispatch.variadic.Variadic`.
    """

    pass


__all__ = [
    "GenericTypeMeta",
    "Variadic",
    "deep_isinstance",
    "deep_issubclass",
    "deep_type",
    "get_args",
    "get_origin",
    "get_type_hints",
    "register_subclasscheck",
    "typing_wrap",
]
