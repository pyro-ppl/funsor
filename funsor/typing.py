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
    return type(obj)


@deep_type.register(tuple)
def _deep_type_tuple(obj):
    return typing.Tuple[tuple(map(deep_type, obj))] if obj else typing.Tuple


@deep_type.register(frozenset)
def _deep_type_frozenset(obj):
    return typing.FrozenSet[next(map(deep_type, obj))] if obj else typing.FrozenSet


_subclasscheck_registry = {}


def register_issubclass(cls):
    """
    Decorator for registering a custom ``__subclasscheck__`` method for ``cls``,
    for use in pattern matching with :func:`deep_issubclass`.
    """

    def _fn(fn):
        _subclasscheck_registry[cls] = fn
        return fn

    return _fn


register_issubclass(typing.Any)(lambda a, b: True)


@register_issubclass(typing.Union)
def _subclasscheck_union(cls, subcls):
    return any(deep_issubclass(arg, cls) for arg in get_args(subcls))


@register_issubclass(frozenset)
@register_issubclass(typing.FrozenSet)
def _subclasscheck_frozenset(cls, subcls):

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


@register_issubclass(tuple)
@register_issubclass(typing.Tuple)
def _subclasscheck_tuple(cls, subcls):

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
    """replaces issubclass()"""
    # return pytypes.is_subtype(subcls, cls)

    # handle unpacking
    if isinstance(subcls, _RuntimeSubclassCheckMeta):
        try:
            return deep_issubclass(subcls.__args__[0], cls)
        except TypeError as e:
            if e.args[0] == "issubclass() arg 1 must be a class":
                return False
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
    """replaces isinstance()"""
    # return pytypes.is_of_type(obj, cls)
    return deep_issubclass(deep_type(obj), cls)


def _type_to_typing(tp):
    if tp is object:
        tp = typing.Any
    return tp


##############################################
# Funsor-compatible typing introspection API
##############################################


def get_args(tp):
    if isinstance(tp, GenericTypeMeta) or sys.version_info[:2] < (3, 7):
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


def get_type_hints(obj, globalns=None, localns=None, **kwargs):
    return typing_extensions.get_type_hints(
        obj, globalns=globalns, localns=localns, **kwargs
    )


######################################################################
# Metaclass for generating parametric types with Tuple-like variance
######################################################################


class GenericTypeMeta(type):
    """
    Metaclass to support subtyping with parameters for pattern matching, e.g. Number[int, int].
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
        assert not any(
            isvariadic(arg_type) for arg_type in arg_types
        ), "nested variadic types not supported"
        arg_types = tuple(map(_type_to_typing, arg_types))
        if arg_types not in cls._type_cache:
            assert not get_args(cls), "cannot subscript a subscripted type {}".format(
                cls
            )
            new_dct = cls.__dict__.copy()
            new_dct.update({"__args__": arg_types})
            # type(cls) to handle GenericTypeMeta subclasses
            cls._type_cache[arg_types] = type(cls)(cls.__name__, (cls,), new_dct)
        return cls._type_cache[arg_types]

    def __subclasscheck__(cls, subcls):  # issubclass(subcls, cls)
        if cls is subcls:
            return True

        if not isinstance(subcls, GenericTypeMeta):
            return super(GenericTypeMeta, get_origin(cls)).__subclasscheck__(subcls)

        if not super(GenericTypeMeta, get_origin(cls)).__subclasscheck__(
            get_origin(subcls)
        ):
            return False

        if len(get_args(cls)) != len(get_args(subcls)):
            return len(get_args(cls)) == 0

        return all(
            deep_issubclass(_type_to_typing(ps), _type_to_typing(pc))
            for ps, pc in zip(get_args(subcls), get_args(cls))
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
    Utility callable for overriding the runtime behavior of `typing` objects.
    """

    pass


class _DeepVariadicSignatureType(type):
    def __getitem__(cls, key):
        if not isinstance(key, tuple):
            key = (key,)
        return _OrigVariadic[tuple(map(typing_wrap, key))]


class Variadic(metaclass=_DeepVariadicSignatureType):
    """
    A typing-compatible drop-in replacement for multipledispatch.variadic.Variadic.
    """

    pass
