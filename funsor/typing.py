# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import typing
import typing_extensions
import weakref

import pytypes

from multipledispatch.conflict import supercedes
from multipledispatch.dispatcher import Dispatcher
from multipledispatch.variadic import VariadicSignatureType, isvariadic


#################################
# Runtime type-checking helpers
#################################

def _type_to_typing(tp):
    if tp is object:
        tp = typing.Any
    if isinstance(tp, tuple):
        tp = typing.Union[tuple(map(_type_to_typing, tp))]
    return tp


def deep_isinstance(obj, cls):
    """replaces isinstance()"""
    return pytypes.is_of_type(obj, cls)


def deep_issubclass(subcls, cls):
    """replaces issubclass()"""
    return pytypes.is_subtype(subcls, cls)


def deep_type(obj):
    """replaces type()"""
    return pytypes.deep_type(obj)


##############################################
# Funsor-compatible typing introspection API
##############################################

def get_args(tp):
    if isinstance(tp, GenericTypeMeta):
        return getattr(tp, "__args__", tp)
    return typing_extensions.get_args(tp)


def get_origin(tp):
    if isinstance(tp, GenericTypeMeta):
        return getattr(tp, "__origin__", tp)
    return typing_extensions.get_origin(tp)


def get_type_hints(obj, globalns=None, localns=None, include_extras=False):
    if isinstance(obj, GenericTypeMeta):
        return getattr(obj, "__annotations__", {})
    return typing_extensions.get_type_hints(obj, globalns=globalns, localns=localns, include_extras=include_extras)


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
            base, = bases
            cls.__origin__ = base
        else:
            cls._type_cache = weakref.WeakValueDictionary()

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = (arg_types,)
        assert not any(isvariadic(arg_type) for arg_type in arg_types), "nested variadic types not supported"
        if arg_types not in cls._type_cache:
            assert not get_args(cls), "cannot subscript a subscripted type {}".format(cls)
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

        if not super(GenericTypeMeta, get_origin(cls)).__subclasscheck__(get_origin(subcls)):
            return False

        if len(get_args(cls)) != len(get_args(subcls)):
            return len(get_args(cls)) == 0

        return all(deep_issubclass(_type_to_typing(ps), _type_to_typing(pc))
                   for ps, pc in zip(get_args(subcls), get_args(cls)))

    def __repr__(cls):
        return get_origin(cls).__name__ + (
            "" if not get_args(cls) else
            "[{}]".format(", ".join(repr(t) for t in get_args(cls))))

    @property
    def classname(cls):
        return repr(cls)


##############################################################
# Tools and overrides for typing-compatible multipledispatch
##############################################################

class _RuntimeSubclassCheckMeta(GenericTypeMeta):
    def __call__(cls, tp):
        return tp if isinstance(tp, GenericTypeMeta) or isvariadic(tp) else cls[tp]

    def __subclasscheck__(cls, subcls):
        if isinstance(subcls, _RuntimeSubclassCheckMeta):
            subcls = subcls.__args__[0]
        return deep_issubclass(subcls, cls.__args__[0])


class typing_wrap(metaclass=_RuntimeSubclassCheckMeta):
    """
    Utility callable for overriding the runtime behavior of `typing` objects.
    """
    pass


def deep_supercedes(xs, ys):
    """typing-compatible version of multipledispatch.conflict.supercedes"""
    return supercedes(tuple(typing_wrap(_type_to_typing(x)) for x in xs),
                      tuple(typing_wrap(_type_to_typing(y)) for y in ys))


class _DeepVariadicSignatureType(VariadicSignatureType):
    pass  # TODO define __getitem__, possibly __eq__/__hash__?


class Variadic(metaclass=_DeepVariadicSignatureType):
    """
    A typing-compatible drop-in replacement for multipledispatch.variadic.Variadic.
    """
    pass  # TODO is there anything else to do here?


class TypingDispatcher(Dispatcher):
    """
    A Dispatcher class designed for compatibility with the typing standard library.
    """
    def register(self, *types):
        types = tuple(map(typing_wrap, map(_type_to_typing, types))
        if getattr(self, "default", None):  # XXX should this class have default?
            objects = (typing_wrap(typing.Any),) * len(types)
            if objects != types and deep_supercedes(types, objects):
                super().register(*objects)(self.default)
        return super().register(*types)

    def partial_call(self, *args):
        """
        Likde :meth:`__call__` but avoids calling ``func()``.
        """
        types = tuple(map(deep_type, args))
        types = tuple(map(_type_to_typing, types))
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if func is None:
                raise NotImplementedError(
                    'Could not find signature for %s: <%s>' %
                    (self.name, ', '.join(cls.__name__ for cls in types)))
            self._cache[types] = func
        return func

    def __call__(self, *args):
        return self.partial_call(*args)(*args)
