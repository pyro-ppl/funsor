# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import weakref

from funsor.registry import PartialDispatcher


def apply(function, args, kwargs={}):
    return function(*args, **kwargs)


def _iter_subclasses(cls):
    yield cls
    for subcls in cls.__subclasses__():
        yield from _iter_subclasses(subcls)


def _snake_to_camel(name):
    return "".join(part.capitalize() for part in name.split("_") if part)


class WeakPartial:
    # Like ``functools.partial(fn, arg)`` but weakly referencing ``arg``.

    def __init__(self, fn, arg):
        self.fn = fn
        self.weak_arg = weakref.ref(arg)
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        arg = self.weak_arg()
        return self.fn(arg, *args, **kwargs)


class OpMeta(type):
    """
    Metaclass for :class:`Op` classes.

    This weakly caches op instances. Caching strategy is to key on
    ``args[arity:],kwargs`` and to weakly retain values. Caching requires all
    non-funsor args to be hashable; for non-hashable args, implement a derived
    metaclass with custom :meth:`hash_args_kwargs` method.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._instance_cache = weakref.WeakValueDictionary()
        cls._subclass_registry = []

        # Register all existing patterns.
        for supercls in reversed(inspect.getmro(cls)):
            for pattern, fn in getattr(supercls, "_subclass_registry", ()):
                cls.dispatcher.add(pattern, WeakPartial(fn, cls))

    @property
    def register(cls):
        return cls.dispatcher.register

    def __call__(cls, *args, **kwargs):
        args = (None,) * cls.arity + args
        bound = cls.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        args = bound.args[cls.arity :]
        kwargs = bound.kwargs
        key = cls.hash_args_kwargs(args, kwargs)
        op = cls._instance_cache.get(key, None)
        if op is None:
            op = cls._instance_cache[key] = super().__call__(*args, **kwargs)
        return op

    @staticmethod
    def hash_args_kwargs(args, kwargs):
        return args, tuple(kwargs.items())


class Op(metaclass=OpMeta):
    r"""
    Abstract base class for all mathematical operations on ground terms.

    Ops take ``arity``-many leftmost positional args that may be funsors,
    followed by additional non-funsor args and kwargs. The additional args and
    kwargs must have default values.

    When wrapping new backend ops, keep in mind these restrictions, which may
    require you to wrap backend functions before making them into ops:

    - Create new ops only by decoraing a default implementation with
      ``@UnaryOp.make``, ``@BinaryOp.make``, etc.
    - Register backend-specific implementations via ``@my_op.register(type1)``,
      ``@my_op.register(type1, type2)`` etc for arity 1, 2, etc. Patterns may
      include only the first ``arity``-many types.
    - Only the first ``arity``-many arguments may be funsors. Remaining args
      and kwargs must all be ground Python data.

    :cvar int arity: The number of funsor arguments this op takes. Must be
        defined by subclasses.
    :param \*args:
    :param \*\*kwargs: All extra arguments to this op, excluding the arguments
        up to ``.arity``,
    """

    arity = NotImplemented  # abstract

    def __init__(self, *args, **kwargs):
        super().__init__()
        cls = type(self)
        args = (None,) * cls.arity + args
        bound = cls.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        self.defaults = bound.arguments
        for key in list(self.defaults)[: cls.arity]:
            del self.defaults[key]

    @property
    def __name__(self):
        return self.name

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return apply, (type(self), (), self.defaults)

    def __repr__(self):
        return "ops." + self.__name__

    def __str__(self):
        return self.__name__

    def __call__(self, *args, **kwargs):
        # Normalize args, kwargs.
        cls = type(self)
        bound = cls.signature.bind_partial(*args, **kwargs)
        for key, value in self.defaults.items():
            bound.arguments.setdefault(key, value)
        args = bound.args
        assert len(args) >= cls.arity
        kwargs = bound.kwargs

        # Dispatch.
        fn = cls.dispatcher.partial_call(*args[: cls.arity])
        return fn(*args, **kwargs)

    def register(self, *pattern):
        if len(pattern) != self.arity:
            raise ValueError(
                f"Invalid pattern for {self}, "
                f"expected {self.arity} types but got {len(pattern)}."
            )
        return type(self).dispatcher.register(*pattern)

    @classmethod
    def subclass_register(cls, *pattern):
        def decorator(fn):
            # Register with all existing sublasses.
            for subcls in _iter_subclasses(cls):
                dispatcher = getattr(subcls, "dispatcher", None)
                if dispatcher is not None:
                    dispatcher.add(pattern, WeakPartial(fn, subcls))
            # Ensure registration with all future subclasses.
            cls._subclass_registry.append((pattern, fn))
            return fn

        return decorator

    @classmethod
    def make(cls, fn=None, *, name=None, metaclass=OpMeta, module_name="funsor.ops"):
        """
        Factory to create a new :class:`Op` subclass together with a new
        default instance of that class.

        :param callable fn: A function whose signature can be inspected.
        :returns: The new default instance.
        :rtype: Op
        """
        if not isinstance(cls.arity, int):
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} with abstract arity"
            )

        # Support use as decorator.
        if fn is None:
            return lambda fn: cls.make(fn, name=name, module_name=module_name)
        assert callable(fn)

        if name is None:
            name = fn.__name__
        assert isinstance(name, str)

        assert issubclass(metaclass, OpMeta)
        classname = _snake_to_camel(name) + "Op"  # e.g. scatter_add -> ScatterAddOp
        signature = inspect.Signature.from_callable(fn)
        op_class = metaclass(
            classname,
            (cls,),
            {
                "name": name,
                "signature": signature,
                "default": staticmethod(fn),
                "dispatcher": PartialDispatcher(fn, name),
            },
        )
        op_class.__module__ = module_name
        op = op_class()
        return op


def declare_op_types(locals_, all_, name_):
    op_types = set(
        v for v in locals_.values() if isinstance(v, type) and issubclass(v, Op)
    )
    # Adds all op types to __all__, and fix their modules.
    for typ in op_types:
        if typ.__module__ == name_:
            typ.__module__ = "funsor.ops"
        all_.append(typ.__name__)
    # Adds type(op) to locals, for each op in locals.
    for name, op in list(locals_.items()):
        if isinstance(op, Op) and type(op) not in op_types:
            locals_[type(op).__name__] = type(op)
            all_.append(type(op).__name__)
    all_.sort()


class NullaryOp(Op):
    arity = 0


class UnaryOp(Op):
    arity = 1


class BinaryOp(Op):
    arity = 2


class TernaryOp(Op):
    arity = 3


class FinitaryOp(Op):
    arity = 1  # encoded as a tuple


# Convert list to tuple for easier typing.
@FinitaryOp.subclass_register(list)
def _list_to_tuple(cls, arg, *args, **kwargs):
    arg = tuple(arg)
    op = cls(*args, **kwargs)
    return op(arg)


class TransformOp(UnaryOp):
    def set_inv(self, fn):
        """
        :param callable fn: A function that inputs an arg ``y`` and outputs a
            value ``x`` such that ``y=self(x)``.
        """
        assert callable(fn)
        self.inv = fn
        return fn

    def set_log_abs_det_jacobian(self, fn):
        """
        :param callable fn: A function that inputs two args ``x, y``, where
            ``y=self(x)``, and returns ``log(abs(det(dy/dx)))``.
        """
        assert callable(fn)
        self.log_abs_det_jacobian = fn
        return fn

    @staticmethod
    def inv(x):
        raise NotImplementedError

    @staticmethod
    def log_abs_det_jacobian(x, y):
        raise NotImplementedError


class WrappedOpMeta(OpMeta):
    """
    Metaclass for ops that wrap temporary backend ops.
    Caching strategy is to key on ``id(backend_op)`` and forget values asap.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._instance_cache = weakref.WeakValueDictionary()

    def __call__(cls, fn):
        if inspect.ismethod(fn):
            key = id(fn.__self__), fn.__func__  # e.g. t.log_abs_det_jacobian
        else:
            key = id(fn)  # e.g. t.inv
        try:
            return cls._instance_cache[key]
        except KeyError:
            op = super().__call__(fn)
            op.fn = fn  # Ensures the key id(fn) is not reused.
            cls._instance_cache[key] = op
            return op


class WrappedTransformOp(TransformOp, metaclass=WrappedOpMeta):
    """
    Wrapper for a backend ``Transform`` object that provides ``.inv`` and
    ``.log_abs_det_jacobian``. This additionally validates shapes on the first
    :meth:`__call__`.
    """

    def __init__(self, fn):
        super().__init__(fn, name=type(fn).__name__)
        self._is_validated = False

    def __call__(self, x):
        if self._is_validated:
            return super().__call__(x)

        try:
            # Check for shape metadata available after
            # https://github.com/pytorch/pytorch/pull/50547
            # https://github.com/pytorch/pytorch/pull/50581
            # https://github.com/pyro-ppl/pyro/pull/2739
            # https://github.com/pyro-ppl/numpyro/pull/876
            self.fn.domain.event_dim
            self.fn.codomain.event_dim
            self.fn.forward_shape
        except AttributeError:
            backend = self.fn.__module__.split(".")[0]
            raise NotImplementedError(
                f"{self.fn} is missing shape metadata; "
                f"try upgrading backend {backend}"
            )

        if len(x.shape) < self.fn.domain.event_dim:
            raise ValueError(f"Too few dimensions for input, in {self.name}")
        event_shape = x.shape[len(x.shape) - self.fn.domain.event_dim :]
        shape = self.fn.forward_shape(event_shape)
        if len(shape) > self.fn.codomain.event_dim:
            raise ValueError(
                f"Cannot treat transform {self.name} as an Op " "because it is batched"
            )
        self._is_validated = True
        return super().__call__(x)

    @property
    def inv(self):
        return WrappedTransformOp(self.fn.inv)

    @property
    def log_abs_det_jacobian(self):
        return LogAbsDetJacobianOp(self.fn.log_abs_det_jacobian)


class LogAbsDetJacobianOp(BinaryOp, metaclass=WrappedOpMeta):
    pass


# Op registration tables.
DISTRIBUTIVE_OPS = set()  # (add, mul) pairs
UNITS = {}  # op -> value
BINARY_INVERSES = {}  # binary op -> inverse binary op
SAFE_BINARY_INVERSES = {}  # binary op -> numerically safe inverse binary op
UNARY_INVERSES = {}  # binary op -> inverse unary op

__all__ = [
    "BINARY_INVERSES",
    "BinaryOp",
    "DISTRIBUTIVE_OPS",
    "FinitaryOp",
    "LogAbsDetJacobianOp",
    "NullaryOp",
    "Op",
    "SAFE_BINARY_INVERSES",
    "TernaryOp",
    "TransformOp",
    "UNARY_INVERSES",
    "UNITS",
    "UnaryOp",
    "WrappedTransformOp",
    "declare_op_types",
]
