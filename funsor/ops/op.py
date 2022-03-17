# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import inspect
import weakref
from collections import OrderedDict

from funsor.registry import PartialDispatcher
from funsor.util import methodof


def apply(function, args, kwargs={}):
    return function(*args, **kwargs)


def _get_name(fn):
    return getattr(fn, "__name__", type(fn).__name__)


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


_TRACE = None
_TRACE_FILTER_ARGS = None


@contextlib.contextmanager
def trace_ops(filter_args):
    global _TRACE, _TRACE_FILTER_ARGS
    assert _TRACE is None, "not reentrant"
    try:
        _TRACE = OrderedDict()
        _TRACE_FILTER_ARGS = filter_args
        yield _TRACE
    finally:
        _TRACE = None
        _TRACE_FILTER_ARGS = None


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

    :cvar int ~.arity: The number of funsor arguments this op takes. Must be
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
        global _TRACE, _TRACE_FILTER_ARGS
        raw_args = args

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
        if _TRACE is None or not _TRACE_FILTER_ARGS(raw_args):
            result = fn(*args, **kwargs)
        else:
            # Trace this op but avoid tracing internal ops.
            try:
                trace, _TRACE = _TRACE, None
                result = fn(*args, **kwargs)
                trace.setdefault(id(result), (result, self, raw_args))
            finally:
                _TRACE = trace

        return result

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
    def make(cls, fn=None, *, name=None, metaclass=None, module_name="funsor.ops"):
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
            return lambda fn: cls.make(
                fn, name=name, metaclass=metaclass, module_name=module_name
            )
        assert callable(fn)

        if name is None:
            name = _get_name(fn)
        assert isinstance(name, str)

        if metaclass is None:
            metaclass = type(cls)
        assert issubclass(metaclass, OpMeta)

        classname = _snake_to_camel(name) + "Op"  # e.g. scatter_add -> ScatterAddOp
        signature = inspect.Signature.from_callable(fn)
        op_class = metaclass(
            classname,
            (cls,),
            {
                "__doc__": fn.__doc__,
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


class ReductionOp(UnaryOp):
    """
    Reduction operations are defined in a broad sense - not only
    associative operations. This helps to unify find_domain logic.
    """

    pass


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

    def hash_args_kwargs(self, args, kwargs):
        if args:
            (fn,) = args
            if inspect.ismethod(fn):
                args = id(fn.__self__), fn.__func__  # e.g. t.log_abs_det_jacobian
            else:
                args = (id(fn),)  # e.g. t.inv
        return super().hash_args_kwargs(args, kwargs)


@TransformOp.make(metaclass=WrappedOpMeta)
def wrapped_transform(x, fn, *, validate_args=True):
    """
    Wrapper for a backend ``Transform`` object that provides ``.inv`` and
    ``.log_abs_det_jacobian``. This additionally validates shapes on the first
    :meth:`__call__`.
    """
    if not validate_args:
        return fn(x)

    try:
        # Check for shape metadata available after
        # https://github.com/pytorch/pytorch/pull/50547
        # https://github.com/pytorch/pytorch/pull/50581
        # https://github.com/pyro-ppl/pyro/pull/2739
        # https://github.com/pyro-ppl/numpyro/pull/876
        fn.domain.event_dim
        fn.codomain.event_dim
        fn.forward_shape
    except AttributeError:
        backend = fn.__module__.split(".")[0]
        raise NotImplementedError(
            f"{fn} is missing shape metadata; try upgrading backend {backend}"
        )

    if len(x.shape) < fn.domain.event_dim:
        raise ValueError(f"Too few dimensions for input, in {fn.__name_}")
    event_shape = x.shape[len(x.shape) - fn.domain.event_dim :]
    shape = fn.forward_shape(event_shape)
    if len(shape) > fn.codomain.event_dim:
        raise ValueError(
            f"Cannot treat transform {_get_name(fn)} as an Op because it is batched"
        )

    return fn(x)


WrappedTransformOp = type(wrapped_transform)


@methodof(WrappedTransformOp)
@property
def inv(self):
    fn = self.defaults["fn"]
    return WrappedTransformOp(fn=fn.inv)


@methodof(WrappedTransformOp)
@property
def log_abs_det_jacobian(self):
    fn = self.defaults["fn"]
    return LogAbsDetJacobianOp(fn=fn.log_abs_det_jacobian)


@BinaryOp.make(metaclass=WrappedOpMeta)
def log_abs_det_jacobian(x, y, fn):
    return fn(x, y)


LogAbsDetJacobianOp = type(log_abs_det_jacobian)


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
    "log_abs_det_jacobian",
    "wrapped_transform",
]
