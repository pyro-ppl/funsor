# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import weakref

from multipledispatch import Dispatcher


class WeakPartial:
    """
    Like ``functools.partial(fn, arg)`` but weakly referencing ``arg``.
    """
    def __init__(self, fn, arg):
        self.fn = fn
        self.weak_arg = weakref.ref(arg)
        functools.update_wrapper(self, fn)

    def __call__(self, *args):
        arg = self.weak_arg()
        return self.fn(arg, *args)


class CachedOpMeta(type):
    """
    Metaclass for caching op instance construction.
    Caching strategy is to key on ``*args`` and retain values forever.
    """
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._instance_cache = {}

    def __call__(cls, *args, **kwargs):
        try:
            return cls._instance_cache[args]
        except KeyError:
            instance = super(CachedOpMeta, cls).__call__(*args, **kwargs)
            cls._instance_cache[args] = instance
            return instance


class WrappedOpMeta(type):
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


class Op(Dispatcher):
    _all_instances = weakref.WeakSet()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclass_registry = []

    def __init__(self, fn, *, name=None):
        if isinstance(fn, str):
            fn, name = None, fn
        if name is None:
            name = fn.__name__
        super(Op, self).__init__(name)
        if fn is not None:
            # register as default operation
            for nargs in (1, 2):
                default_signature = (object,) * nargs
                self.add(default_signature, fn)

        # Register all existing patterns.
        for supercls in reversed(inspect.getmro(type(self))):
            for pattern, fn in getattr(supercls, "_subclass_registry", ()):
                self.add(pattern, WeakPartial(fn, self))
        # Save self for registering future patterns.
        Op._all_instances.add(self)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return self.__name__

    def __repr__(self):
        return "ops." + self.__name__

    def __str__(self):
        return self.__name__

    @classmethod
    def subclass_register(cls, *pattern):
        def decorator(fn):
            # Register with all existing instances.
            for op in Op._all_instances:
                if isinstance(op, cls):
                    op.add(pattern, WeakPartial(fn, op))
            # Ensure registration with all future instances.
            cls._subclass_registry.append((pattern, fn))
            return fn
        return decorator


def make_op(fn=None, parent=None, *, name=None, module_name="funsor.ops"):
    """
    Factory to create a new :class:`Op` subclass and a new instance of that class.
    """
    # Support use as decorator.
    if fn is None:
        return lambda fn: make_op(fn, parent, name=name, module_name=module_name)

    if parent is None:
        parent = Op
    assert issubclass(parent, Op)

    if name is None:
        name = fn if isinstance(fn, str) else fn.__name__
    assert isinstance(name, str)

    classname = name.capitalize().rstrip("_") + "Op"  # e.g. add -> AddOp
    cls = type(classname, (parent,), {})
    cls.__module__ = module_name
    op = cls(fn, name=name)
    return op


def declare_op_types(locals_, all_, name_):
    op_types = set(v for v in locals_.values()
                   if isinstance(v, type) and issubclass(v, Op))
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


class UnaryOp(Op):
    pass


class BinaryOp(Op):
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
            backend = self.fn.__module__.split(">")[0]
            raise NotImplementedError(f"{self.fn} is missing shape metadata; "
                                      f"try upgrading backend {backend}")
        if len(x.shape) < self.fn.domain.event_dim:
            raise ValueError(f"Too few dimensions for input, in {self.name}")
        event_shape = x.shape[len(x.shape) - self.fn.domain.event_dim:]
        shape = self.fn.forward_shape(event_shape)
        if len(shape) > self.fn.codomain.event_dim:
            raise ValueError(f"Cannot treat transform {self.name} as an Op "
                             "because it is batched")
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
UNITS = {}                # op -> value
PRODUCT_INVERSES = {}     # op -> inverse op

__all__ = [
    'BinaryOp',
    'CachedOpMeta',
    'DISTRIBUTIVE_OPS',
    'LogAbsDetJacobianOp',
    'Op',
    'PRODUCT_INVERSES',
    'TransformOp',
    'UNITS',
    'UnaryOp',
    'WrappedTransformOp',
    'declare_op_types',
    'make_op',
]
