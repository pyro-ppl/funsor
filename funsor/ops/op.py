# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings

from multipledispatch import Dispatcher


class CachedOpMeta(type):
    """
    Metaclass for caching op instance construction.
    """
    def __call__(cls, *args, **kwargs):
        try:
            return cls._cache[args]
        except KeyError:
            instance = super(CachedOpMeta, cls).__call__(*args, **kwargs)
            cls._cache[args] = instance
            return instance
        except TypeError as e:
            if str(e).startswith("unhashable type"):
                warnings.warn(f"Cannot memoize Op ({e})")
                instance = super(CachedOpMeta, cls).__call__(*args, **kwargs)
                return instance
            raise e from None


class Op(Dispatcher):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._cache = {}

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
    cls = CachedOpMeta(classname, (parent,), {})
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


class TransformOp(Op):
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


class LogAbsDetJacobianOp(Op):
    pass


# TODO Memoize or use weakrefs. This currently creates reference cycles,
# delaying gc of backend_transform which may be a large expensive per-iteration
# data structure such as a normalizing flow defined by a hyper net.
def make_transform_op(backend_transform):
    """
    Factory to create a collection four new ops corresponding to a bijective
    transform.

    :returns: An ``op`` from which can be accessed ``op.inv``,
        ``op.log_abs_det_jacobian``, and ``op.inv.log_abs_det_jacobian``.
    :rtype: TransformOp
    """
    name = type(backend_transform).__name__

    # TODO Check that the op is not batched. Something like:
    # if backend_transform.batch_shape:
    #     raise ValueError("Cannot create an op from a transform "
    #                      f"{name} with nontrivial batch shape "
    #                      f"{backend_transform.batch_shape}.")

    # Create four ops.
    op = make_op(backend_transform, TransformOp, name=name)
    op_ldaj = make_op(backend_transform.log_abs_det_jacobian,
                      LogAbsDetJacobianOp,
                      name=name + "_log_abs_det_jacobian")
    inv = make_op(backend_transform.inv, TransformOp,
                  name=name + "_inv")
    inv_ldaj = make_op(backend_transform.log_abs_det_jacobian,
                       LogAbsDetJacobianOp,
                       name=name + "_inv_log_abs_det_jacobian")

    # Register relationships.
    op.set_inv(inv)
    op.set_log_abs_det_jacobian(op_ldaj)
    inv.set_inv(op)
    inv.set_log_abs_det_jacobian(inv_ldaj)

    # Register funsor conversions.
    from funsor.terms import Binary, Funsor, Unary
    op.register(Funsor)(functools.partial(Unary, op))
    inv.register(Funsor)(functools.partial(Unary, inv))
    op_ldaj.register(Funsor, Funsor)(functools.partial(Binary, op_ldaj))
    inv_ldaj.register(Funsor, Funsor)(functools.partial(Binary, inv_ldaj))

    return op


# Op registration tables.
DISTRIBUTIVE_OPS = set()  # (add, mul) pairs
UNITS = {}                # op -> value
PRODUCT_INVERSES = {}     # op -> inverse op

__all__ = [
    'CachedOpMeta',
    'DISTRIBUTIVE_OPS',
    'LogAbsDetJacobianOp',
    'Op',
    'PRODUCT_INVERSES',
    'TransformOp',
    'UNITS',
    'declare_op_types',
    'make_op',
    'make_transform_op',
]
