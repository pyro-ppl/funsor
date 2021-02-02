# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copyreg
import functools
import operator
import warnings
from functools import reduce

import funsor.ops as ops
from funsor.typing import GenericTypeMeta
from funsor.util import broadcast_shape, get_backend, get_tracing_state, quote


class Domain(GenericTypeMeta):
    pass


class ArrayType(Domain):
    """
    Base class of array-like domains.
    """

    def __getitem__(cls, dtype_shape):
        dtype, shape = dtype_shape
        assert dtype is not None
        assert shape is not None

        # in some JAX versions, shape can be np.int64 type
        if get_tracing_state() or get_backend() == "jax":
            if dtype not in (None, "real"):
                dtype = int(dtype)
            if shape is not None:
                shape = tuple(map(int, shape))

        return super().__getitem__((dtype, shape))

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, ArrayType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True

    @property
    def dtype(cls):
        return cls.__args__[0]

    @property
    def shape(cls):
        return cls.__args__[1]

    @property
    def num_elements(cls):
        return reduce(operator.mul, cls.shape, 1)

    @property
    def size(cls):
        return cls.dtype

    def __iter__(cls):
        from funsor.terms import Number

        return (Number(i, cls.size) for i in range(cls.size))


class BintType(ArrayType):
    def __getitem__(cls, size_shape):
        if isinstance(size_shape, tuple):
            size, shape = size_shape[0], size_shape[1:]
        else:
            size, shape = size_shape, ()
        return Array.__getitem__((size, shape))


class RealsType(ArrayType):
    def __getitem__(cls, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return Array.__getitem__(("real", shape))


def _pickle_array(cls):
    if cls in (Array, Bint, Reals):
        return repr(cls)
    return operator.getitem, (Array, (cls.dtype, cls.shape))


copyreg.pickle(ArrayType, _pickle_array)
copyreg.pickle(BintType, _pickle_array)
copyreg.pickle(RealsType, _pickle_array)


class Array(metaclass=ArrayType):
    """
    Generic factory for :class:`Reals` or :class:`Bint`.

        Arary["real", (3, 3)] = Reals[3, 3]
        Array["real", ()] = Real
    """

    pass


class Bint(Array, metaclass=BintType):
    """
    Factory for bounded integer types::

        Bint[5]           # integers ranging in {0,1,2,3,4}
        Bint[2, 3, 3]     # 3x3 matrices with entries in {0,1}
    """

    pass


class Reals(Array, metaclass=RealsType):
    """
    Type of a real-valued array with known shape::

        Reals[()] = Real  # scalar
        Reals[8]          # vector of length 8
        Reals[3, 3]       # 3x3 matrix
    """

    pass


Real = Reals[()]


# DEPRECATED
def reals(*args):
    warnings.warn(
        "reals(...) is deprecated, use Real or Reals[...] instead", DeprecationWarning
    )
    return Reals[args]


# DEPRECATED
def bint(size):
    warnings.warn("bint(...) is deprecated, use Bint[...] instead", DeprecationWarning)
    return Bint[size]


class ProductDomain(Domain):
    @property
    def shape(cls):
        return (len(cls.__args__),)


class Product(tuple, metaclass=ProductDomain):
    """like typing.Tuple, but works with issubclass"""

    pass


@quote.register(BintType)
@quote.register(RealsType)
def _(arg, indent, out):
    out.append((indent, repr(arg)))


@functools.singledispatch
def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    raise NotImplementedError


@find_domain.register(ops.UnaryOp)
def _find_domain_pointwise_unary_generic(op, domain):
    if isinstance(domain, ArrayType):
        return Array[domain.dtype, domain.shape]
    raise NotImplementedError


@find_domain.register(ops.LogOp)
@find_domain.register(ops.ExpOp)
def _find_domain_log_exp(op, domain):
    return Array["real", domain.shape]


@find_domain.register(ops.ReshapeOp)
def _find_domain_reshape(op, domain):
    return Array[domain.dtype, op.shape]


@find_domain.register(ops.GetitemOp)
def _find_domain_getitem(op, lhs_domain, rhs_domain):
    if isinstance(lhs_domain, ArrayType):
        dtype = lhs_domain.dtype
        shape = lhs_domain.shape[: op.offset] + lhs_domain.shape[1 + op.offset :]
        return Array[dtype, shape]
    elif isinstance(lhs_domain, ProductDomain):
        # XXX should this return a Union?
        raise NotImplementedError(
            "Cannot statically infer domain from: " f"{lhs_domain}[{rhs_domain}]"
        )


@find_domain.register(ops.BinaryOp)
def _find_domain_pointwise_binary_generic(op, lhs, rhs):
    if (
        isinstance(lhs, ArrayType)
        and isinstance(rhs, ArrayType)
        and lhs.dtype == rhs.dtype
    ):
        return Array[lhs.dtype, broadcast_shape(lhs.shape, rhs.shape)]
    raise NotImplementedError("TODO")


@find_domain.register(ops.FloordivOp)
def _find_domain_floordiv(op, lhs, rhs):
    if isinstance(lhs, ArrayType) and isinstance(rhs, ArrayType):
        shape = broadcast_shape(lhs.shape, rhs.shape)
        if isinstance(lhs.dtype, int) and isinstance(rhs.dtype, int):
            size = (lhs.size - 1) // (rhs.size - 1) + 1
            return Array[size, shape]
        if lhs.dtype == "real" and rhs.dtype == "real":
            return Reals[shape]
    raise NotImplementedError("TODO")


@find_domain.register(ops.ModOp)
def _find_domain_mod(op, lhs, rhs):
    if isinstance(lhs, ArrayType) and isinstance(rhs, ArrayType):
        shape = broadcast_shape(lhs.shape, rhs.shape)
        if isinstance(lhs.dtype, int) and isinstance(rhs.dtype, int):
            dtype = max(0, rhs.dtype - 1)
            return Array[dtype, shape]
        if lhs.dtype == "real" and rhs.dtype == "real":
            return Reals[shape]
    raise NotImplementedError("TODO")


@find_domain.register(ops.MatmulOp)
def _find_domain_matmul(op, lhs, rhs):
    assert lhs.shape and rhs.shape
    if len(rhs.shape) == 1:
        assert lhs.shape[-1] == rhs.shape[-1]
        shape = lhs.shape[:-1]
    elif len(lhs.shape) == 1:
        assert lhs.shape[-1] == rhs.shape[-2]
        shape = rhs.shape[:-2] + rhs.shape[-1:]
    else:
        assert lhs.shape[-1] == rhs.shape[-2]
        shape = broadcast_shape(lhs.shape[:-1], rhs.shape[:-2] + (1,)) + rhs.shape[-1:]
    return Reals[shape]


@find_domain.register(ops.AssociativeOp)
def _find_domain_associative_generic(op, *domains):

    assert 1 <= len(domains) <= 2

    if len(domains) == 1:
        return Array[domains[0].dtype, ()]

    lhs, rhs = domains
    if lhs.dtype == "real" or rhs.dtype == "real":
        dtype = "real"
    elif op in (ops.add, ops.mul, ops.pow, ops.max, ops.min):
        dtype = op(lhs.dtype - 1, rhs.dtype - 1) + 1
    elif op in (ops.and_, ops.or_, ops.xor):
        dtype = 2
    elif lhs.dtype == rhs.dtype:
        dtype = lhs.dtype
    else:
        raise NotImplementedError("TODO")

    shape = broadcast_shape(lhs.shape, rhs.shape)
    return Array[dtype, shape]


@find_domain.register(ops.WrappedTransformOp)
def _transform_find_domain(op, domain):
    fn = op.dispatch(object)
    shape = fn.forward_shape(domain.shape)
    return Array[domain.dtype, shape]


@find_domain.register(ops.LogAbsDetJacobianOp)
def _transform_log_abs_det_jacobian(op, domain, codomain):
    # TODO do we need to handle batch shape here?
    return Real


__all__ = [
    "Bint",
    "BintType",
    "Domain",
    "Real",
    "RealsType",
    "Reals",
    "bint",
    "find_domain",
    "reals",
]
