# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copyreg
import functools
import inspect
import operator
import warnings
from functools import reduce
from weakref import WeakValueDictionary

import funsor.ops as ops
from funsor.ops.builtin import parse_ellipsis, parse_slice
from funsor.util import broadcast_shape, get_backend, get_tracing_state, quote

Domain = type


class ArrayType(Domain):
    """
    Base class of array-like domains.
    """

    _type_cache = WeakValueDictionary()

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

        assert cls.dtype in (None, dtype)
        assert cls.shape in (None, shape)
        key = dtype, shape
        result = ArrayType._type_cache.get(key, None)
        if result is None:
            if dtype == "real":
                assert all(isinstance(size, int) and size >= 0 for size in shape)
                name = (
                    "Reals[{}]".format(",".join(map(str, shape))) if shape else "Real"
                )
                result = RealsType(name, (), {"shape": shape})
            elif isinstance(dtype, int):
                assert dtype >= 0
                name = "Bint[{}]".format(",".join(map(str, (dtype,) + shape)))
                result = BintType(name, (), {"dtype": dtype, "shape": shape})
            else:
                raise ValueError("invalid dtype: {}".format(dtype))
            ArrayType._type_cache[key] = result
        return result

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, ArrayType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True

    def __repr__(cls):
        return cls.__name__

    def __str__(cls):
        return cls.__name__

    @property
    def num_elements(cls):
        return reduce(operator.mul, cls.shape, 1)

    @property
    def is_concrete(cls):
        # FIXME this should simply be isinstance(cls, Domain)
        return cls.dtype is not None and cls.shape is not None


class BintType(ArrayType):
    def __getitem__(cls, size_shape):
        if isinstance(size_shape, tuple):
            size, shape = size_shape[0], size_shape[1:]
        else:
            size, shape = size_shape, ()
        return super().__getitem__((size, shape))

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, BintType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True

    @property
    def size(cls):
        return cls.dtype

    def __iter__(cls):
        from funsor.terms import Number

        return (Number(i, cls.size) for i in range(cls.size))


class RealsType(ArrayType):
    dtype = "real"

    def __getitem__(cls, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return super().__getitem__(("real", shape))

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, RealsType):
            return False
        if cls.dtype not in (None, subcls.dtype):
            return False
        if cls.shape not in (None, subcls.shape):
            return False
        return True


def _pickle_array(cls):
    if cls in (Array, Bint, Real, Reals):
        return cls.__name__
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

    dtype = None
    shape = None


class Bint(metaclass=BintType):
    """
    Factory for bounded integer types::

        Bint[5]           # integers ranging in {0,1,2,3,4}
        Bint[2, 3, 3]     # 3x3 matrices with entries in {0,1}
    """

    dtype = None
    shape = None


class Reals(metaclass=RealsType):
    """
    Type of a real-valued array with known shape::

        Reals[()] = Real  # scalar
        Reals[8]          # vector of length 8
        Reals[3, 3]       # 3x3 matrix
    """

    shape = None


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

    _type_cache = WeakValueDictionary()

    def __getitem__(cls, arg_domains):
        try:
            return ProductDomain._type_cache[arg_domains]
        except KeyError:
            assert isinstance(arg_domains, tuple)
            assert all(isinstance(arg_domain, Domain) for arg_domain in arg_domains)
            subcls = type("Product_", (Product,), {"__args__": arg_domains})
            ProductDomain._type_cache[arg_domains] = subcls
            return subcls

    def __repr__(cls):
        return "Product[{}]".format(", ".join(map(repr, cls.__args__)))

    @property
    def __origin__(cls):
        return Product

    @property
    def shape(cls):
        return (len(cls.__args__),)


class Product(tuple, metaclass=ProductDomain):
    """like typing.Tuple, but works with issubclass"""

    __args__ = NotImplemented


class DependentMeta(type):
    def __getitem__(cls, fn):
        return cls(fn)


class Dependent(metaclass=DependentMeta):
    """
    Type hint for dependently type-decorated functions.

    Examples::

        Dependent[Real]  # a constant known domain
        Dependent[lambda x: Array[x.dtype, x.shape[1:]]  # args are Domains
        Dependent[lambda x, y: Bint[x.size + y.size]]

    :param callable fn: A lambda taking named arguments (in any order)
        which will be filled in with the domain of the similarly named
        funsor argument to the decorated function. This lambda should
        compute a desired resulting domain given domains of arguments.
    """

    def __init__(self, fn):
        function = type(lambda: None)
        self.fn = fn if isinstance(fn, function) else lambda: fn
        self.args = inspect.getfullargspec(fn)[0]

    def __call__(self, **kwargs):
        return self.fn(*map(kwargs.__getitem__, self.args))


################################################################################
# Function registration

quote.register_repr(BintType)
quote.register_repr(RealsType)


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


@find_domain.register(ops.AstypeOp)
def _find_domain_astype(op, domain):
    if op.defaults["dtype"] in ("float", "double", "float32", "float64"):
        dtype = "real"
    elif op.defaults["dtype"] in ("bool"):
        dtype = 2
    elif op.defaults["dtype"] in ("int", "int8", "int16", "int32", "int64", "uint8"):
        dtype = domain.dtype
    else:
        raise NotImplementedError
    return Array[dtype, domain.shape]


@find_domain.register(ops.LogOp)
@find_domain.register(ops.ExpOp)
def _find_domain_log_exp(op, domain):
    return Array["real", domain.shape]


@find_domain.register(ops.ReductionOp)
def _find_domain_reduction(op, domain):
    # Canonicalize dim.
    dim = op.defaults.get("dim", None)
    ndims = len(domain.shape)
    if dim is None:
        dims = set(range(ndims))
    elif isinstance(dim, int):
        dims = {dim % ndims}
    else:
        dims = {i % ndims for i in dim}

    # Compute shape.
    if op.defaults.get("keepdims", False):
        shape = tuple(1 if i in dims else domain.shape[i] for i in range(ndims))
    else:
        shape = tuple(domain.shape[i] for i in range(ndims) if i not in dims)

    # Compute domain.
    if op.name in ("all", "any"):
        dtype = 2
    elif domain.dtype == "real":
        dtype = "real"
    else:
        raise NotImplementedError("TODO")

    return Array[dtype, shape]


@find_domain.register(ops.ReshapeOp)
def _find_domain_reshape(op, domain):
    return Array[domain.dtype, op.defaults["shape"]]


@find_domain.register(ops.GetitemOp)
def _find_domain_getitem(op, lhs_domain, rhs_domain):
    if isinstance(lhs_domain, ArrayType):
        offset = op.defaults["offset"]
        dtype = lhs_domain.dtype
        shape = lhs_domain.shape[:offset] + lhs_domain.shape[1 + offset :]
        return Array[dtype, shape]
    elif isinstance(lhs_domain, ProductDomain):
        # XXX should this return a Union?
        raise NotImplementedError(
            "Cannot statically infer domain from: " f"{lhs_domain}[{rhs_domain}]"
        )


@find_domain.register(ops.GetsliceOp)
def _find_domain_getslice(op, domain):
    index = op.defaults["index"]
    if isinstance(domain, ArrayType):
        dtype = domain.dtype
        shape = list(domain.shape)
        left, right = parse_ellipsis(index)

        i = 0
        for part in left:
            if part is None:
                shape.insert(i, 1)
                i += 1
            elif isinstance(part, int):
                del shape[i]
            elif isinstance(part, slice):
                start, stop, step = parse_slice(part, shape[i])
                shape[i] = max(0, (stop - start + step - 1) // step)
                i += 1
            else:
                raise ValueError(part)

        i = -1
        for part in reversed(right):
            if part is None:
                shape.insert(len(shape) + i + 1, 1)
                i -= 1
            elif isinstance(part, int):
                del shape[i]
            elif isinstance(part, slice):
                start, stop, step = parse_slice(part, shape[i])
                shape[i] = max(0, (stop - start + step - 1) // step)
                i -= 1
            else:
                raise ValueError(part)

        return Array[dtype, tuple(shape)]

    if isinstance(domain, ProductDomain):
        if isinstance(index, tuple):
            assert len(index) == 1
            index = index[0]
        if isinstance(index, int):
            return domain.__args__[index]
        elif isinstance(index, slice):
            return Product[domain.__args__[index]]
        else:
            raise ValueError(index)

    raise NotImplementedError("TODO")


@find_domain.register(ops.BinaryOp)
def _find_domain_pointwise_binary_generic(op, lhs, rhs):
    if (
        isinstance(lhs, ArrayType)
        and isinstance(rhs, ArrayType)
        and lhs.dtype == rhs.dtype
    ):
        return Array[lhs.dtype, broadcast_shape(lhs.shape, rhs.shape)]
    raise NotImplementedError("TODO")


@find_domain.register(ops.ComparisonOp)
def _find_domain_comparison(op, lhs, rhs):
    if isinstance(lhs, ArrayType) and isinstance(rhs, ArrayType):
        return Array[2, broadcast_shape(lhs.shape, rhs.shape)]
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
    fn = op.defaults["fn"]
    shape = fn.forward_shape(domain.shape)
    return Array[domain.dtype, shape]


@find_domain.register(ops.LogAbsDetJacobianOp)
def _transform_log_abs_det_jacobian(op, domain, codomain):
    # TODO do we need to handle batch shape here?
    return Real


@find_domain.register(ops.StackOp)
def _find_domain_stack(op, parts):
    shape = broadcast_shape(*(x.shape for x in parts))
    dim = op.defaults["dim"]
    if dim >= 0:
        dim = dim - len(shape) - 1
    assert dim < 0
    split = dim + len(shape) + 1
    shape = shape[:split] + (len(parts),) + shape[split:]
    output = Array[parts[0].dtype, shape]
    return output


@find_domain.register(ops.CatOp)
def _find_domain_cat(op, parts):
    dim = op.defaults["axis"]
    if dim >= 0:
        event_dims = {len(x.shape) for x in parts}
        assert len(event_dims) == 1, "undefined"
        dim = dim - next(iter(event_dims))
    assert dim < 0
    shape = broadcast_shape(*(x.shape[:dim] for x in parts))
    shape += (sum(x.shape[dim] for x in parts),)
    if dim < -1:
        shape += broadcast_shape(*(x.shape[dim + 1 :] for x in parts))
    output = Array[parts[0].dtype, shape]
    return output


@find_domain.register(ops.EinsumOp)
def _find_domain_einsum(op, operands):
    equation = op.defaults["equation"]
    ein_inputs, ein_output = equation.split("->")
    ein_inputs = ein_inputs.split(",")
    size_dict = {}
    for ein_input, x in zip(ein_inputs, operands):
        assert x.dtype == "real"
        assert len(ein_input) == len(x.shape)
        for name, size in zip(ein_input, x.shape):
            other_size = size_dict.setdefault(name, size)
            if other_size != size:
                raise ValueError(
                    "Size mismatch at {}: {} vs {}".format(name, size, other_size)
                )
    return Reals[tuple(size_dict[d] for d in ein_output)]


__all__ = [
    "Bint",
    "BintType",
    "Dependent",
    "Domain",
    "Real",
    "Reals",
    "RealsType",
    "bint",
    "find_domain",
    "reals",
]
