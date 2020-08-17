# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copyreg
import functools
import operator
import warnings
from functools import reduce
from weakref import WeakValueDictionary

import funsor.ops as ops
from funsor.util import broadcast_shape, get_backend, get_tracing_state, quote


Domain = type


class RealsType(Domain):
    _type_cache = WeakValueDictionary()

    def __getitem__(cls, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        # in some JAX versions, shape can be np.int64 type
        if get_tracing_state() or get_backend() == "jax":
            shape = tuple(map(int, shape))

        result = RealsType._type_cache.get(shape, None)
        if result is None:
            assert cls is Reals
            assert all(isinstance(size, int) and size >= 0 for size in shape)
            name = "Reals[{}]".format(",".join(map(str, shape))) if shape else "Real"
            result = RealsType(name, (), {"shape": shape})
            RealsType._type_cache[shape] = result
        return result

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, RealsType):
            return False
        return cls is Reals or cls is subcls

    def __repr__(cls):
        return cls.__name__

    def __str__(cls):
        return cls.__name__

    @property
    def num_elements(cls):
        return reduce(operator.mul, cls.shape, 1)

    # DEPRECATED
    @property
    def dtype(self):
        warnings.warn("domain.dtype is deprecated, "
                      "use isinstance(domain, RealsType) instead",
                      DeprecationWarning)
        return "real"


@functools.partial(copyreg.pickle, RealsType)
def _pickle_real(cls):
    if cls is Reals:
        return "Reals"
    return operator.getitem, (Reals, cls.shape)


class Reals(metaclass=RealsType):
    """
    Type of a real-valued array with known shape::

        Reals[()] = Real  # scalar
        Reals[8]          # vector of length 8
        Reals[3,3]        # 3x3 matrix

    To dispatch on domain type, we recommend either ``@singledispatch``,
    ``@multipledispatch``, or ``isinstance(domain, RealsType)``.
    """


Real = Reals[()]  # just an alias


class BintType(type):
    _type_cache = WeakValueDictionary()

    def __getitem__(cls, size):
        # in some JAX versions, shape can be np.int64 type
        if get_tracing_state() or get_backend() == "jax":
            size = int(size)

        result = BintType._type_cache.get(size, None)
        if result is None:
            assert cls is Bint
            assert isinstance(size, int) and size >= 0, size
            name = "Bint[{}]".format(size)
            result = BintType(name, (), {"size": size})
            BintType._type_cache[size] = result
        return result

    def __subclasscheck__(cls, subcls):
        if not isinstance(subcls, BintType):
            return False
        return cls is Bint or cls is subcls

    def __repr__(cls):
        return cls.__name__

    def __str__(cls):
        return cls.__name__

    def __iter__(cls):
        from funsor.terms import Number
        return (Number(i, cls.size) for i in range(cls.size))

    # DEPRECATED
    @property
    def dtype(cls):
        warnings.warn("domain.dtype is deprecated, "
                      "use isinstance(domain, BintType) instead",
                      DeprecationWarning)
        return cls.size

    # DEPRECATED
    @property
    def shape(cls):
        warnings.warn("Bint[n].shape is deprecated",
                      DeprecationWarning)
        return ()

    # DEPRECATED
    @property
    def num_elements(cls):
        warnings.warn("Bint[n].num_elements is deprecated",
                      DeprecationWarning)
        return 1


@functools.partial(copyreg.pickle, BintType)
def _pickle_bint(cls):
    if cls is Bint:
        return "Bint"
    return operator.getitem, (Bint, cls.size)


class Bint(metaclass=BintType):
    """
    Factory for bounded integer types::

        Bint[5]  # integers ranging in {0,1,2,3,4}

    To dispatch on domain type, we recommend either ``@singledispatch``,
    ``@multipledispatch``, or ``isinstance(domain, BintType)``.
    """


# DEPRECATED
def reals(*args):
    warnings.warn("reals(...) is deprecated, use Reals[...] instead",
                  DeprecationWarning)
    return Reals[args]


# DEPRECATED
def bint(size):
    warnings.warn("reals(...) is deprecated, use Reals[...] instead",
                  DeprecationWarning)
    return Bint[size]


# DEPRECATED
def make_domain(shape, dtype):
    warnings.warn("make_domain is deprecated, use Bint or Reals instead",
                  DeprecationWarning)
    return Reals[shape] if dtype == "real" else Bint[dtype]


@quote.register(BintType)
@quote.register(RealsType)
def _(arg, indent, out):
    out.append((indent, repr(arg)))


def find_domain(op, *domains):
    r"""
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \*domains: One or more input domains.
    """
    assert callable(op), op
    assert all(isinstance(arg, Domain) for arg in domains)
    if len(domains) == 1:
        dtype = domains[0].dtype
        shape = domains[0].shape
        if op is ops.log or op is ops.exp:
            dtype = 'real'
        elif isinstance(op, ops.ReshapeOp):
            shape = op.shape
        elif isinstance(op, ops.AssociativeOp):
            shape = ()
        return reals(*shape) if dtype == "real" else bint(dtype)

    lhs, rhs = domains
    if isinstance(op, ops.GetitemOp):
        dtype = lhs.dtype
        shape = lhs.shape[:op.offset] + lhs.shape[1 + op.offset:]
        return reals(*shape) if dtype == "real" else bint(dtype)
    elif op == ops.matmul:
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
        return reals(*shape)

    if lhs.dtype == 'real' or rhs.dtype == 'real':
        dtype = 'real'
    elif op in (ops.add, ops.mul, ops.pow, ops.max, ops.min):
        dtype = op(lhs.dtype - 1, rhs.dtype - 1) + 1
    elif op in (ops.and_, ops.or_, ops.xor):
        dtype = 2
    elif lhs.dtype == rhs.dtype:
        dtype = lhs.dtype
    else:
        raise NotImplementedError('TODO')

    if lhs.shape == rhs.shape:
        shape = lhs.shape
    else:
        shape = broadcast_shape(lhs.shape, rhs.shape)
    return reals(*shape) if dtype == "real" else bint(dtype)


__all__ = [
    'Bint',
    'BintType',
    'Domain',
    'Real',
    'RealsType',
    'Reals',
    'bint',
    'find_domain',
    'reals',
]
