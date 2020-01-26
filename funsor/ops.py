# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import operator
from numbers import Number

import numpy as np
from multipledispatch import Dispatcher

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow


class Op(Dispatcher):
    def __init__(self, fn):
        super(Op, self).__init__(fn.__name__)
        # register as default operation
        for nargs in (1, 2):
            default_signature = (object,) * nargs
            self.add(default_signature, fn)

    def __repr__(self):
        return "ops." + self.__name__

    def __str__(self):
        return self.__name__


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


# FIXME Most code assumes this is an AssociativeCommutativeOp.
class AssociativeOp(Op):
    pass


class AddOp(AssociativeOp):
    pass


class MulOp(AssociativeOp):
    pass


class MatmulOp(Op):  # Associtive but not commutative.
    pass


class LogAddExpOp(AssociativeOp):
    pass


class SubOp(Op):
    pass


class NegOp(Op):
    pass


class DivOp(Op):
    pass


class NullOp(AssociativeOp):
    """Placeholder associative op that unifies with any other op"""
    pass


@NullOp
def nullop(x, y):
    raise ValueError("should never actually evaluate this!")


class ReshapeMeta(type):
    _cache = {}

    def __call__(cls, shape):
        shape = tuple(shape)
        try:
            return ReshapeMeta._cache[shape]
        except KeyError:
            instance = super().__call__(shape)
            ReshapeMeta._cache[shape] = instance
            return instance


class ReshapeOp(Op, metaclass=ReshapeMeta):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(self._default)

    def _default(self, x):
        return x.reshape(self.shape)


class GetitemMeta(type):
    _cache = {}

    def __call__(cls, offset):
        try:
            return GetitemMeta._cache[offset]
        except KeyError:
            instance = super(GetitemMeta, cls).__call__(offset)
            GetitemMeta._cache[offset] = instance
            return instance


class GetitemOp(Op, metaclass=GetitemMeta):
    """
    Op encoding an index into one dimension, e.g. ``x[:,:,y]`` for offset of 2.
    """
    def __init__(self, offset):
        assert isinstance(offset, int)
        assert offset >= 0
        self.offset = offset
        self._prefix = (slice(None),) * offset
        super(GetitemOp, self).__init__(self._default)
        self.__name__ = 'GetitemOp({})'.format(offset)

    def _default(self, x, y):
        return x[self._prefix + (y,)] if self.offset else x[y]


getitem = GetitemOp(0)

eq = Op(operator.eq)
ge = Op(operator.ge)
gt = Op(operator.gt)
invert = Op(operator.invert)
le = Op(operator.le)
lt = Op(operator.lt)
ne = Op(operator.ne)
neg = NegOp(operator.neg)
sub = SubOp(operator.sub)
truediv = DivOp(operator.truediv)

add = AddOp(operator.add)
and_ = AssociativeOp(operator.and_)
mul = MulOp(operator.mul)
matmul = MatmulOp(operator.matmul)
or_ = AssociativeOp(operator.or_)
xor = AssociativeOp(operator.xor)


@add.register(object)
def _unary_add(x):
    return x.sum()


@Op
def abs(x):
    return x.abs()


@abs.register(Number)
def _abs(x):
    return _builtin_abs(x)


@Op
def sqrt(x):
    return np.sqrt(x)


class ExpOp(TransformOp):
    pass


@ExpOp
def exp(x):
    return np.exp(x)


@exp.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return add(x)


class LogOp(TransformOp):
    pass


@LogOp
def log(x):
    return np.log(x)


@log.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return -add(y)


exp.set_inv(log)
log.set_inv(exp)


@Op
def log1p(x):
    return np.log1p(x)


@Op
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@Op
def pow(x, y):
    return x ** y


@AssociativeOp
def min(x, y):
    if hasattr(x, '__min__'):
        return x.__min__(y)
    if hasattr(y, '__min__'):
        return y.__min__(x)
    return _builtin_min(x, y)


@AssociativeOp
def max(x, y):
    if hasattr(x, '__max__'):
        return x.__max__(y)
    if hasattr(y, '__max__'):
        return y.__max__(x)
    return _builtin_max(x, y)


@LogAddExpOp
def logaddexp(x, y):
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


@SubOp
def safesub(x, y):
    if isinstance(y, Number):
        return sub(x, y)


@DivOp
def safediv(x, y):
    if isinstance(y, Number):
        return truediv(x, y)


class ReciprocalOp(Op):
    pass


@ReciprocalOp
def reciprocal(x):
    if isinstance(x, Number):
        return 1. / x
    raise ValueError("No reciprocal for type {}".format(type(x)))


DISTRIBUTIVE_OPS = frozenset([
    (logaddexp, add),
    (add, mul),
    (max, mul),
    (min, mul),
    (max, add),
    (min, add),
])


UNITS = {
    mul: 1.,
    add: 0.,
}


PRODUCT_INVERSES = {
    mul: safediv,
    add: safesub,
}


# Numeric Array Ops

@Op
def sum(x, dim):
    raise NotImplementedError


@Op
def prod(x, dim):
    raise NotImplementedError


@Op
def all(x, dim):
    raise NotImplementedError


@Op
def any(x, dim):
    raise NotImplementedError


@Op
def logsumexp(x, dim):
    raise NotImplementedError


@Op
def min_(x, dim):
    raise NotImplementedError


@Op
def max_(x, dim):
    raise NotImplementedError


@Op
def cholesky(x):
    raise NotImplementedError


@Op
def cholesky_inverse(x):
    raise NotImplementedError


@Op
def triangular_solve_op(x, y, upper, transpose):
    raise NotImplementedError


def triangular_solve(x, y, upper=False, transpose=False):
    return triangular_solve_op(x, y, upper, transpose)


@Op
def cat_op(dim, *x):
    raise NotImplementedError


def cat(x, dim=0):
    return cat_op(dim, *x)


@Op
def new_zeros(x, shape):
    raise NotImplementedError


@Op
def new_eye(x, shape):
    raise NotImplementedError


@Op
def new_arange(x, start, stop, step):
    raise NotImplementedError


@Op
def full_like(x, shape):
    raise NotImplementedError


@Op
def unsqueeze(x, dim):
    raise NotImplementedError


@Op
def expand(x, dim):
    raise NotImplementedError


@Op
def diagonal(x, dim1, dim2):
    raise NotImplementedError


@Op
def transpose(x, dim0, dim1):
    raise NotImplementedError


@Op
def permute(x, dims):
    raise NotImplementedError


__all__ = [
    'AddOp',
    'AssociativeOp',
    'DISTRIBUTIVE_OPS',
    'ExpOp',
    'GetitemOp',
    'LogAddExpOp',
    'LogOp',
    'NegOp',
    'Op',
    'PRODUCT_INVERSES',
    'ReciprocalOp',
    'SubOp',
    'ReshapeOp',
    'UNITS',
    'abs',
    'add',
    'and_',
    'cat',
    'cholesky',
    'cholesky_inverse',
    'diagonal',
    'eq',
    'exp',
    'expand',
    'full_like',
    'ge',
    'getitem',
    'gt',
    'invert',
    'le',
    'log',
    'log1p',
    'lt',
    'matmul',
    'max',
    'min',
    'mul',
    'ne',
    'neg',
    'new_arange',
    'new_eye',
    'new_zeros',
    'or_',
    'pow',
    'safediv',
    'safesub',
    'sigmoid',
    'sqrt',
    'sub',
    'transpose',
    'triangular_solve',
    'truediv',
    'unsqueeze',
    'xor',
]

__doc__ = """
Built-in operations
-------------------

{}

Operation classes
-----------------
""".format("\n".join(f".. autodata:: {_name}\n"
                     for _name in __all__ if isinstance(globals()[_name], Op)))
