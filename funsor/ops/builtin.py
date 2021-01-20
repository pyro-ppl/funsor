# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import operator
from numbers import Number

from .op import (
    DISTRIBUTIVE_OPS,
    PRODUCT_INVERSES,
    UNITS,
    CachedOpMeta,
    Op,
    TransformOp,
    UnaryOp,
    declare_op_types,
    make_op
)

_builtin_abs = abs
_builtin_pow = pow
_builtin_sum = sum


def sigmoid(x):
    return 1 / (1 + exp(-x))


def softplus(x):
    return log(1. + exp(x))


def reciprocal(x):
    if isinstance(x, Number):
        return 1. / x
    raise ValueError("No reciprocal for type {}".format(type(x)))


# FIXME Most code assumes this is an AssociativeCommutativeOp.
class AssociativeOp(Op):
    pass


class NullOp(AssociativeOp):
    """Placeholder associative op that unifies with any other op"""
    pass


@NullOp
def nullop(x, y):
    raise ValueError("should never actually evaluate this!")


class GetitemOp(Op, metaclass=CachedOpMeta):
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

    def __reduce__(self):
        return GetitemOp, (self.offset,)

    def _default(self, x, y):
        return x[self._prefix + (y,)] if self.offset else x[y]


getitem = GetitemOp(0)
abs = make_op(_builtin_abs, UnaryOp)
eq = make_op(operator.eq, Op)
ge = make_op(operator.ge, Op)
gt = make_op(operator.gt, Op)
invert = make_op(operator.invert, UnaryOp)
le = make_op(operator.le, Op)
lt = make_op(operator.lt, Op)
ne = make_op(operator.ne, Op)
neg = make_op(operator.neg, Op)
pow = make_op(operator.pow, Op)
sub = make_op(operator.sub, Op)
truediv = make_op(operator.truediv, Op)

add = make_op(operator.add, AssociativeOp)
and_ = make_op(operator.and_, AssociativeOp)
mul = make_op(operator.mul, AssociativeOp)
matmul = make_op(operator.matmul, Op)
or_ = make_op(operator.or_, AssociativeOp)
xor = make_op(operator.xor, AssociativeOp)
max = make_op(max, AssociativeOp)
min = make_op(min, AssociativeOp)

lgamma = make_op(math.lgamma, UnaryOp)
log1p = make_op(math.log1p, UnaryOp)
sqrt = make_op(math.sqrt, UnaryOp)

reciprocal = make_op(reciprocal, UnaryOp)
softplus = make_op(softplus, UnaryOp)

exp = make_op(math.exp, TransformOp)
log = make_op(lambda x: math.log(x) if x > 0 else -math.inf,
              parent=TransformOp, name="log")
tanh = make_op(math.tanh, TransformOp)
atanh = make_op(math.atanh, TransformOp)
sigmoid = make_op(sigmoid, TransformOp)


@make_op(parent=type(sub))
def safesub(x, y):
    if isinstance(y, Number):
        return sub(x, y)


@make_op(parent=type(truediv))
def safediv(x, y):
    if isinstance(y, Number):
        return operator.truediv(x, y)


@add.register(object)
def _unary_add(x):
    return x.sum()


@exp.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return add(x)


@log.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return -add(y)


exp.set_inv(log)
log.set_inv(exp)


@tanh.set_inv
def tanh_inv(y):
    return atanh(y)


@tanh.set_log_abs_det_jacobian
def tanh_log_abs_det_jacobian(x, y):
    return 2. * (math.log(2.) - x - softplus(-2. * x))


@atanh.set_inv
def atanh_inv(y):
    return tanh(y)


@atanh.set_log_abs_det_jacobian
def atanh_log_abs_det_jacobian(x, y):
    return -tanh.log_abs_det_jacobian(y, x)


@sigmoid.set_inv
def sigmoid_inv(y):
    return log(y) - log1p(-y)


@sigmoid.set_log_abs_det_jacobian
def sigmoid_log_abs_det_jacobian(x, y):
    return -softplus(-x) - softplus(x)


DISTRIBUTIVE_OPS.add((add, mul))
DISTRIBUTIVE_OPS.add((max, mul))
DISTRIBUTIVE_OPS.add((min, mul))
DISTRIBUTIVE_OPS.add((max, add))
DISTRIBUTIVE_OPS.add((min, add))

UNITS[mul] = 1.
UNITS[add] = 0.

PRODUCT_INVERSES[mul] = truediv
PRODUCT_INVERSES[add] = sub

__all__ = [
    'abs',
    'add',
    'and_',
    'atanh',
    'eq',
    'exp',
    'ge',
    'getitem',
    'gt',
    'invert',
    'le',
    'lgamma',
    'log',
    'log1p',
    'lt',
    'matmul',
    'max',
    'min',
    'mul',
    'ne',
    'neg',
    'nullop',
    'or_',
    'pow',
    'reciprocal',
    'safediv',
    'safesub',
    'sigmoid',
    'sqrt',
    'sub',
    'tanh',
    'truediv',
    'xor',
]

declare_op_types(globals(), __all__, __name__)

__doc__ = "\n".join(".. autodata:: {}\n".format(_name)
                    for _name in __all__ if isinstance(globals()[_name], Op))
