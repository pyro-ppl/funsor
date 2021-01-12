# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import operator
from numbers import Number

from .op import DISTRIBUTIVE_OPS, PRODUCT_INVERSES, UNITS, Op, CachedOpMeta, TransformOp, make_op_and_type

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow
_builtin_sum = sum


def sigmoid(x):
    return 1 / (1 + exp(-x))


def softplus(x):
    return log(1. + exp(x))


def min(x, y):
    if hasattr(x, '__min__'):
        return x.__min__(y)
    if hasattr(y, '__min__'):
        return y.__min__(x)
    return _builtin_min(x, y)


def max(x, y):
    if hasattr(x, '__max__'):
        return x.__max__(y)
    if hasattr(y, '__max__'):
        return y.__max__(x)
    return _builtin_max(x, y)


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
abs, AbsOp = make_op_and_type(_builtin_abs, Op)
eq, EqOp = make_op_and_type(operator.eq, Op)
ge, GeOp = make_op_and_type(operator.ge, Op)
gt, GtOp = make_op_and_type(operator.gt, Op)
invert, InvertOp = make_op_and_type(operator.invert, Op)
le, LeOp = make_op_and_type(operator.le, Op)
lt, LtOp = make_op_and_type(operator.lt, Op)
ne, NeOp = make_op_and_type(operator.ne, Op)
neg, NegOp = make_op_and_type(operator.neg, Op)
pow, PowOp = make_op_and_type(operator.pow, Op)
sub, SubOp = make_op_and_type(operator.sub, Op)
truediv, DivOp = make_op_and_type(operator.truediv, DivOp)

add, AddOp = make_op_and_type(operator.add, AssociativeOp)
and_, AndOp = make_op_and_type(operator.and_, AssociativeOp)
mul, MulOp = make_op_and_type(operator.mul, AssociativeOp)
matmul, MatmulOp = make_op_and_type(operator.matmul, Op)
or_, OrOp = make_op_and_type(operator.or_, AssociativeOp)
xor, XorOp = make_op_and_type(operator.xor, AssociativeOp)

log1p, Log1pOp = make_op_and_type(math.log1p, Op)
sqrt, SqrtOp = make_op_and_type(math.sqrt, Op)
exp, ExpOp = make_op_and_type(math.exp, TransformOp)
tanh, TanhOp = make_op_and_type(math.tanh, TransformOp)
atanh, AtanhOp = make_op_and_type(math.atanh, TransformOp)
log, LogOp = make_op_and_type(lambda x: math.log(x) if x > 0 else -math.inf,
                              parent=TransformOp, name="log")

sigmoid, SigmoidOp = make_op_and_type(sigmoid, TransformOp)
softplus, SoftplusOp = make_op_and_type(softplus, Op)
max, MaxOp = make_op_and_type(max, AssociativeOp)
min, MinOp = make_op_and_type(min, AssociativeOp)
reciprocal, ReciprocalOp = make_op_and_type(reciprocal, Op)
lgamma, LgammaOp = make_op_and_type(math.lgamma, Op)


@SubOp
def safesub(x, y):
    if isinstance(y, Number):
        return sub(x, y)


@DivOp
def safediv(x, y):
    if isinstance(y, Number):
        return truediv(x, y)


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

PRODUCT_INVERSES[mul] = safediv
PRODUCT_INVERSES[add] = safesub

__all__ = [
    'AddOp',
    'AssociativeOp',
    'AtanhOp',
    'DivOp',
    'ExpOp',
    'GetitemOp',
    'LogOp',
    'MatmulOp',
    'MulOp',
    'NegOp',
    'NullOp',
    'ReciprocalOp',
    'SigmoidOp',
    'SubOp',
    'TanhOp',
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

__doc__ = "\n".join(".. autodata:: {}\n".format(_name)
                    for _name in __all__ if isinstance(globals()[_name], Op))
