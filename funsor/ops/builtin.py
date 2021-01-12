# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import operator
from numbers import Number

from .op import DISTRIBUTIVE_OPS, PRODUCT_INVERSES, UNITS, Op, OpCacheMeta, TransformOp

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow
_builtin_sum = sum


# FIXME Most code assumes this is an AssociativeCommutativeOp.
class AssociativeOp(Op):
    pass


class AddOp(AssociativeOp):
    pass


class MulOp(AssociativeOp):
    pass


class MatmulOp(Op):  # Associtive but not commutative.
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


class GetitemOp(Op, metaclass=OpCacheMeta):
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
abs = Op(_builtin_abs)
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
def sqrt(x):
    return math.sqrt(x)


class ExpOp(TransformOp):
    pass


@ExpOp
def exp(x):
    return math.exp(x)


@exp.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return add(x)


class LogOp(TransformOp):
    pass


@LogOp
def log(x):
    return math.log(x) if x > 0 else -math.inf


@log.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return -add(y)


exp.set_inv(log)
log.set_inv(exp)


class TanhOp(TransformOp):
    pass


@TanhOp
def tanh(x):
    return math.tanh(x)


@tanh.set_inv
def tanh_inv(y):
    return atanh(y)


@tanh.set_log_abs_det_jacobian
def tanh_log_abs_det_jacobian(x, y):
    return 2. * (math.log(2.) - x - softplus(-2. * x))


class AtanhOp(TransformOp):
    pass


@AtanhOp
def atanh(x):
    return math.atanh(x)


@atanh.set_inv
def atanh_inv(y):
    return tanh(y)


@atanh.set_log_abs_det_jacobian
def atanh_log_abs_det_jacobian(x, y):
    return -tanh.log_abs_det_jacobian(y, x)


@Op
def log1p(x):
    return math.log1p(x)


class SigmoidOp(TransformOp):
    pass


@SigmoidOp
def sigmoid(x):
    return 1 / (1 + exp(-x))


@sigmoid.set_inv
def sigmoid_inv(y):
    return log(y) - log1p(-y)


@sigmoid.set_log_abs_det_jacobian
def sigmoid_log_abs_det_jacobian(x, y):
    return -softplus(-x) - softplus(x)


@Op
def pow(x, y):
    return x ** y


@Op
def softplus(x):
    return log(1. + exp(x))


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


@Op
def lgamma(x):
    return math.lgamma(x)


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
