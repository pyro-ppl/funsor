# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import operator
from numbers import Number

from .op import (
    BINARY_INVERSES,
    DISTRIBUTIVE_OPS,
    UNARY_INVERSES,
    UNITS,
    BinaryOp,
    CachedOpMeta,
    Op,
    TransformOp,
    UnaryOp,
    declare_op_types,
    make_op,
)

_builtin_abs = abs
_builtin_pow = pow
_builtin_sum = sum


def sigmoid(x):
    return 1 / (1 + exp(-x))


def softplus(x):
    return log(1.0 + exp(x))


def reciprocal(x):
    if isinstance(x, Number):
        return 1.0 / x
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
        self.__name__ = "GetitemOp({})".format(offset)

    def __reduce__(self):
        return GetitemOp, (self.offset,)

    def _default(self, x, y):
        return x[self._prefix + (y,)] if self.offset else x[y]


getitem = GetitemOp(0)
abs = make_op(_builtin_abs, UnaryOp)
eq = make_op(operator.eq, BinaryOp)
ge = make_op(operator.ge, BinaryOp)
gt = make_op(operator.gt, BinaryOp)
invert = make_op(operator.invert, UnaryOp)
le = make_op(operator.le, BinaryOp)
lt = make_op(operator.lt, BinaryOp)
ne = make_op(operator.ne, BinaryOp)
pos = make_op(operator.pos, UnaryOp)
neg = make_op(operator.neg, UnaryOp)
pow = make_op(operator.pow, BinaryOp)
sub = make_op(operator.sub, BinaryOp)
truediv = make_op(operator.truediv, BinaryOp)
floordiv = make_op(operator.floordiv, BinaryOp)
add = make_op(operator.add, AssociativeOp)
and_ = make_op(operator.and_, AssociativeOp)
mul = make_op(operator.mul, AssociativeOp)
matmul = make_op(operator.matmul, BinaryOp)
mod = make_op(operator.mod, BinaryOp)
lshift = make_op(operator.lshift, BinaryOp)
rshift = make_op(operator.rshift, BinaryOp)
not_ = make_op(operator.not_, UnaryOp)
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
log = make_op(
    lambda x: math.log(x) if x > 0 else -math.inf, parent=TransformOp, name="log"
)
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
tanh.set_inv(atanh)
atanh.set_inv(tanh)


@tanh.set_log_abs_det_jacobian
def tanh_log_abs_det_jacobian(x, y):
    return 2.0 * (math.log(2.0) - x - softplus(-2.0 * x))


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

UNITS[mul] = 1.0
UNITS[add] = 0.0

BINARY_INVERSES[mul] = truediv
BINARY_INVERSES[add] = sub

UNARY_INVERSES[mul] = reciprocal
UNARY_INVERSES[add] = neg

__all__ = [
    "AssociativeOp",
    "GetitemOp",
    "NullOp",
    "abs",
    "add",
    "and_",
    "atanh",
    "eq",
    "exp",
    "floordiv",
    "ge",
    "getitem",
    "gt",
    "invert",
    "le",
    "lgamma",
    "log",
    "log1p",
    "lshift",
    "lt",
    "matmul",
    "max",
    "min",
    "mod",
    "mul",
    "ne",
    "neg",
    "not_",
    "nullop",
    "or_",
    "pos",
    "pow",
    "reciprocal",
    "rshift",
    "safediv",
    "safesub",
    "sigmoid",
    "sqrt",
    "sub",
    "tanh",
    "truediv",
    "xor",
]

declare_op_types(globals(), __all__, __name__)

__doc__ = "\n".join(
    ".. autodata:: {}\n".format(_name)
    for _name in __all__
    if isinstance(globals()[_name], Op)
)
