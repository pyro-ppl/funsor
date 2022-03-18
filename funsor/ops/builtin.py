# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import operator
from numbers import Number

from .op import (
    BINARY_INVERSES,
    DISTRIBUTIVE_OPS,
    SAFE_BINARY_INVERSES,
    UNARY_INVERSES,
    UNITS,
    BinaryOp,
    Op,
    OpMeta,
    TransformOp,
    UnaryOp,
    declare_op_types,
)

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow
_builtin_sum = sum


# FIXME Most code assumes this is an AssociativeCommutativeOp.
class AssociativeOp(BinaryOp):
    pass


class ComparisonOp(BinaryOp):
    pass


@AssociativeOp.make
def null(x, y):
    """Placeholder associative op that unifies with any other op"""
    raise ValueError("should never actually evaluate this!")


@BinaryOp.make
def getitem(lhs, rhs, offset=0):
    if offset == 0:
        return lhs[rhs]
    return lhs[(slice(None),) * offset + (rhs,)]


class GetsliceMeta(OpMeta):
    """
    Works around slice objects not being hashable.
    """

    def hash_args_kwargs(cls, args, kwargs):
        index = args[0] if args else kwargs["index"]
        if not isinstance(index, tuple):
            index = (index,)
        key = tuple(
            (x.start, x.stop, x.step) if isinstance(x, slice) else x for x in index
        )
        return key


@UnaryOp.make(metaclass=GetsliceMeta)
def getslice(x, index=Ellipsis):
    return x[index]


getslice.supported_types = (type(None), type(Ellipsis), int, slice)


def parse_ellipsis(index):
    """
    Helper to split a slice into parts left and right of Ellipses.

    :param index: A tuple, or other object (None, int, slice, Funsor).
    :returns: a pair of tuples ``left, right``.
    :rtype: tuple
    """
    if not isinstance(index, tuple):
        index = (index,)
    left = []
    i = 0
    for part in index:
        i += 1
        if part is Ellipsis:
            break
        left.append(part)
    right = []
    for part in reversed(index[i:]):
        if part is Ellipsis:
            break
        right.append(part)
    right.reverse()
    return tuple(left), tuple(right)


def normalize_ellipsis(index, size):
    """
    Expand Ellipses in an index to fill the given number of dimensions.

    This should satisfy the equation::

        x[i] == x[normalize_ellipsis(i, len(x.shape))]
    """
    left, right = parse_ellipsis(index)
    if len(left) + len(right) > size:
        raise ValueError(f"Index is too wide: {index}")
    middle = (slice(None),) * (size - len(left) - len(right))
    return left + middle + right


def parse_slice(s, size):
    """
    Helper to determine nonnegative integers (start, stop, step) of a slice.

    :param slice s: A slice.
    :param int size: The size of the array being indexed into.
    :returns: A tuple of nonnegative integers ``start, stop, step``.
    :rtype: tuple
    """
    start = s.start
    if start is None:
        start = 0
    assert isinstance(start, int)
    if start >= 0:
        start = min(size, start)
    else:
        start = max(0, size + start)

    stop = s.stop
    if stop is None:
        stop = size
    assert isinstance(stop, int)
    if stop >= 0:
        stop = min(size, stop)
    else:
        stop = max(0, size + stop)

    step = s.step
    if step is None:
        step = 1
    assert isinstance(step, int)

    return start, stop, step


abs = UnaryOp.make(_builtin_abs)
eq = ComparisonOp.make(operator.eq)
ge = ComparisonOp.make(operator.ge)
gt = ComparisonOp.make(operator.gt)
invert = UnaryOp.make(operator.invert)
le = ComparisonOp.make(operator.le)
lt = ComparisonOp.make(operator.lt)
ne = ComparisonOp.make(operator.ne)
pos = UnaryOp.make(operator.pos)
neg = UnaryOp.make(operator.neg)
pow = BinaryOp.make(operator.pow)
sub = BinaryOp.make(operator.sub)
truediv = BinaryOp.make(operator.truediv)
floordiv = BinaryOp.make(operator.floordiv)
add = AssociativeOp.make(operator.add)
and_ = AssociativeOp.make(operator.and_)
mul = AssociativeOp.make(operator.mul)
matmul = BinaryOp.make(operator.matmul)
mod = BinaryOp.make(operator.mod)
lshift = BinaryOp.make(operator.lshift)
rshift = BinaryOp.make(operator.rshift)
or_ = AssociativeOp.make(operator.or_)
xor = AssociativeOp.make(operator.xor)


@AssociativeOp.make
def max(lhs, rhs):
    return _builtin_max(lhs, rhs)


@AssociativeOp.make
def min(lhs, rhs):
    return _builtin_min(lhs, rhs)


lgamma = UnaryOp.make(math.lgamma)
log1p = UnaryOp.make(math.log1p)
sqrt = UnaryOp.make(math.sqrt)


@UnaryOp.make
def reciprocal(x):
    if isinstance(x, Number):
        return 1.0 / x
    raise ValueError("No reciprocal for type {}".format(type(x)))


@UnaryOp.make
def softplus(x):
    return log(1.0 + exp(x))


@TransformOp.make
def log(x):
    return math.log(x) if x > 0 else -math.inf


exp = TransformOp.make(math.exp)
tanh = TransformOp.make(math.tanh)
atanh = TransformOp.make(math.atanh)


@TransformOp.make
def sigmoid(x):
    return 1 / (1 + exp(-x))


@sub.make
def safesub(x, y):
    if isinstance(y, Number):
        return sub(x, y)


@truediv.make
def safediv(x, y):
    if isinstance(y, Number):
        return operator.truediv(x, y)


@exp.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return x.sum()


@log.set_log_abs_det_jacobian
def log_abs_det_jacobian(x, y):
    return -y.sum()


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
DISTRIBUTIVE_OPS.add((or_, and_))

UNITS[mul] = 1.0
UNITS[add] = 0.0
UNITS[max] = -math.inf
UNITS[min] = math.inf
UNITS[and_] = False
UNITS[xor] = False
UNITS[or_] = True

BINARY_INVERSES[mul] = truediv
BINARY_INVERSES[add] = sub
BINARY_INVERSES[xor] = xor

SAFE_BINARY_INVERSES[mul] = safediv
SAFE_BINARY_INVERSES[add] = safesub

UNARY_INVERSES[mul] = reciprocal
UNARY_INVERSES[add] = neg

__all__ = [
    "AssociativeOp",
    "ComparisonOp",
    "abs",
    "add",
    "and_",
    "atanh",
    "eq",
    "exp",
    "floordiv",
    "ge",
    "getitem",
    "getslice",
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
    "null",
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
