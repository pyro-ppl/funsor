from __future__ import absolute_import, division, print_function

import functools
from collections import namedtuple
from numbers import Number
import operator

import numpy as np
import torch

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow


class Op(namedtuple('Op', ['fn'])):
    def __new__(cls, fn):
        result = super(Op, cls).__new__(cls, fn)
        functools.update_wrapper(result, fn)
        return result

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return self.__name__


class AssociativeOp(Op):
    pass


eq = Op(operator.eq)
ge = Op(operator.ge)
getitem = Op(operator.getitem)
gt = Op(operator.gt)
invert = Op(operator.invert)
le = Op(operator.le)
lt = Op(operator.lt)
ne = Op(operator.ne)
neg = Op(operator.neg)
sub = Op(operator.sub)
truediv = Op(operator.truediv)

add = AssociativeOp(operator.add)
and_ = AssociativeOp(operator.and_)
mul = AssociativeOp(operator.mul)
or_ = AssociativeOp(operator.or_)
xor = AssociativeOp(operator.xor)


@Op
def abs(x):
    return np.abs(x)


@Op
def sqrt(x):
    return np.sqrt(x)


@Op
def exp(x):
    return np.exp(x)


@Op
def log(x):
    return np.log(x)


@Op
def log1p(x):
    return np.log1p(x)


@Op
def pow(x, y):
    result = x ** y
    # work around shape bug https://github.com/pytorch/pytorch/issues/16685
    if isinstance(x, Number) and isinstance(y, torch.Tensor):
        result = result.reshape(y.shape)
    return result


@AssociativeOp
def min(x, y):
    if hasattr(x, '__min__'):
        return x.__min__(y)
    if hasattr(y, '__min__'):
        return y.__min__(x)
    if isinstance(x, torch.Tensor):
        if isinstance(y, torch.Tensor):
            return torch.min(x, y)
        return x.clamp(max=y)
    if isinstance(y, torch.Tensor):
        return y.clamp(max=x)
    return _builtin_min(x, y)


@AssociativeOp
def max(x, y):
    return np.max(x, y)


@AssociativeOp
def logaddexp(x, y):
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


# just a placeholder
@Op
def marginal(x, y):
    raise ValueError


# just a placeholder
@Op
def sample(x, y):
    raise ValueError


@Op
def reciprocal(x):
    if isinstance(x, Number):
        return 1. / x
    if isinstance(x, torch.Tensor):
        result = x.reciprocal()
        result.clamp_(max=torch.finfo(result.dtype).max)
        return result
    raise ValueError("No reciprocal for type {}".format(type(x)))


REDUCE_OP_TO_TORCH = {
    add: torch.sum,
    mul: torch.prod,
    and_: torch.all,
    or_: torch.any,
    logaddexp: torch.logsumexp,
    min: torch.min,
    max: torch.max,
}


DISTRIBUTIVE_OPS = frozenset([
    (logaddexp, add),
    (add, mul),
    (max, mul),
    (min, mul),
    (max, add),
    (min, add),
])


PRODUCT_INVERSES = {
    mul: reciprocal,
    add: neg,
}


__all__ = [
    'AssociativeOp',
    'DISTRIBUTIVE_OPS',
    'Op',
    'PRODUCT_INVERSES',
    'REDUCE_OP_TO_TORCH',
    'abs',
    'add',
    'and_',
    'eq',
    'exp',
    'ge',
    'getitem',
    'gt',
    'invert',
    'le',
    'log',
    'log1p',
    'lt',
    'marginal',
    'max',
    'min',
    'mul',
    'ne',
    'neg',
    'or_',
    'pow',
    'reciprocal',
    'sample',
    'sqrt',
    'sub',
    'truediv',
    'xor',
]
