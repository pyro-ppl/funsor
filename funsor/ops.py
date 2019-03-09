from __future__ import absolute_import, division, print_function

from funsor.six import singledispatch
from numbers import Number
from operator import add, and_, eq, ge, getitem, gt, invert, le, lt, mul, ne, neg, or_, sub, truediv, xor

import numpy as np
import torch

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow


def abs(x):
    return np.abs(x)


def sqrt(x):
    return np.sqrt(x)


def exp(x):
    return np.exp(x)


def log(x):
    return np.log(x)


def log1p(x):
    return np.log1p(x)


def pow(x, y):
    result = x ** y
    # work around shape bug https://github.com/pytorch/pytorch/issues/16685
    if isinstance(x, Number) and isinstance(y, torch.Tensor):
        result = result.reshape(y.shape)
    return result


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


def max(x, y):
    return np.max(x, y)


def logaddexp(x, y):
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


# just a placeholder
def marginal(x, y):
    raise ValueError


# just a placeholder
def sample(x, y):
    raise ValueError


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


ASSOCIATIVE_OPS = frozenset([
    add,
    mul,
    logaddexp,
    and_,
    or_,
    min,
    max,
])


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
    'ASSOCIATIVE_OPS',
    'DISTRIBUTIVE_OPS',
    'PRODUCT_INVERSES',
    'REDUCE_OP_TO_TORCH',
    'abs',
    'add',
    'and_',
    'eq',
    'ge',
    'getitem',
    'gt',
    'invert',
    'le',
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
    'sub',
    'truediv',
    'xor',
    'sqrt',
    'exp',
    'log',
    'log1p',
]
