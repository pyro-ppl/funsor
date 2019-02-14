from __future__ import absolute_import, division, print_function

import math
from numbers import Number
from operator import add, and_, eq, ge, gt, invert, le, lt, mul, ne, neg, or_, sub, truediv, xor

import torch

_builtin_abs = abs
_builtin_max = max
_builtin_min = min
_builtin_pow = pow


def abs(x):
    return _builtin_abs(x) if isinstance(x, Number) else x.abs()


def sqrt(x):
    return math.sqrt(x) if isinstance(x, Number) else x.sqrt()


def exp(x):
    return math.exp(x) if isinstance(x, Number) else x.exp()


def log(x):
    return math.log(x) if isinstance(x, Number) else x.log()


def log1p(x):
    return math.log1p(x) if isinstance(x, Number) else x.log1p()


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
    if hasattr(x, '__max__'):
        return x.__max__(y)
    if hasattr(y, '__max__'):
        return y.__max__(x)
    if isinstance(x, torch.Tensor):
        if isinstance(y, torch.Tensor):
            return torch.max(x, y)
        return x.clamp(min=y)
    if isinstance(y, torch.Tensor):
        return y.clamp(min=x)
    return _builtin_max(x, y)


def logaddexp(x, y):
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


# just a placeholder
def marginal(x, y):
    raise ValueError


# just a placeholder
def sample(x, y):
    raise ValueError


REDUCE_OP_TO_TORCH = {
    add: torch.sum,
    mul: torch.prod,
    and_: torch.all,
    or_: torch.any,
    logaddexp: torch.logsumexp,
    min: torch.min,
    max: torch.max,
}


__all__ = [
    'REDUCE_OP_TO_TORCH',
    'abs',
    'add',
    'and_',
    'eq',
    'ge',
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
    'sample',
    'sub',
    'truediv',
    'xor',
    'sqrt',
    'exp',
    'log',
    'log1p',
]
