# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import numbers

import numpy as np

from .builtin import AssociativeOp, add, atanh, exp, log, log1p, max, min, reciprocal, safediv, safesub, sqrt, tanh
from .op import DISTRIBUTIVE_OPS, CachedOpMeta, Op, declare_op_types, make_op

_builtin_all = all
_builtin_any = any

# This is used only for pattern matching.
array = (np.ndarray, np.generic)

all = make_op(np.all)
amax = make_op(np.amax)
amin = make_op(np.amin)
any = make_op(np.any)
astype = make_op("astype")
cat = make_op("cat")
clamp = make_op("clamp")
diagonal = make_op("diagonal")
einsum = make_op("einsum")
full_like = make_op(np.full_like)
isnan = make_op(np.isnan)
prod = make_op(np.prod)
stack = make_op("stack")
sum = make_op(np.sum)
transpose = make_op("transpose")

sqrt.register(array)(np.sqrt)
exp.register(array)(np.exp)
log1p.register(array)(np.log1p)
tanh.register(array)(np.tanh)
atanh.register(array)(np.arctanh)


@log.register(array)
def _log(x):
    if x.dtype == 'bool':
        return np.where(x, 0., -math.inf)
    with np.errstate(divide='ignore'):  # skip the warning of log(0.)
        return np.log(x)


def _logaddexp(x, y):
    shift = max(detach(x), detach(y))
    return log(exp(x - shift) + exp(y - shift)) + shift


logaddexp = make_op(_logaddexp, AssociativeOp, name="logaddexp")
sample = make_op(_logaddexp, type(logaddexp), name="sample")


class ReshapeMeta(CachedOpMeta):
    def __call__(cls, shape):
        shape = tuple(shape)  # necessary to convert torch.Size to tuple
        return super().__call__(shape)


class ReshapeOp(Op, metaclass=ReshapeMeta):
    def __init__(self, shape):
        self.shape = shape
        super().__init__(self._default)

    def __reduce__(self):
        return ReshapeOp, (self.shape,)

    def _default(self, x):
        return x.reshape(self.shape)


@astype.register(array, str)
def _astype(x, dtype):
    return x.astype(dtype)


@cat.register(int, [array])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@clamp.register(array, numbers.Number, numbers.Number)
@clamp.register(array, numbers.Number, type(None))
@clamp.register(array, type(None), numbers.Number)
@clamp.register(array, type(None), type(None))
def _clamp(x, min, max):
    return np.clip(x, a_min=min, a_max=max)


@Op
def cholesky(x):
    """
    Like :func:`numpy.linalg.cholesky` but uses sqrt for scalar matrices.
    """
    if x.shape[-1] == 1:
        return np.sqrt(x)
    return np.linalg.cholesky(x)


@Op
def cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    return cholesky_solve(new_eye(x, x.shape[:-1]), x)


@Op
def cholesky_solve(x, y):
    y_inv = np.linalg.inv(y)
    A = np.swapaxes(y_inv, -2, -1) @ y_inv
    return A @ x


@Op
def detach(x):
    return x


@diagonal.register(array, int, int)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@einsum.register(str, [array])
def _einsum(x, *operand):
    return np.einsum(x, *operand)


@Op
def expand(x, shape):
    prepend_dim = len(shape) - np.ndim(x)
    assert prepend_dim >= 0
    shape = shape[:prepend_dim] + tuple(dx if size == -1 else size
                                        for dx, size in zip(np.shape(x), shape[prepend_dim:]))
    return np.broadcast_to(x, shape)


@Op
def finfo(x):
    return np.finfo(x.dtype)


@Op
def is_numeric_array(x):
    return True if isinstance(x, array) else False


@Op
def logsumexp(x, dim):
    amax = np.amax(x, axis=dim, keepdims=True)
    # treat the case x = -inf
    amax = np.where(np.isfinite(amax), amax, 0.)
    return log(np.sum(np.exp(x - amax), axis=dim)) + amax.squeeze(axis=dim)


@max.register(array, array)
def _max(x, y):
    return np.maximum(x, y)


@max.register((int, float), array)
def _max(x, y):
    return np.clip(y, a_min=x, a_max=None)


@max.register(array, (int, float))
def _max(x, y):
    return np.clip(x, a_min=y, a_max=None)


@min.register(array, array)
def _min(x, y):
    return np.minimum(x, y)


@min.register((int, float), array)
def _min(x, y):
    return np.clip(y, a_min=None, a_max=x)


@min.register(array, (int, float))
def _min(x, y):
    return np.clip(x, a_min=None, a_max=y)


@Op
def new_arange(x, stop):
    return np.arange(stop)


@new_arange.register(array, int, int, int)
def _new_arange(x, start, stop, step):
    return np.arange(start, stop, step)


@Op
def new_zeros(x, shape):
    return np.zeros(shape, dtype=x.dtype)


@Op
def new_eye(x, shape):
    n = shape[-1]
    return np.broadcast_to(np.eye(n), shape + (n,))


@Op
def permute(x, dims):
    return np.transpose(x, axes=dims)


@reciprocal.register(array)
def _reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(x.dtype).max)
    return result


@safediv.register(numbers.Number, array)
def _safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_min=None, a_max=finfo.max)


@safesub.register(numbers.Number, array)
def _safesub(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x + np.clip(-y, a_min=None, a_max=finfo.max)


@stack.register(int, [array])
def _stack(dim, *x):
    return np.stack(x, axis=dim)


@transpose.register(array, int, int)
def _transpose(x, dim1, dim2):
    return np.swapaxes(x, dim1, dim2)


@Op
def triangular_solve(x, y, upper=False, transpose=False):
    if transpose:
        y = np.swapaxes(y, -2, -1)
    return np.linalg.inv(y) @ x


@Op
def unsqueeze(x, dim):
    return np.expand_dims(x, axis=dim)


DISTRIBUTIVE_OPS.add((logaddexp, add))
DISTRIBUTIVE_OPS.add((sample, add))


__all__ = [
    'all',
    'amax',
    'amin',
    'any',
    'astype',
    'cat',
    'cholesky',
    'cholesky_inverse',
    'cholesky_solve',
    'clamp',
    'detach',
    'diagonal',
    'einsum',
    'expand',
    'finfo',
    'full_like',
    'is_numeric_array',
    'isnan',
    'logaddexp',
    'logsumexp',
    'new_arange',
    'new_eye',
    'new_zeros',
    'permute',
    'prod',
    'sample',
    'stack',
    'sum',
    'transpose',
    'triangular_solve',
    'unsqueeze',
]

declare_op_types(globals(), __all__, __name__)

__doc__ = "\n".join(".. autodata:: {}\n".format(_name)
                    for _name in __all__ if isinstance(globals()[_name], Op))
