# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import numbers
import typing
from functools import singledispatch

import numpy as np

from .builtin import (
    AssociativeOp,
    add,
    atanh,
    exp,
    log,
    log1p,
    max,
    min,
    reciprocal,
    safediv,
    safesub,
    sqrt,
    tanh,
)
from .op import (
    DISTRIBUTIVE_OPS,
    UNITS,
    BinaryOp,
    FinitaryOp,
    Op,
    OpMeta,
    TernaryOp,
    UnaryOp,
    declare_op_types,
)

_builtin_all = all
_builtin_any = any

# This is used only for pattern matching.
array = (np.ndarray, np.generic)
arraylist = typing.Tuple[typing.Union[array], ...]

sqrt.register(array)(np.sqrt)
exp.register(array)(np.exp)
log1p.register(array)(np.log1p)
tanh.register(array)(np.tanh)
atanh.register(array)(np.arctanh)


@UnaryOp.make
def all(x, dim=None):
    return np.all(x, dim)


@UnaryOp.make
def any(x, dim=None):
    return np.any(x, dim)


@UnaryOp.make
def amax(x, dim=None, keepdims=False):
    return np.amax(x, dim, keepdims=keepdims)


@UnaryOp.make
def amin(x, dim=None, keepdims=False):
    return np.amax(x, dim, keepdims=keepdims)


@UnaryOp.make
def sum(x, dim=None, keepdims=False):
    return np.sum(x, dim, keepdims=keepdims)


@UnaryOp.make
def prod(x, dim=None):
    return np.prod(x, dim)


@UnaryOp.make
def isnan(x):
    return np.isnan(x)


@UnaryOp.make
def full_like(prototype, fill_value):
    return np.full_like(prototype, fill_value)


@log.register(array)
def _log(x):
    if x.dtype == "bool":
        return np.where(x, 0.0, -math.inf)
    with np.errstate(divide="ignore"):  # skip the warning of log(0.)
        return np.log(x)


@AssociativeOp.make
def logaddexp(x, y):
    shift = max(detach(x), detach(y))
    return log(exp(x - shift) + exp(y - shift)) + shift


sample = logaddexp.make(logaddexp.default, name="sample")


class ReshapeMeta(OpMeta):
    def hash_args_kwargs(cls, args, kwargs):
        assert not kwargs
        if args:
            (shape,) = args
            shape = tuple(shape)  # necessary to convert torch.Size to tuple
            args = (shape,)
        return super().hash_args_kwargs(args, kwargs)


@UnaryOp.make(metaclass=ReshapeMeta)
def reshape(x, shape):
    return x.reshape(shape)


@UnaryOp.make
def astype(x, dtype):
    raise NotImplementedError


@astype.register(array)
def _astype(x, dtype):
    return x.astype(dtype)


@FinitaryOp.make
def cat(parts, axis):
    raise NotImplementedError


cat.register(arraylist)(np.concatenate)


@UnaryOp.make
def clamp(x, min=None, max=None):
    return min(max(x, min), max)


clamp.register(array)(np.clip)


@UnaryOp.make
def cholesky(x):
    """
    Like :func:`numpy.linalg.cholesky` but uses sqrt for scalar matrices.
    """
    if x.shape[-1] == 1:
        return np.sqrt(x)
    return np.linalg.cholesky(x)


@UnaryOp.make
def cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    return cholesky_solve(new_eye(x, x.shape[:-1]), x)


@BinaryOp.make
def cholesky_solve(x, y):
    y_inv = np.linalg.inv(y)
    A = np.swapaxes(y_inv, -2, -1) @ y_inv
    return A @ x


@UnaryOp.make
def detach(x):
    return x


@UnaryOp.make
def diagonal(x, dim1, dim2):
    raise NotImplementedError


@diagonal.register(array)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@FinitaryOp.make
def einsum(operands, equation):
    raise NotImplementedError


@einsum.register(arraylist)
def _einsum(operands, equation):
    return np.einsum(equation, *operands)


@UnaryOp.make
def expand(x, shape):
    prepend_dim = len(shape) - np.ndim(x)
    assert prepend_dim >= 0
    shape = shape[:prepend_dim] + tuple(
        dx if size == -1 else size for dx, size in zip(np.shape(x), shape[prepend_dim:])
    )
    return np.broadcast_to(x, shape)


@UnaryOp.make
def finfo(x):
    return np.finfo(x.dtype)


# this isn't really a mathematical op
@singledispatch
def is_numeric_array(x):
    """
    Returns whether an object is a ground numeric array.
    """
    return False


for typ in array:

    @is_numeric_array.register(typ)
    def _is_numeric_array(x):
        return True


@logaddexp.register(array, array)
def _safe_logaddexp_tensor_tensor(x, y):
    finfo = np.finfo(x.dtype)
    shift = np.clip(max(detach(x), detach(y)), a_max=None, a_min=finfo.min)
    return np.log(np.exp(x - shift) + np.exp(y - shift)) + shift


@logaddexp.register(numbers.Number, array)
def _safe_logaddexp_number_tensor(x, y):
    finfo = np.finfo(y.dtype)
    shift = np.clip(detach(y), a_max=None, a_min=max(x, finfo.min))
    return np.log(np.exp(x - shift) + np.exp(y - shift)) + shift


@logaddexp.register(array, numbers.Number)
def _safe_logaddexp_tensor_number(x, y):
    return _safe_logaddexp_number_tensor(y, x)


@UnaryOp.make
def logsumexp(x, dim):
    amax = np.amax(x, axis=dim, keepdims=True)
    # treat the case x = -inf
    amax = np.where(np.isfinite(amax), amax, 0.0)
    return log(np.sum(np.exp(x - amax), axis=dim)) + amax.squeeze(axis=dim)


max.register(array, array)(np.maximum)
min.register(array, array)(np.minimum)


@max.register((int, float), array)
def _max(x, y):
    return np.clip(y, a_min=x, a_max=None)


@max.register(array, (int, float))
def _max(x, y):
    return np.clip(x, a_min=y, a_max=None)


@min.register((int, float), array)
def _min(x, y):
    return np.clip(y, a_min=None, a_max=x)


@min.register(array, (int, float))
def _min(x, y):
    return np.clip(x, a_min=None, a_max=y)


@UnaryOp.make
def argmax(x, dim):
    raise NotImplementedError


@argmax.register(array)
def _argmax(x, dim):
    return np.argmax(x, dim)


@UnaryOp.make
def new_arange(x, start=None, stop=None, step=None):
    raise NotImplementedError


@new_arange.register(array)
def _new_arange(x, start, stop, step):
    if step is not None:
        return np.arange(start, stop, step)
    if stop is not None:
        return np.arange(start, stop)
    return np.arange(start)


@UnaryOp.make
def new_zeros(x, shape):
    return np.zeros(shape, dtype=x.dtype)


@UnaryOp.make
def new_full(x, shape, value):
    return np.full(shape, value, dtype=x.dtype)


@UnaryOp.make
def new_eye(x, shape):
    n = shape[-1]
    return np.broadcast_to(np.eye(n), shape + (n,))


@UnaryOp.make
def permute(x, dims):
    return np.transpose(x, axes=dims)


@reciprocal.register(array)
def _reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(x.dtype).max)
    return result


@safediv.register(array, array)
@safediv.register(numbers.Number, array)
def _safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_min=None, a_max=finfo.max)


@safesub.register(array, array)
@safesub.register(numbers.Number, array)
def _safesub(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x + np.clip(-y, a_min=None, a_max=finfo.max)


@TernaryOp.make
def scatter(destin, indices, source):
    raise NotImplementedError


@scatter.register(array, tuple, array)
def _scatter(destin, indices, source):
    result = destin.copy()
    result[indices] = source
    return result


@TernaryOp.make
def scatter_add(destin, indices, source):
    raise NotImplementedError


@scatter_add.register(array, tuple, array)
def _scatter_add(destin, indices, source):
    result = destin.copy()
    np.add.at(result, indices, source)
    return result


@FinitaryOp.make
def stack(parts, dim=0):
    raise NotImplementedError


stack.register(arraylist)(np.stack)


@UnaryOp.make
def transpose(array, axis1, axis2):
    raise NotImplementedError


transpose.register(array)(np.swapaxes)


@BinaryOp.make
def triangular_solve(x, y, upper=False, transpose=False):
    if transpose:
        y = np.swapaxes(y, -2, -1)
    return np.linalg.inv(y) @ x


@UnaryOp.make
def unsqueeze(x, dim):
    return np.expand_dims(x, axis=dim)


DISTRIBUTIVE_OPS.add((logaddexp, add))
DISTRIBUTIVE_OPS.add((sample, add))

UNITS[logaddexp] = -math.inf

__all__ = [
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "astype",
    "cat",
    "cholesky",
    "cholesky_inverse",
    "cholesky_solve",
    "clamp",
    "detach",
    "diagonal",
    "einsum",
    "expand",
    "finfo",
    "full_like",
    "is_numeric_array",
    "isnan",
    "logaddexp",
    "logsumexp",
    "new_arange",
    "new_eye",
    "new_full",
    "new_zeros",
    "permute",
    "prod",
    "sample",
    "scatter",
    "scatter_add",
    "stack",
    "sum",
    "transpose",
    "triangular_solve",
    "unsqueeze",
]

declare_op_types(globals(), __all__, __name__)

__doc__ = "\n".join(
    ".. autodata:: {}\n".format(_name)
    for _name in __all__
    if isinstance(globals()[_name], Op)
)
