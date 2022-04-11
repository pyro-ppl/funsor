# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numbers
import typing

import jax.numpy as np
import jax.random
import numpy as onp
from jax import lax
from jax.core import Tracer
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import expit, gammaln, logsumexp

from .. import ops

################################################################################
# Register Ops
################################################################################

array = (onp.generic, onp.ndarray, np.ndarray, Tracer)
ops.atanh.register(array)(np.arctanh)
ops.clamp.register(array)(np.clip)
ops.exp.register(array)(np.exp)
ops.log1p.register(array)(np.log1p)
ops.max.register(array, array)(np.maximum)
ops.min.register(array, array)(np.minimum)
ops.permute.register(array)(np.transpose)
ops.sigmoid.register(array)(expit)
ops.sqrt.register(array)(np.sqrt)
ops.tanh.register(array)(np.tanh)
ops.transpose.register(array)(np.swapaxes)
ops.flip.register(array)(np.flip)
ops.unsqueeze.register(array)(np.expand_dims)
ops.qr.register(array)(np.linalg.qr)


###########################################
# Reduction Ops
###########################################


@ops.all.register(array)
def _all(x, axis, keepdims):
    return np.all(x, axis, keepdims=keepdims)


@ops.any.register(array)
def _any(x, axis, keepdims):
    return np.any(x, axis, keepdims=keepdims)


@ops.amax.register(array)
def _amax(x, axis, keepdims):
    return np.amax(x, axis, keepdims=keepdims)


@ops.amin.register(array)
def _amin(x, axis, keepdims):
    return np.amax(x, axis, keepdims=keepdims)


@ops.sum.register(array)
def _sum(x, axis, keepdims):
    return np.sum(x, axis, keepdims=keepdims)


@ops.prod.register(array)
def _prod(x, axis, keepdims):
    return np.prod(x, axis, keepdims=keepdims)


@ops.logsumexp.register(array)
def _logsumexp(x, axis, keepdims):
    return logsumexp(x, axis, keepdims=keepdims)


@ops.mean.register(array)
def _mean(x, axis, keepdims):
    return np.mean(x, axis, keepdims=keepdims)


@ops.std.register(array)
def _std(x, axis, ddof, keepdims):
    return np.std(x, axis, ddof=ddof, keepdims=keepdims)


@ops.var.register(array)
def _var(x, axis, ddof, keepdims):
    return np.var(x, axis, ddof=ddof, keepdims=keepdims)


###########################################


@ops.argmax.register(array)
def _argmax(x, axis, keepdims):
    if keepdims:
        return np.expand_dims(np.argmax(x, axis), axis)
    return np.argmax(x, axis)


@ops.argmin.register(array)
def _argmin(x, axis, keepdims):
    if keepdims:
        return np.expand_dims(np.argmin(x, axis), axis)
    return np.argmin(x, axis)


@ops.astype.register(array)
def _astype(x, dtype):
    return x.astype(np.result_type(dtype))


ops.cat.register(typing.Tuple[typing.Union[array], ...])(np.concatenate)


@ops.cholesky.register(array)
def _cholesky(x):
    """
    Like :func:`numpy.linalg.cholesky` but uses sqrt for scalar matrices.
    """
    if x.shape[-1] == 1:
        return np.sqrt(x)
    return np.linalg.cholesky(x)


@ops.cholesky_inverse.register(array)
def _cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    return _cholesky_solve(_new_eye(x, x.shape[:-1]), x)


@ops.cholesky_solve.register(array, array)
def _cholesky_solve(x, y):
    return cho_solve((y, True), x)


@ops.detach.register(array)
def _detach(x):
    return lax.stop_gradient(x)


@ops.diagonal.register(array)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@ops.einsum.register(typing.Tuple[typing.Union[array], ...])
def _einsum(operands, equation):
    return np.einsum(equation, *operands)


@ops.expand.register(array)
def _expand(x, shape):
    prepend_dim = len(shape) - np.ndim(x)
    assert prepend_dim >= 0
    shape = shape[:prepend_dim] + tuple(
        dx if size == -1 else size for dx, size in zip(np.shape(x), shape[prepend_dim:])
    )
    return np.broadcast_to(x, shape)


@ops.finfo.register(array)
def _finfo(x):
    return np.finfo(np.result_type(x))


for typ in array:

    @ops.is_numeric_array.register(typ)
    def _is_numeric_array(x):
        return True


@ops.isnan.register(array)
def _isnan(x):
    return np.isnan(x)


@ops.lgamma.register(array)
def _lgamma(x):
    return gammaln(x)


@ops.log.register(array)
def _log(x):
    return np.log(x)


@ops.logaddexp.register(array, array)
def _safe_logaddexp_tensor_tensor(x, y):
    finfo = np.finfo(np.result_type(x))
    shift = np.clip(ops.max(ops.detach(x), ops.detach(y)), a_max=None, a_min=finfo.min)
    return np.log(np.exp(x - shift) + np.exp(y - shift)) + shift


@ops.logaddexp.register(numbers.Number, array)
def _safe_logaddexp_number_tensor(x, y):
    finfo = np.finfo(np.result_type(y))
    shift = np.clip(ops.detach(y), a_max=None, a_min=max(x, finfo.min))
    return np.log(np.exp(x - shift) + np.exp(y - shift)) + shift


@ops.logaddexp.register(array, numbers.Number)
def _safe_logaddexp_tensor_number(x, y):
    return _safe_logaddexp_number_tensor(y, x)


ops.max.register(array, array)(np.maximum)
ops.min.register(array, array)(np.minimum)


@ops.max.register((int, float), array)
def _max(x, y):
    return np.clip(y, a_min=x, a_max=None)


@ops.max.register(array, (int, float))
def _max(x, y):
    return np.clip(x, a_min=y, a_max=None)


# TODO: replace (int, float) by numbers.Number
@ops.min.register((int, float), array)
def _min(x, y):
    return np.clip(y, a_min=None, a_max=x)


@ops.min.register(array, (int, float))
def _min(x, y):
    return np.clip(x, a_min=None, a_max=y)


@ops.new_full.register(array)
def _new_full(x, shape, value):
    return np.full(shape, value, dtype=np.result_type(x))


@ops.new_arange.register(array)
def _new_arange(x, start, stop, step):
    if step is not None:
        return np.arange(start, stop, step)
    if stop is not None:
        return np.arange(start, stop)
    return np.arange(start)


@ops.new_eye.register(array)
def _new_eye(x, shape):
    n = shape[-1]
    return np.broadcast_to(np.eye(n), shape + (n,))


@ops.new_zeros.register(array)
def _new_zeros(x, shape):
    return onp.zeros(shape, dtype=np.result_type(x))


@ops.randn.register(array)
def _randn(prototype, shape, rng_key=None):
    assert isinstance(shape, tuple)
    return jax.random.normal(rng_key, shape, dtype=prototype.dtype)


@ops.reciprocal.register(array)
def _reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(np.result_type(x)).max)
    return result


@ops.safediv.register(array, array)
@ops.safediv.register((int, float), array)
def _safediv(x, y):
    try:
        finfo = np.finfo(np.result_type(y))
    except ValueError:
        finfo = np.iinfo(np.result_type(y))
    return x * np.clip(np.reciprocal(y), a_min=None, a_max=finfo.max)


@ops.safesub.register(array, array)
@ops.safesub.register((int, float), array)
def _safesub(x, y):
    try:
        finfo = np.finfo(np.result_type(y))
    except ValueError:
        finfo = np.iinfo(np.result_type(y))
    return x + np.clip(-y, a_min=None, a_max=finfo.max)


@ops.scatter.register(array, tuple, array)
def _scatter(dest, indices, src):
    return np.asarray(dest).at[indices].set(src)


@ops.stack.register(typing.Tuple[typing.Union[array + (int, float)], ...])
def _stack(parts, dim=0):
    return np.stack(parts, axis=dim)


@ops.triangular_solve.register(array, array)
def _triangular_solve(x, y, upper=False, transpose=False):
    assert np.ndim(x) >= 2 and np.ndim(y) >= 2
    n, m = x.shape[-2:]
    assert y.shape[-2:] == (n, n)
    # NB: JAX requires x and y have the same batch_shape
    batch_shape = lax.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x = np.broadcast_to(x, batch_shape + (n, m))
    if y.shape[:-2] == batch_shape:
        return solve_triangular(y, x, trans=int(transpose), lower=not upper)

    # The following procedure handles the case: y.shape = (i, 1, n, n), x.shape = (..., i, j, n, m)
    # because we don't want to broadcast y to the shape (i, j, n, n).
    # We are going to make x have shape (..., 1, j,  i, 1, n) to apply batched triangular_solve
    dx = x.ndim
    prepend_ndim = dx - y.ndim  # ndim of ... part
    # Reshape x with the shape (..., 1, i, j, 1, n, m)
    x_new_shape = batch_shape[:prepend_ndim]
    for (sy, sx) in zip(y.shape[:-2], batch_shape[prepend_ndim:]):
        x_new_shape += (sx // sy, sy)
    x_new_shape += (n, m)
    x = np.reshape(x, x_new_shape)
    # Permute y to make it have shape (..., 1, j, m, i, 1, n)
    batch_ndim = x.ndim - 2
    permute_dims = (
        tuple(range(prepend_ndim))
        + tuple(range(prepend_ndim, batch_ndim, 2))
        + (batch_ndim + 1,)
        + tuple(range(prepend_ndim + 1, batch_ndim, 2))
        + (batch_ndim,)
    )
    x = np.transpose(x, permute_dims)
    x_permute_shape = x.shape

    # reshape to (-1, i, 1, n)
    x = np.reshape(x, (-1,) + y.shape[:-1])
    # permute to (i, 1, n, -1)
    x = np.moveaxis(x, 0, -1)

    sol = solve_triangular(
        y, x, trans=int(transpose), lower=not upper
    )  # shape: (i, 1, n, -1)
    sol = np.moveaxis(sol, -1, 0)  # shape: (-1, i, 1, n)
    sol = np.reshape(sol, x_permute_shape)  # shape: (..., 1, j, m, i, 1, n)

    # now we permute back to x_new_shape = (..., 1, i, j, 1, n, m)
    permute_inv_dims = tuple(range(prepend_ndim))
    for i in range(y.ndim - 2):
        permute_inv_dims += (prepend_ndim + i, dx + i - 1)
    permute_inv_dims += (sol.ndim - 1, prepend_ndim + y.ndim - 2)
    sol = np.transpose(sol, permute_inv_dims)
    return sol.reshape(batch_shape + (n, m))


@ops.triangular_inv.register(array)
def _triangular_inv(x, upper=False):
    if x.shape[-1] == 1:
        return 1 / x
    eye = _new_eye(x, x.shape[:-1])
    return _triangular_solve(eye, x, upper=upper)
