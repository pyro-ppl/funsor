# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as onp

import funsor.ops as ops
from funsor.util import quote

try:
    import jax  # noqa:401
except ModuleNotFoundError:
    try:
        import scipy  # noqa:401
    except ModuleNotFoundError:
        def expit(x):
            return 1 / (1 + onp.exp(-x))

        def logsumexp(x, axis=None):
            amax = onp.amax(x, axis=axis, keepdims=True)
            return onp.log(onp.sum(onp.exp(x - amax), axis=axis)) + amax.squeeze(axis=axis)

        def cho_solve(c_and_lower, b):
            c, lower = c_and_lower
            if lower:
                A = c @ onp.swapaxes(c, -2, -1)
            else:
                A = onp.swapaxes(c, -2, -1) @ c
            return onp.linalg.solve(A, b)

        def solve_triangular(a, b, trans=0, lower=False):
            if trans:
                a = onp.swapaxes(a, -2, -1)
            return onp.linalg.solve(a, b)
    else:
        from scipy.special import expit, logsumexp
        from scipy.linalg import cho_solve, solve_triangular

    np = onp
    unhashable_array = (onp.ndarray,)
    array = (onp.ndarray, onp.generic)
    USING_JAX = False

    def canonicalize_dtype(x):
        return x
else:
    import jax.numpy as np
    from jax import lax
    from jax.abstract_arrays import UnshapedArray
    from jax.dtypes import canonicalize_dtype
    from jax.interpreters.xla import DeviceArray
    from jax.scipy.linalg import cho_solve, solve_triangular
    from jax.scipy.special import expit, logsumexp

    unhashable_array = (onp.ndarray, DeviceArray)
    array = (onp.ndarray, onp.generic, UnshapedArray, DeviceArray)
    USING_JAX = True


__all__ = ["array", "canonicalize_dtype", "unhashable_array", "USING_JAX"]


################################################################################
# Register Ops
################################################################################

ops.abs.register(array)(abs)
ops.sigmoid.register(array)(expit)
ops.sqrt.register(array)(np.sqrt)
ops.exp.register(array)(np.exp)
ops.log1p.register(array)(np.log1p)
ops.min.register(array)(np.minimum)
ops.max.register(array)(np.maximum)
ops.unsqueeze.register(array, int)(np.expand_dims)
ops.expand.register(array, tuple)(np.broadcast_to)
ops.permute.register(array, (tuple, list))(np.transpose)
ops.transpose.register(array, int, int)(np.swapaxes)
ops.full_like.register(array, object)(np.full_like)
ops.clamp.register(array, object, object)(np.clip)


@ops.log.register(array)
def _log(x):
    if not USING_JAX:
        if isinstance(x, bool) or (isinstance(x, onp.ndarray) and x.dtype == 'bool'):
            # we cast to np.float64 because np.log(True) returns a np.float16 array
            return np.log(x, dtype=np.float64)
    return np.log(x)


@ops.einsum.register(str, [array])
def _einsum(equation, *operands):
    return np.einsum(equation, *operands)


for arr in unhashable_array:
    @quote.register(arr)
    def _quote(x, indent, out):
        """
        Work around NumPy not supporting reproducible repr.
        """
        out.append((indent, f"np.array({repr(onp.asarray(x).tolist())}, dtype=np.{x.dtype})"))


@ops.min.register(array, array)
def _min(x, y):
    return np.minimum(x, y)


# TODO: replace (int, float) by object
@ops.min.register((int, float), array)
def _min(x, y):
    return np.clip(y, a_min=None, a_max=x)


@ops.min.register(array, (int, float))
def _min(x, y):
    return np.clip(x, a_min=None, a_max=y)


@ops.max.register(array, array)
def _max(x, y):
    return np.maximum(x, y)


@ops.max.register((int, float), array)
def _max(x, y):
    return np.clip(y, a_min=x, a_max=None)


@ops.max.register(array, (int, float))
def _max(x, y):
    return np.clip(x, a_min=y, a_max=None)


@ops.reciprocal.register(array)
def _reciprocal(x):
    result = np.clip(np.reciprocal(x), a_max=np.finfo(x.dtype).max)
    return result


@ops.safesub.register((int, float), array)
def _safesub(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x + np.clip(-y, a_min=None, a_max=finfo.max)


@ops.safediv.register((int, float), array)
def _safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_min=None, a_max=finfo.max)


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
    if not USING_JAX:
        batch_shape = np.broadcast(x[..., 0, 0], y[..., 0, 0]).shape
        xs = np.broadcast_to(x, batch_shape + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
        ys = np.broadcast_to(y, batch_shape + y.shape[-2:]).reshape((-1,) + y.shape[-2:])
        ans = [cho_solve((y, True), x) for (x, y) in zip(xs, ys)]
        ans = np.stack(ans)
        return ans.reshape(batch_shape + ans.shape[-2:])

    return cho_solve((y, True), x)


@ops.triangular_solve_op.register(array, array, bool, bool)
def _triangular_solve(x, y, upper, transpose):
    if not USING_JAX:
        batch_shape = np.broadcast(x[..., 0, 0], y[..., 0, 0]).shape
        xs = np.broadcast_to(x, batch_shape + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
        ys = np.broadcast_to(y, batch_shape + y.shape[-2:]).reshape((-1,) + y.shape[-2:])
        ans = [solve_triangular(y, x, trans=int(transpose), lower=not upper)
               for (x, y) in zip(xs, ys)]
        ans = np.stack(ans)
        return ans.reshape(batch_shape + ans.shape[-2:])

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
    x_new_shape += (n, m,)
    x = np.reshape(x, x_new_shape)
    # Permute y to make it have shape (..., 1, j, m, i, 1, n)
    batch_ndim = x.ndim - 2
    permute_dims = (tuple(range(prepend_ndim))
                    + tuple(range(prepend_ndim, batch_ndim, 2))
                    + (batch_ndim + 1,)
                    + tuple(range(prepend_ndim + 1, batch_ndim, 2))
                    + (batch_ndim,))
    x = np.transpose(x, permute_dims)
    x_permute_shape = x.shape

    # reshape to (-1, i, 1, n)
    x = np.reshape(x, (-1,) + y.shape[:-1])
    # permute to (i, 1, n, -1)
    x = np.moveaxis(x, 0, -1)

    sol = solve_triangular(y, x, trans=int(transpose), lower=not upper)  # shape: (i, 1, n, -1)
    sol = np.moveaxis(sol, -1, 0)  # shape: (-1, i, 1, n)
    sol = np.reshape(sol, x_permute_shape)  # shape: (..., 1, j, m, i, 1, n)

    # now we permute back to x_new_shape = (..., 1, i, j, 1, n, m)
    permute_inv_dims = tuple(range(prepend_ndim))
    for i in range(y.ndim - 2):
        permute_inv_dims += (prepend_ndim + i, dx + i - 1)
    permute_inv_dims += (sol.ndim - 1, prepend_ndim + y.ndim - 2)
    sol = np.transpose(sol, permute_inv_dims)
    return sol.reshape(batch_shape + (n, m))


@ops.diagonal.register(array, int, int)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@ops.cat.register(int, [array])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@ops.stack.register(int, [array])
def _stack(dim, *x):
    return np.stack(x, axis=dim)


@ops.new_zeros.register(array, tuple)
def _new_zeros(x, shape):
    return onp.zeros(shape, dtype=x.dtype)


@ops.new_eye.register(array, tuple)
def _new_eye(x, shape):
    n = shape[-1]
    return np.broadcast_to(np.eye(n), shape + (n,))


@ops.new_arange.register(array, int, int, int)
def _new_arange(x, start, stop, step):
    return np.arange(start, stop, step)


@ops.new_arange.register(array, int)
def _new_arange(x, stop):
    return np.arange(stop)


@ops.finfo.register(array)
def _finfo(x):
    return np.finfo(x.dtype)


@ops.sum.register(array, (int, type(None)))
def _sum(x, dim):
    return np.sum(x, axis=dim)


@ops.prod.register(array, (int, type(None)))
def _prod(x, dim):
    return np.prod(x, axis=dim)


@ops.all.register(array, (int, type(None)))
def _all(x, dim):
    return np.all(x, axis=dim)


@ops.any.register(array, (int, type(None)))
def _any(x, dim):
    return np.any(x, axis=dim)


@ops.logsumexp.register(array, (int, type(None)))
def _logsumexp(x, dim):
    return logsumexp(x, axis=dim)


@ops.amin.register(array, (int, type(None)))
def _amin(x, dim):
    return np.amin(x, axis=dim)


@ops.amax.register(array, (int, type(None)))
def _amax(x, dim):
    return np.amax(x, axis=dim)
