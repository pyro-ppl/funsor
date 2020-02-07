# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as np
import numpy as onp
from jax.abstract_arrays import UnshapedArray
from jax.dtypes import canonicalize_dtype
from jax.interpreters.xla import DeviceArray
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import expit, logsumexp

import funsor.ops as ops
from funsor.util import quote


################################################################################
# Register Ops
################################################################################

# take care of scalar numpy objects
array = (onp.ndarray, onp.generic, UnshapedArray, DeviceArray)

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
    return np.log(x)


@ops.einsum.register(str, [array])
def _einsum(equation, *operands):
    return np.einsum(equation, *operands)


@quote.register(onp.ndarray)
def _quote(x, indent, out):
    """
    Work around NumPy not supporting reproducible repr.
    """
    out.append((indent, f"onp.array({repr(x.tolist())}, dtype=np.{x.dtype})"))


@quote.register(DeviceArray)
def _quote(x, indent, out):
    """
    Work around JAX DeviceArray not supporting reproducible repr.
    """
    out.append((indent, f"np.array({repr(x.copy().tolist())}, dtype=np.{x.dtype})"))


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
    return _cholesky_solve(np.eye(x.shape[-1]), x)


@ops.cholesky_solve.register(array, array)
def _cholesky_solve(x, y):
    return cho_solve((y, True), x)


@ops.triangular_solve_op.register(array, array, bool, bool)
def _triangular_solve(x, y, upper, transpose):
    return solve_triangular(y, x, trans=int(transpose), lower=not upper)


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
    return np.zeros(shape, dtype=canonicalize_dtype(x.dtype))


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
