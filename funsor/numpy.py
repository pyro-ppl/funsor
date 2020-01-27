# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import funsor.ops as ops
from funsor.util import quote


################################################################################
# Register Ops
################################################################################

try:
    from scipy.special import expit as _sigmoid
except ImportError:
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

# take care of scalar numpy objects
array = (np.ndarray, np.generic)

ops.abs.register(array)(abs)
ops.sigmoid.register(array)(_sigmoid)
ops.sqrt.register(array)(np.sqrt)
ops.exp.register(array)(np.exp)
ops.log.register(array)(np.log)
ops.log1p.register(array)(np.log1p)
ops.min.register(array)(np.minimum)
ops.max.register(array)(np.maximum)
ops.unsqueeze.register(array, int)(np.expand_dims)
ops.expand.register(array, tuple)(np.broadcast_to)
ops.permute.register(array, (tuple, list))(np.transpose)
ops.transpose.register(array, int, int)(np.swapaxes)
ops.full_like.register(array, object)(np.full_like)
ops.clamp.register(array, object, object)(np.clip)
ops.einsum.register(str, [array])(np.einsum)


@quote.register(np.ndarray)
def _quote(x, indent, out):
    """
    Work around NumPy not supporting reproducible repr.
    """
    out.append((indent, f"np.array({repr(x.tolist())}, dtype=np.{x.dtype})"))


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
    return x + np.clip(-y, a_max=finfo)


@ops.safediv.register((int, float), array)
def _safediv(x, y):
    try:
        finfo = np.finfo(y.dtype)
    except ValueError:
        finfo = np.iinfo(y.dtype)
    return x * np.clip(np.reciprocal(y), a_max=finfo)


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
    from scipy.linalg import cho_solve

    return cho_solve((x, False), np.eye(x.shape[-1]))


@ops.triangular_solve_op.register(array, array, bool, bool)
def _triangular_solve(x, y, upper, transpose):
    from scipy.linalg import solve_triangular

    # TODO: remove this logic when using JAX
    # work around the issue of scipy which does not support batched input
    batch_shape = np.broadcast(x[..., 0, 0], y[..., 0, 0]).shape
    xs = np.broadcast_to(x, batch_shape + x.shape[-2:]).reshape((-1,) + x.shape[-2:])
    ys = np.broadcast_to(y, batch_shape + y.shape[-2:]).reshape((-1,) + y.shape[-2:])
    ans = [solve_triangular(y, x, trans=int(transpose), lower=not upper)
           for (x, y) in zip(xs, ys)]
    ans = np.stack(ans)
    return ans.reshape(batch_shape + ans.shape[-2:])


@ops.diagonal.register(array, int, int)
def _diagonal(x, dim1, dim2):
    return np.diagonal(x, axis1=dim1, axis2=dim2)


@ops.cat_op.register(int, [array])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@ops.new_zeros.register(array, tuple)
def _new_zeros(x, shape):
    return np.zeros(shape, dtype=x.dtype)


@ops.new_eye.register(array, tuple)
def _new_eye(x, shape):
    return np.broadcast_to(np.eye(shape[-1]), shape + (-1,))


@ops.new_arange.register(array, int, int, int)
def _new_arange(x, start, stop, step):
    return np.arange(start, stop, step)


@ops.new_arange.register(array, int)
def _new_arange(x, stop):
    return np.arange(stop)


@ops.finfo.register(array)
def _finfo(x):
    return np.finfo(x.dtype)
