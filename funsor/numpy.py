# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import funsor.ops as ops
from funsor.util import quote


def einsum(equation, *operands):
    """
    Log-sum-exp implementation of einsum.
    """
    # rename symbols to support PyTorch 0.4.1 and earlier,
    # which allow only symbols a-z.
    symbols = sorted(set(equation) - set(',->'))
    rename = dict(zip(symbols, 'abcdefghijklmnopqrstuvwxyz'))
    equation = ''.join(rename.get(s, s) for s in equation)

    inputs, output = equation.split('->')
    if inputs == output:
        return operands[0][...]  # create a new object
    inputs = inputs.split(',')

    shifts = []
    exp_operands = []
    for dims, operand in zip(inputs, operands):
        shift = operand
        for i, dim in enumerate(dims):
            if dim not in output:
                shift = np.max(shift, i, keepdims=True)
        # avoid nan due to -inf - -inf
        shift = np.clip(shift, a_min=np.finfo(shift.dtype).min, a_max=None)
        exp_operands.append(np.exp(operand - shift))

        # permute shift to match output
        shift = shift.reshape([size for size, dim in zip(operand.shape, dims) if dim in output])
        if shift.ndim:
            shift = shift.reshape((1,) * (len(output) - shift.ndim) + shift.shape)
            dims = [dim for dim in dims if dim in output]
            dims = [dim for dim in output if dim not in dims] + dims
            shift = np.transpose(shift, [dims.index(dim) for dim in output])
        shifts.append(shift)

    result = np.log(np.einsum(equation, *exp_operands))
    return sum(shifts + [result])


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


@ops.einsum.register(str, [array])
def _einsum(equation, *operands):
    return np.einsum(equation, *operands)


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


@ops.cholesky_solve.register(array, array)
def _cholesky_solve(x, y):
    from scipy.linalg import cho_solve

    return cho_solve((y, False), x)


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


@ops.cat.register(int, [array])
def _cat(dim, *x):
    return np.concatenate(x, axis=dim)


@ops.stack.register(int, [array])
def _stack(dim, *x):
    return np.stack(x, axis=dim)


@ops.new_zeros.register(array, tuple)
def _new_zeros(x, shape):
    return np.zeros(shape, dtype=x.dtype)


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
    from scipy.special import logsumexp

    return logsumexp(x, axis=dim)


@ops.amin.register(array, (int, type(None)))
def _amin(x, dim):
    return np.amin(x, axis=dim)


@ops.amax.register(array, (int, type(None)))
def _amax(x, dim):
    return np.amax(x, axis=dim)
