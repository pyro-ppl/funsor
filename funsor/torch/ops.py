# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numbers

import torch

import funsor.ops as ops

################################################################################
# Register Ops
################################################################################

ops.abs.register(torch.Tensor)(torch.abs)
ops.atanh.register(torch.Tensor)(torch.atanh)
ops.cholesky_solve.register(torch.Tensor, torch.Tensor)(torch.cholesky_solve)
ops.clamp.register(torch.Tensor, numbers.Number, numbers.Number)(torch.clamp)
ops.clamp.register(torch.Tensor, numbers.Number, type(None))(torch.clamp)
ops.clamp.register(torch.Tensor, type(None), numbers.Number)(torch.clamp)
ops.clamp.register(torch.Tensor, type(None), type(None))(torch.clamp)
ops.exp.register(torch.Tensor)(torch.exp)
ops.full_like.register(torch.Tensor, numbers.Number)(torch.full_like)
ops.log1p.register(torch.Tensor)(torch.log1p)
ops.sigmoid.register(torch.Tensor)(torch.sigmoid)
ops.sqrt.register(torch.Tensor)(torch.sqrt)
ops.tanh.register(torch.Tensor)(torch.tanh)
ops.transpose.register(torch.Tensor, int, int)(torch.transpose)
ops.unsqueeze.register(torch.Tensor, int)(torch.unsqueeze)


@ops.all.register(torch.Tensor, (int, type(None)))
def _all(x, dim):
    return x.all() if dim is None else x.all(dim=dim)


@ops.amax.register(torch.Tensor, (int, type(None)))
def _amax(x, dim, keepdims=False):
    return x.max() if dim is None else x.max(dim, keepdims)[0]


@ops.amin.register(torch.Tensor, (int, type(None)))
def _amin(x, dim, keepdims=False):
    return x.min() if dim is None else x.min(dim, keepdims)[0]


@ops.any.register(torch.Tensor, (int, type(None)))
def _any(x, dim):
    return x.any() if dim is None else x.any(dim=dim)


@ops.astype.register(torch.Tensor, str)
def _astype(x, dtype):
    return x.type(getattr(torch, dtype))


@ops.cat.register(int, [torch.Tensor])
def _cat(dim, *x):
    if len(x) == 1:
        return x[0]
    return torch.cat(x, dim=dim)


@ops.cholesky.register(torch.Tensor)
def _cholesky(x):
    """
    Like :func:`torch.cholesky` but uses sqrt for scalar matrices.
    Works around https://github.com/pytorch/pytorch/issues/24403 often.
    """
    if x.size(-1) == 1:
        return x.sqrt()
    return x.cholesky()


@ops.cholesky_inverse.register(torch.Tensor)
def _cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    if x.dim() == 2:
        return x.cholesky_inverse()
    return torch.eye(x.size(-1)).cholesky_solve(x)


@ops.detach.register(torch.Tensor)
def _detach(x):
    return x.detach()


@ops.diagonal.register(torch.Tensor, int, int)
def _diagonal(x, dim1, dim2):
    return x.diagonal(dim1=dim1, dim2=dim2)


@ops.einsum.register(str, [torch.Tensor])
def _einsum(equation, *operands):
    return torch.einsum(equation, *operands)


@ops.expand.register(torch.Tensor, tuple)
def _expand(x, shape):
    return x.expand(shape)


@ops.finfo.register(torch.Tensor)
def _finfo(x):
    return torch.finfo(x.dtype)


@ops.is_numeric_array.register(torch.Tensor)
def _is_numeric_array(x):
    return True


@ops.isnan.register(torch.Tensor)
def _isnan(x):
    return torch.isnan(x)


@ops.lgamma.register(torch.Tensor)
def _lgamma(x):
    return x.lgamma()


@ops.log.register(torch.Tensor)
def _log(x):
    if x.dtype in (torch.bool, torch.uint8, torch.long):
        x = x.to(dtype=torch.get_default_dtype())
    return x.log()


@ops.logsumexp.register(torch.Tensor, (int, type(None)))
def _logsumexp(x, dim):
    return x.reshape(-1).logsumexp(0) if dim is None else x.logsumexp(dim)


@ops.max.register(torch.Tensor, torch.Tensor)
def _max(x, y):
    return torch.max(x, y)


@ops.max.register(numbers.Number, torch.Tensor)
def _max(x, y):
    return y.clamp(min=x)


@ops.max.register(torch.Tensor, numbers.Number)
def _max(x, y):
    return x.clamp(min=y)


@ops.min.register(torch.Tensor, torch.Tensor)
def _min(x, y):
    return torch.min(x, y)


@ops.min.register(numbers.Number, torch.Tensor)
def _min(x, y):
    return y.clamp(max=x)


@ops.min.register(torch.Tensor, numbers.Number)
def _min(x, y):
    return x.clamp(max=y)


@ops.new_arange.register(torch.Tensor, int, int, int)
def _new_arange(x, start, stop, step):
    return torch.arange(start, stop, step)


@ops.new_arange.register(torch.Tensor, (int, torch.Tensor))
def _new_arange(x, stop):
    return torch.arange(stop)


@ops.new_eye.register(torch.Tensor, tuple)
def _new_eye(x, shape):
    return torch.eye(shape[-1]).expand(shape + (-1,))


@ops.new_zeros.register(torch.Tensor, tuple)
def _new_zeros(x, shape):
    return x.new_zeros(shape)


@ops.permute.register(torch.Tensor, (tuple, list))
def _permute(x, dims):
    return x.permute(dims)


@ops.pow.register(numbers.Number, torch.Tensor)
def _pow(x, y):
    result = x ** y
    # work around shape bug https://github.com/pytorch/pytorch/issues/16685
    return result.reshape(y.shape)


@ops.pow.register(torch.Tensor, numbers.Number)
@ops.pow.register(torch.Tensor, torch.Tensor)
def _pow(x, y):
    return x ** y


@ops.prod.register(torch.Tensor, (int, type(None)))
def _prod(x, dim):
    return x.prod() if dim is None else x.prod(dim=dim)


@ops.reciprocal.register(torch.Tensor)
def _reciprocal(x):
    result = x.reciprocal().clamp(max=torch.finfo(x.dtype).max)
    return result


@ops.safediv.register(numbers.Number, torch.Tensor)
def _safediv(x, y):
    try:
        finfo = torch.finfo(y.dtype)
    except TypeError:
        finfo = torch.iinfo(y.dtype)
    return x * y.reciprocal().clamp(max=finfo.max)


@ops.safesub.register(numbers.Number, torch.Tensor)
def _safesub(x, y):
    try:
        finfo = torch.finfo(y.dtype)
    except TypeError:
        finfo = torch.iinfo(y.dtype)
    return x + (-y).clamp(max=finfo.max)


@ops.stack.register(int, [torch.Tensor])
def _stack(dim, *x):
    return torch.stack(x, dim=dim)


@ops.sum.register(torch.Tensor, (int, type(None)))
def _sum(x, dim):
    return x.sum() if dim is None else x.sum(dim)


@ops.triangular_solve.register(torch.Tensor, torch.Tensor)
def _triangular_solve(x, y, upper=False, transpose=False):
    return x.triangular_solve(y, upper, transpose).solution
