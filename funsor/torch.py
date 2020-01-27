# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import funsor.ops as ops
from funsor.util import quote


################################################################################
# Register Ops
################################################################################

ops.abs.register(torch.Tensor)(torch.abs)
ops.sqrt.register(torch.Tensor)(torch.sqrt)
ops.exp.register(torch.Tensor)(torch.exp)
ops.log1p.register(torch.Tensor)(torch.log1p)
ops.unsqueeze.register(torch.Tensor, int)(torch.unsqueeze)
ops.transpose.register(torch.Tensor, int, int)(torch.transpose)
ops.full_like.register(torch.Tensor, object)(torch.full_like)
ops.clamp.register(torch.Tensor, object, object)(torch.clamp)


@quote.register(torch.Tensor)
def _quote(x, indent, out):
    """
    Work around PyTorch not supporting reproducible repr.
    """
    out.append((indent, f"torch.tensor({repr(x.tolist())}, dtype={x.dtype})"))


@ops.log.register(torch.Tensor)
def _log(x):
    if x.dtype in (torch.bool, torch.uint8, torch.long):
        x = x.float()
    return x.log()


@ops.pow.register(object, torch.Tensor)
def _pow(x, y):
    result = x ** y
    # work around shape bug https://github.com/pytorch/pytorch/issues/16685
    return result.reshape(y.shape)


@ops.pow.register(torch.Tensor, (object, torch.Tensor))
def _pow(x, y):
    return x ** y


@ops.min.register(torch.Tensor, torch.Tensor)
def _min(x, y):
    return torch.min(x, y)


@ops.min.register(object, torch.Tensor)
def _min(x, y):
    return y.clamp(max=x)


@ops.min.register(torch.Tensor, object)
def _min(x, y):
    return x.clamp(max=y)


@ops.max.register(torch.Tensor, torch.Tensor)
def _max(x, y):
    return torch.max(x, y)


@ops.max.register(object, torch.Tensor)
def _max(x, y):
    return y.clamp(min=x)


@ops.max.register(torch.Tensor, object)
def _max(x, y):
    return x.clamp(min=y)


@ops.reciprocal.register(torch.Tensor)
def _reciprocal(x):
    result = x.reciprocal().clamp(max=torch.finfo(x.dtype).max)
    return result


@ops.safesub.register(object, torch.Tensor)
def _safesub(x, y):
    try:
        finfo = torch.finfo(y.dtype)
    except TypeError:
        finfo = torch.iinfo(y.dtype)
    return x + (-y).clamp(max=finfo.max)


@ops.safediv.register(object, torch.Tensor)
def _safediv(x, y):
    try:
        finfo = torch.finfo(y.dtype)
    except TypeError:
        finfo = torch.iinfo(y.dtype)
    return x * y.reciprocal().clamp(max=finfo.max)


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


@ops.triangular_solve_op.register(torch.Tensor, torch.Tensor, bool, bool)
def _triangular_solve(x, y, upper, transpose):
    return x.triangular_solve(y, upper=upper, transpose=transpose).solution


@ops.diagonal.register(torch.Tensor, int, int)
def _diagonal(x, dim1, dim2):
    return x.diagonal(dim1=dim1, dim2=dim2)


@ops.cat_op.register(int, [torch.Tensor])
def _cat(dim, *x):
    return torch.cat(x, dim=dim)


@ops.new_zeros.register(torch.Tensor, tuple)
def _new_zeros(x, shape):
    return x.new_zeros(shape)


@ops.new_eye.register(torch.Tensor, tuple)
def _new_eye(x, shape):
    return torch.eye(shape[-1]).expand(shape + (-1,))


@ops.new_arange.register(torch.Tensor, int, int, int)
def _new_arange(x, start, stop, step):
    return torch.arange(start, stop, step)


@ops.new_arange.register(torch.Tensor, int)
def _new_arange(x, stop):
    return torch.arange(stop)


@ops.expand.register(torch.Tensor, tuple)
def _expand(x, shape):
    return x.expand(shape)


@ops.permute.register(torch.Tensor, (tuple, list))
def _permute(x, dims):
    return x.permute(dims)


@ops.finfo.register(torch.Tensor)
def _finfo(x):
    return torch.finfo(x.dtype)
