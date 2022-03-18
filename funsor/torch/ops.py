# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numbers
import typing

import torch

import funsor.ops as ops

################################################################################
# Register Ops
################################################################################

ops.abs.register(torch.Tensor)(torch.abs)
ops.atanh.register(torch.Tensor)(torch.atanh)
ops.cholesky_solve.register(torch.Tensor, torch.Tensor)(torch.cholesky_solve)
ops.clamp.register(torch.Tensor)(torch.clamp)
ops.exp.register(torch.Tensor)(torch.exp)
ops.full_like.register(torch.Tensor)(torch.full_like)
ops.log1p.register(torch.Tensor)(torch.log1p)
ops.sigmoid.register(torch.Tensor)(torch.sigmoid)
ops.sqrt.register(torch.Tensor)(torch.sqrt)
ops.tanh.register(torch.Tensor)(torch.tanh)
ops.transpose.register(torch.Tensor)(torch.transpose)
ops.flip.register(torch.Tensor)(torch.flip)
ops.unsqueeze.register(torch.Tensor)(torch.unsqueeze)
ops.qr.register(torch.Tensor)(torch.linalg.qr)


###########################################
# Reduction Ops
###########################################


def _flatten_reduced_dim(x, dim):
    # Canonicalize reduced dim.
    reduced_dim = (
        tuple(range(x.dim())) if dim is None else tuple(d % x.dim() for d in dim)
    )
    nonreduced_dim = tuple(i for i in range(x.dim()) if i not in reduced_dim)
    # permute & flatten reduced dim.
    permutation = nonreduced_dim + reduced_dim
    return x.permute(permutation).flatten(-len(reduced_dim), -1), reduced_dim


@ops.all.register(torch.Tensor)
def _all(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.all(x)

    if isinstance(axis, int):
        return torch.all(x, axis, keepdim=keepdims)

    # reduce over multiple dims.
    x_flattened, reduced_dim = _flatten_reduced_dim(x, axis)
    if keepdims:
        shape = tuple(1 if i in reduced_dim else x.shape[i] for i in range(x.dim()))
        return torch.all(x_flattened, -1).view(shape)
    return torch.all(x_flattened, -1)


@ops.any.register(torch.Tensor)
def _any(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.any(x)

    if isinstance(axis, int):
        return torch.any(x, axis, keepdim=keepdims)

    # reduce over multiple dims.
    x_flattened, reduced_dim = _flatten_reduced_dim(x, axis)
    if keepdims:
        shape = tuple(1 if i in reduced_dim else x.shape[i] for i in range(x.dim()))
        return torch.any(x_flattened, -1).view(shape)
    return torch.any(x_flattened, -1)


@ops.amax.register(torch.Tensor)
def _amax(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.amax(x)
    axis = tuple(range(x.dim())) if axis is None else axis
    return torch.amax(x, axis, keepdim=keepdims)


@ops.amin.register(torch.Tensor)
def _amin(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.amin(x)
    axis = tuple(range(x.dim())) if axis is None else axis
    return torch.amin(x, axis, keepdim=keepdims)


@ops.sum.register(torch.Tensor)
def _sum(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.sum(x)
    axis = tuple(range(x.dim())) if axis is None else axis
    return torch.sum(x, axis, keepdim=keepdims)


@ops.prod.register(torch.Tensor)
def _prod(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.prod(x)

    if isinstance(axis, int):
        return torch.prod(x, axis, keepdim=keepdims)

    # reduce over multiple dims.
    x_flattened, reduced_dim = _flatten_reduced_dim(x, axis)
    if keepdims:
        shape = tuple(1 if i in reduced_dim else x.shape[i] for i in range(x.dim()))
        return torch.prod(x_flattened, -1).view(shape)
    return torch.prod(x_flattened, -1)


@ops.logsumexp.register(torch.Tensor)
def _logsumexp(x, axis, keepdims):
    axis = tuple(range(x.dim())) if axis is None else axis
    return torch.logsumexp(x, axis, keepdim=keepdims)


@ops.mean.register(torch.Tensor)
def _mean(x, axis, keepdims):
    if axis is None and not keepdims:
        return torch.mean(x)
    axis = tuple(range(x.dim())) if axis is None else axis
    return torch.mean(x, axis, keepdim=keepdims)


@ops.std.register(torch.Tensor)
def _std(x, axis, ddof, keepdims):
    axis = tuple(range(x.dim())) if axis is None else axis
    if ddof == 0:
        return torch.std(x, axis, unbiased=False, keepdim=keepdims)
    if ddof == 1:
        return torch.std(x, axis, keepdim=keepdims)
    raise NotImplementedError


@ops.var.register(torch.Tensor)
def _var(x, axis, ddof, keepdims):
    axis = tuple(range(x.dim())) if axis is None else axis
    if ddof == 0:
        return torch.var(x, axis, unbiased=False, keepdim=keepdims)
    if ddof == 1:
        return torch.var(x, axis, keepdim=keepdims)
    raise NotImplementedError


###########################################


@ops.argmax.register(torch.Tensor)
def _argmax(x, axis, keepdims):
    # FIXME find_domain
    return torch.argmax(x, axis, keepdim=keepdims)


@ops.argmin.register(torch.Tensor)
def _argmin(x, axis, keepdims):
    # FIXME find_domain
    return torch.argmin(x, axis, keepdim=keepdims)


@ops.astype.register(torch.Tensor)
def _astype(x, dtype):
    return x.type(getattr(torch, dtype))


ops.cat.register(typing.Tuple[torch.Tensor, ...])(torch.cat)


@ops.cholesky.register(torch.Tensor)
def _cholesky(x):
    """
    Like :func:`torch.linalg.cholesky` but uses sqrt for scalar matrices.
    Works around https://github.com/pytorch/pytorch/issues/24403 often.
    """
    if x.size(-1) == 1:
        return x.sqrt()
    return torch.linalg.cholesky(x)


@ops.cholesky_inverse.register(torch.Tensor)
def _cholesky_inverse(x):
    """
    Like :func:`torch.cholesky_inverse` but supports batching and gradients.
    """
    if x.dim() == 2:
        return x.cholesky_inverse()
    return torch.eye(x.size(-1)).cholesky_solve(x)


@ops.triangular_inv.register(torch.Tensor)
def _triangular_inv(x, upper=False):
    if x.size(-1) == 1:
        return x.reciprocal()
    return torch.linalg.solve_triangular(x, torch.eye(x.size(-1)), upper=upper)


@ops.detach.register(torch.Tensor)
def _detach(x):
    return x.detach()


@ops.diagonal.register(torch.Tensor)
def _diagonal(x, dim1, dim2):
    return x.diagonal(dim1=dim1, dim2=dim2)


@ops.einsum.register(typing.Tuple[torch.Tensor, ...])
def _einsum(operands, equation):
    return torch.einsum(equation, *operands)


@ops.expand.register(torch.Tensor)
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


@ops.logaddexp.register(torch.Tensor, torch.Tensor)
def _safe_logaddexp_tensor_tensor(x, y):
    finfo = torch.finfo(x.dtype)
    shift = torch.max(x.detach(), y.detach()).clamp(min=finfo.min)
    return torch.log(torch.exp(x - shift) + torch.exp(y - shift)) + shift


@ops.logaddexp.register(numbers.Number, torch.Tensor)
def _safe_logaddexp_number_tensor(x, y):
    finfo = torch.finfo(y.dtype)
    shift = y.detach().clamp(min=max(x, finfo.min))
    return torch.log(torch.exp(x - shift) + torch.exp(y - shift)) + shift


@ops.logaddexp.register(torch.Tensor, numbers.Number)
def _safe_logaddexp_tensor_number(x, y):
    return _safe_logaddexp_number_tensor(y, x)


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


@ops.new_arange.register(torch.Tensor)
def _new_arange(x, start, stop, step):
    if step is not None:
        return torch.arange(start, stop, step)
    if stop is not None:
        return torch.arange(start, stop)
    return torch.arange(start)


@ops.new_eye.register(torch.Tensor)
def _new_eye(x, shape):
    return torch.eye(shape[-1]).expand(shape + (-1,))


@ops.new_zeros.register(torch.Tensor)
def _new_zeros(x, shape):
    return x.new_zeros(shape)


@ops.new_full.register(torch.Tensor)
def _new_full(x, shape, value):
    return x.new_full(shape, value)


@ops.randn.register(torch.Tensor)
def _randn(prototype, shape, rng_key=None):
    assert isinstance(shape, tuple)
    return torch.randn(shape, dtype=prototype.dtype, device=prototype.device)


@ops.permute.register(torch.Tensor)
def _permute(x, dims):
    return x.permute(dims)


@ops.pow.register(numbers.Number, torch.Tensor)
def _pow(x, y):
    result = x**y
    # work around shape bug https://github.com/pytorch/pytorch/issues/16685
    return result.reshape(y.shape)


@ops.pow.register(torch.Tensor, numbers.Number)
@ops.pow.register(torch.Tensor, torch.Tensor)
def _pow(x, y):
    return x**y


@ops.reciprocal.register(torch.Tensor)
def _reciprocal(x):
    if torch.is_complex(x):
        return x.reciprocal()
    result = x.reciprocal().clamp(max=torch.finfo(x.dtype).max)
    return result


@ops.safediv.register(torch.Tensor, torch.Tensor)
@ops.safediv.register(numbers.Number, torch.Tensor)
def _safediv(x, y):
    try:
        finfo = torch.finfo(y.dtype)
    except TypeError:
        finfo = torch.iinfo(y.dtype)
    return x * y.reciprocal().clamp(max=finfo.max)


@ops.safesub.register(torch.Tensor, torch.Tensor)
@ops.safesub.register(numbers.Number, torch.Tensor)
def _safesub(x, y):
    try:
        finfo = torch.finfo(y.dtype)
    except TypeError:
        finfo = torch.iinfo(y.dtype)
    return x + (-y).clamp(max=finfo.max)


@ops.scatter.register(torch.Tensor, tuple, torch.Tensor)
def _scatter(destin, indices, source):
    result = destin.clone()
    result[indices] = source
    return result


@ops.scatter_add.register(torch.Tensor, tuple, torch.Tensor)
def _scatter_add(destin, indices, source):
    result = destin.clone()
    return result.index_put(indices, source, accumulate=True)


ops.stack.register(typing.Tuple[torch.Tensor, ...])(torch.stack)


@ops.triangular_solve.register(torch.Tensor, torch.Tensor)
def _triangular_solve(x, y, upper=False, transpose=False):
    if y.size(-1) == 1:
        return x / y
    if transpose:
        y = y.transpose(-1, -2)
        upper = not upper
    return torch.linalg.solve_triangular(y, x, upper=upper)
