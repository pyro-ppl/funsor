from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
from six import add_metaclass, integer_types
from torch.distributions.multivariate_normal import _batch_mahalanobis

import funsor.ops as ops
from funsor.domains import reals
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager
from funsor.torch import Tensor, align_tensors, arange


def _issubshape(subshape, supershape):
    if len(subshape) > len(supershape):
        return False
    for sub, sup in zip(reversed(subshape), reversed(supershape)):
        if sub not in (1, sup):
            return False
    return True


def _scale_tril_to_precision(scale_tril):
    scale_tril_inv = torch.inv(scale_tril)
    return torch.matmul(scale_tril_inv.transpose(-1, -2), scale_tril_inv)


def to_affine(x):
    """
    Attempt to convert a Funsor to a combination of
    :class:`~funsor.terms.Number`s, :class:`Tensor`s, and affine functions of
    :class:`~funsor.terms.Variable`s by substituting :func:`arange`s into its
    free variables and rearranging terms.
    """
    assert isinstance(x, Funsor)
    if isinstance(x, (Number, Tensor)):
        return x
    subs = []
    for name, domain in x.inputs.items():
        if not isinstance(domain.dtype, integer_types):
            raise NotImplementedError('TODO')
        assert not domain.shape
        subs.append((name, arange(name, domain.dtype)))
    subs = tuple(subs)
    return x.eager_subs(subs)


class GaussianMeta(FunsorMeta):
    """
    Wrapper to convert between OrderedDict and tuple.
    """
    def __call__(cls, log_density, loc, scale_tril, inputs):
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        return super(GaussianMeta, cls).__call__(log_density, loc, scale_tril, inputs)


@add_metaclass(GaussianMeta)
class Gaussian(Funsor):
    """
    Funsor representing a batched joint Gaussian distribution as a log-density
    function.
    """
    def __init__(self, log_density, loc, scale_tril, inputs):
        assert isinstance(log_density, torch.Tensor)
        assert isinstance(loc, torch.Tensor)
        assert isinstance(scale_tril, torch.Tensor)
        assert isinstance(inputs, tuple)
        inputs = OrderedDict(inputs)

        # Compute total dimension of all real inputs.
        dim = sum(d.num_elements for d in inputs.values() if d.dtype == 'real')
        assert loc.dim() >= 1 and loc.size(-1) == dim
        assert scale_tril.dim() >= 2 and scale_tril.shape[-2:] == (dim, dim)

        # Compute total shape of all bint inputs.
        batch_shape = tuple(d.dtype for d in inputs.values()
                            if isinstance(d.dtype, integer_types))
        assert _issubshape(log_density.shape, batch_shape)
        assert _issubshape(loc.shape, batch_shape + (dim,))
        assert _issubshape(scale_tril.shape, batch_shape + (dim, dim))

        output = reals()
        super(Gaussian, self).__init__(inputs, output)
        self.log_density = log_density
        self.loc = loc
        self.scale_tril = scale_tril
        self.batch_shape = batch_shape
        self.event_shape = (dim,)

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        subs = {k: to_affine(v) for k, v in subs if k in self.inputs}
        if not subs:
            return self

        raise NotImplementedError('TODO')

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            raise NotImplementedError('TODO marginalize a real variable')
        elif op is ops.add:
            raise NotImplementedError('TODO product-reduce along a plate dimension')

        return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Number)
def eager_binary_gaussian_number(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        log_density = op(lhs.log_density, rhs.data)
        return Gaussian(log_density, lhs.loc, lhs.scale_tril)

    return None  # defer to default implementation


@eager.register(Binary, object, Number, Gaussian)
def eager_binary_number_gaussian(op, lhs, rhs):
    if op is ops.add:
        log_density = op(lhs.data, rhs.log_density)
        return Gaussian(log_density, rhs.loc, rhs.scale_tril)

    return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Tensor)
def eager_binary_gaussian_tensor(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        inputs, (rhs_data, log_density, loc, scale_tril) = align_tensors(
                rhs.data,
                Tensor(lhs.log_density, lhs.inputs),
                Tensor(lhs.loc, lhs.inputs),
                Tensor(lhs.scale_tril, lhs.inputs))
        log_density = op(log_density, rhs_data)
        return Gaussian(log_density, loc, scale_tril, inputs)

    return None  # defer to default implementation


@eager.register(Binary, object, Tensor, Gaussian)
def eager_binary_tensor_gaussian(op, lhs, rhs):
    if op is ops.add:
        inputs, (lhs_data, log_density, loc, scale_tril) = align_tensors(
                lhs.data,
                Tensor(rhs.log_density, lhs.inputs),
                Tensor(rhs.loc, lhs.inputs),
                Tensor(rhs.scale_tril, lhs.inputs))
        log_density = op(lhs_data, log_density)
        return Gaussian(log_density, loc, scale_tril, inputs)

    return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Gaussian)
def eager_binary_gaussian_gaussian(op, lhs, rhs):
    if op is ops.add:

        if lhs.inputs != rhs.inputs:
            raise NotImplementedError('TODO align vectors and matrices')

        inputs = lhs.inputs
        lhs_precision = _scale_tril_to_precision(lhs.scale_tril)
        rhs_precision = _scale_tril_to_precision(rhs.scale_tril)
        precision = lhs_precision + rhs_precision
        scale_tril_inv = torch.cholesky(precision)
        scale_tril = torch.inverse(scale_tril_inv)
        loc = torch.matmul(scale_tril,
                           torch.matmul(scale_tril.transpose(-1, 2),
                                        torch.matmul(lhs_precision, lhs.loc) +
                                        torch.matmul(rhs_precision, rhs.loc)))
        log_density = (lhs.log_density + rhs.log_density +  # FIXME add missing terms
                       _batch_mahalanobis(scale_tril, lhs.loc, rhs.loc))
        return Gaussian(log_density, loc, scale_tril, inputs)

    return None  # defer to default implementation


__all__ = [
    'Gaussian',
    'to_affine',
]
