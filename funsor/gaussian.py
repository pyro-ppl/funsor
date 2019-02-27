from __future__ import absolute_import, division, print_function

import torch
from six import integer_types

import funsor.ops as ops
from funsor.terms import Binary, Funsor, Number, eager
from funsor.torch import Tensor, materialize, align_tensors
from funsor.domains import reals


def _issubshape(subshape, supershape):
    if len(subshape) > len(supershape):
        return False
    for sub, sup in zip(reversed(subshape, supershape)):
        if sub not in (1, sup):
            return False
    return True


class Gaussian(Funsor):
    """
    Funsor representing a batched joint Gaussian distribution as a log-density
    function.
    """
    def __init__(self, log_density, loc, scale_tril, inputs):
        assert isinstance(log_density, torch.Tensor)
        assert isinstance(loc, torch.Tensor)
        assert isinstance(scale_tril, torch.Tensor)

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
        subs = {k: materialize(v) for k, v in subs if k in self.inputs}
        if not subs:
            return self

        raise NotImplementedError('TODO')

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            raise NotImplementedError('TODO')
        elif op is ops.add:
            raise NotImplementedError('TODO')

        return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Number)
def eager_binary_gaussian_number(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        log_density = op(lhs.log_density, rhs.data)
        return Gaussian(log_density, lhs.loc, lhs.scale_tril)

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


@eager.register(Binary, object, Gaussian, Gaussian)
def eager_binary_gaussian_gaussian(op, lhs, rhs):
    if op is ops.add:
        raise NotImplementedError('TODO Gaussian fusion')

    return None  # defer to default implementation
