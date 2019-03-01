from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
from pyro.distributions.util import broadcast_shape
from six import add_metaclass, integer_types
from torch.distributions.multivariate_normal import _batch_mahalanobis

import funsor.ops as ops
from funsor.domains import reals
from funsor.terms import Binary, Funsor, FunsorMeta, Number, eager
from funsor.torch import Tensor, align_tensor, align_tensors, materialize


def _issubshape(subshape, supershape):
    if len(subshape) > len(supershape):
        return False
    for sub, sup in zip(reversed(subshape), reversed(supershape)):
        if sub not in (1, sup):
            return False
    return True


def _scale_tril_to_precision(scale_tril):
    scale_tril_inv = torch.inverse(scale_tril)
    return torch.matmul(scale_tril_inv.transpose(-1, -2), scale_tril_inv)


def _mv(mat, vec):
    return torch.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)


def _compute_offsets(inputs):
    """
    Compute offsets of real dims into the concatenated Gaussian dims.

    :param OrderedDict inputs: A schema mapping variable name to domain.
    :return: a pair ``(offsets, total)``.
    :rtype: tuple
    """
    assert isinstance(inputs, OrderedDict)
    offsets = {}
    total = 0
    for key, domain in inputs.items():
        if domain.dtype == 'real':
            offsets[key] = total
            total += domain.num_elements
    return offsets, total


def align_gaussian(new_inputs, old_inputs, log_density, loc, precision):
    """
    Align data of a Gaussian distribution to a new ``inputs`` shape.

    Note that this inputs and returns ``precision`` matrices rather than
    ``scale_tril`` matrices, since precision matrices can correctly be
    augmented with zero columns and rows.
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(old_inputs, OrderedDict)
    assert isinstance(log_density, torch.Tensor)
    assert isinstance(loc, torch.Tensor)
    assert isinstance(precision, torch.Tensor)

    # Align int inputs.
    # Since these are are managed as in Tensor, we can defer to align_tensor().
    new_ints = OrderedDict((k, d) for k, d in new_inputs.items() if d.dtype != 'real')
    old_ints = OrderedDict((k, d) for k, d in old_inputs.items() if d.dtype != 'real')
    if new_ints != old_ints:
        log_density = align_tensor(new_ints, Tensor(log_density, old_ints))
        loc = align_tensor(new_ints, Tensor(loc, old_ints))
        precision = align_tensor(new_ints, Tensor(precision, old_ints))

    # Align real inputs, which are all concatenated in the rightmost dims.
    new_offsets, new_dim = _compute_offsets(new_inputs)
    old_offsets, old_dim = _compute_offsets(old_inputs)
    assert loc.shape[-1:] == (old_dim,)
    assert precision.shape[-2:] == (old_dim, old_dim)
    if new_offsets != old_offsets:
        old_loc = loc
        old_precision = precision
        loc = old_loc.new_zeros(old_loc.shape[:-1] + (new_dim,))
        precision = old_loc.new_zeros(old_loc.shape[:-1] + (new_dim, new_dim))
        for key, new_offset in new_offsets.items():
            if key in old_offsets:
                offset = old_offsets[key]
                num_elements = old_inputs[key].num_elements
                old_slice = slice(offset, offset + num_elements)
                new_slice = slice(new_offset, new_offset + num_elements)
                loc[..., new_slice] = old_loc[..., old_slice]
                precision[..., new_slice, new_slice] = old_precision[..., old_slice, old_slice]

    return log_density, loc, precision


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
        assert dim
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
        subs = OrderedDict((k, materialize(v)) for k, v in subs if k in self.inputs)
        if not subs:
            return self

        # This currently handles only substitution of constants.
        if not all(isinstance(v, (Number, Tensor)) for v in subs.values()):
            raise NotImplementedError('TODO handle substitution of affine functions of variables')

        # First perform any integer substitution, i.e. slicing into a batch.
        int_subs = tuple((k, v) for k, v in subs.items() if v.dtype != 'real')
        real_subs = tuple((k, v) for k, v in subs.items() if v.dtype == 'real')
        if int_subs:
            int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
            real_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype == 'real')
            tensors = [self.log_density, self.loc, self.scale_tril]
            funsors = [Tensor(x, int_inputs).eager_subs(int_subs) for x in tensors]
            inputs = funsors[0].inputs.copy()
            inputs.update(real_inputs)
            int_result = Gaussian(funsors[0].data, funsors[1].data, funsors[2].data, inputs)
            return int_result.eager_subs(real_subs)

        # Try to perform a complete substitution of all real variables, resulting in a Tensor.
        assert real_subs and not int_subs
        if all(k in subs for k, d in self.inputs.items() if d.dtype == 'real'):
            int_inputs = OrderedDict((k, d) for k, d in self.inputs.items() if d.dtype != 'real')
            tensors = [Tensor(self.log_density, int_inputs),
                       Tensor(self.loc, int_inputs),
                       Tensor(self.scale_tril, int_inputs)]
            tensors.extend(subs.values())
            inputs, tensors = align_tensors(*tensors)
            offsets, event_size = _compute_offsets(self.inputs)
            batch_dim = tensors[0].dim()
            batch_shape = broadcast_shape(*(x.shape[:batch_dim] for x in tensors))
            (log_density, loc, scale_tril), values = tensors[:3], tensors[3:]
            value = loc.new_empty(batch_shape + (event_size,))
            for k, value_k in zip(subs, values):
                offset = offsets[k]
                value_k = value_k.reshape(value_k.shape[:batch_dim] + (-1,))
                assert value_k.size(-1) == self.inputs[k].num_elements
                value[..., offset: offset + self.inputs[k].num_elements] = value_k
            result = log_density - 0.5 * _batch_mahalanobis(scale_tril, value - loc)
            return Tensor(result, inputs)

        raise NotImplementedError('TODO implement partial substitution of real variables')

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp:
            # Marginalize out real variables.
            real_vars = frozenset(k for k, d in self.inputs.items() if d.dtype == "real")
            if reduced_vars.isdisjoint(real_vars):
                return None  # defer to default implementation
            inputs = OrderedDict((k, d) for k, d in self.inputs.items() if k not in real_vars)
            log_density = self.log_density
            if real_vars <= reduced_vars:
                result = Tensor(log_density, inputs)
            else:
                offsets, _ = _compute_offsets(self.inputs)
                index = []
                for key, domain in inputs.items():
                    if domain.dtype == 'real':
                        index.extend(range(offsets[key], offsets[key] + domain.num_elements))
                index = torch.tensor(index)

                loc = self.loc[..., index]
                self_covariance = torch.matmul(self.scale_tril,
                                               self.scale_tril.transpose(-1, -2))
                covariance = self_covariance[..., index.unsqueeze(-1), index]
                scale_tril = torch.cholesky(covariance)
                result = Gaussian(log_density, loc, scale_tril, inputs)
            # FIXME add log (2 pi) terms?
            return result.reduce(ops.logaddexp, reduced_vars - real_vars)

        elif op is ops.add:
            raise NotImplementedError('TODO product-reduce along a plate dimension')

        return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Number)
def eager_binary_gaussian_number(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        # Add a constant log_density term to a Gaussian.
        log_density = op(lhs.log_density, rhs.data)
        return Gaussian(log_density, lhs.loc, lhs.scale_tril, lhs.inputs)

    if op is ops.mul or op is ops.truediv:
        # Scale a Gaussian, as under pyro.poutine.scale.
        raise NotImplementedError('TODO')

    return None  # defer to default implementation


@eager.register(Binary, object, Number, Gaussian)
def eager_binary_number_gaussian(op, lhs, rhs):
    if op is ops.add:
        # Add a constant log_density term to a Gaussian.
        log_density = op(lhs.data, rhs.log_density)
        return Gaussian(log_density, rhs.loc, rhs.scale_tril, rhs.inputs)

    if op is ops.mul:
        # Scale a Gaussian, as under pyro.poutine.scale.
        raise NotImplementedError('TODO')

    return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Tensor)
def eager_binary_gaussian_tensor(op, lhs, rhs):
    if op is ops.add or op is ops.sub:
        # Add a batch-dependent log_density term to a Gaussian.
        nonreal_inputs = OrderedDict((k, d) for k, d in lhs.inputs.items()
                                     if d.dtype != 'real')
        inputs, (rhs_data, log_density, loc, scale_tril) = align_tensors(
                rhs,
                Tensor(lhs.log_density, nonreal_inputs),
                Tensor(lhs.loc, nonreal_inputs),
                Tensor(lhs.scale_tril, nonreal_inputs))
        log_density = op(log_density, rhs_data)
        inputs.update(lhs.inputs)
        return Gaussian(log_density, loc, scale_tril, inputs)

    if op is ops.mul or op is ops.truediv:
        # Scale a Gaussian, as under pyro.poutine.scale.
        raise NotImplementedError('TODO')

    return None  # defer to default implementation


@eager.register(Binary, object, Tensor, Gaussian)
def eager_binary_tensor_gaussian(op, lhs, rhs):
    if op is ops.add:
        # Add a batch-dependent log_density term to a Gaussian.
        nonreal_inputs = OrderedDict((k, d) for k, d in rhs.inputs.items()
                                     if d.dtype != 'real')
        inputs, (lhs_data, log_density, loc, scale_tril) = align_tensors(
                lhs,
                Tensor(rhs.log_density, nonreal_inputs),
                Tensor(rhs.loc, nonreal_inputs),
                Tensor(rhs.scale_tril, nonreal_inputs))
        log_density = op(lhs_data, log_density)
        inputs.update(rhs.inputs)
        return Gaussian(log_density, loc, scale_tril, inputs)

    if op is ops.mul:
        # Scale a Gaussian, as under pyro.poutine.scale.
        raise NotImplementedError('TODO')

    return None  # defer to default implementation


@eager.register(Binary, object, Gaussian, Gaussian)
def eager_binary_gaussian_gaussian(op, lhs, rhs):
    if op is ops.add:
        # Fuse two Gaussians by adding their log-densities pointwise.
        # This is similar to a Kalman filter update, but also keeps track of
        # the marginal likelihood.

        # Align data.
        inputs = lhs.inputs.copy()
        inputs.update(rhs.inputs)
        lhs_precision = _scale_tril_to_precision(lhs.scale_tril)
        rhs_precision = _scale_tril_to_precision(rhs.scale_tril)
        lhs_log_density, lhs_loc, lhs_precision = align_gaussian(
                inputs, lhs.inputs, lhs.log_density, lhs.loc, lhs_precision)
        rhs_log_density, rhs_loc, rhs_precision = align_gaussian(
                inputs, rhs.inputs, rhs.log_density, rhs.loc, rhs_precision)

        # Fuse aligned Gaussians.
        precision = lhs_precision + rhs_precision
        scale_tril_inv = torch.cholesky(precision)
        scale_tril = torch.inverse(scale_tril_inv)
        precision_loc = _mv(lhs_precision, lhs_loc) + _mv(rhs_precision, rhs_loc)
        loc = _mv(scale_tril, _mv(scale_tril.transpose(-1, -2), precision_loc))
        log_density = (lhs_log_density + rhs_log_density +  # FIXME add missing terms
                       _batch_mahalanobis(scale_tril, lhs_loc - rhs_loc))
        return Gaussian(log_density, loc, scale_tril, inputs)

    return None  # defer to default implementation


__all__ = [
    'Gaussian',
    'align_gaussian',
]
