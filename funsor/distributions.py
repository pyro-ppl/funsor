from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import pyro.distributions as dist
import torch
from six import add_metaclass

import funsor.delta
import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.gaussian import Gaussian
from funsor.terms import Funsor, FunsorMeta, Number, Variable, eager, to_funsor
from funsor.torch import Tensor, align_tensors, materialize


def numbers_to_tensors(*args):
    """
    Convert :class:`~funsor.terms.Number`s to :class:`funsor.torch.Tensor`s,
    using any provided tensor as a prototype, if available.
    """
    if any(isinstance(x, Number) for x in args):
        new_tensor = torch.tensor
        for x in args:
            if isinstance(x, Tensor):
                new_tensor = x.data.new_tensor
                break
        args = tuple(Tensor(new_tensor(x.data), dtype=x.dtype) if isinstance(x, Number) else x
                     for x in args)
    return args


class DistributionMeta(FunsorMeta):
    """
    Wrapper to fill in default values and convert Numbers to Tensors.
    """
    def __call__(cls, *args, **kwargs):
        args = cls._fill_defaults(*args, **kwargs)
        args = numbers_to_tensors(*args)
        return super(DistributionMeta, cls).__call__(*args)


@add_metaclass(DistributionMeta)
class Distribution(Funsor):
    """
    Funsor backed by a PyTorch distribution object.
    """
    dist_class = "defined by derived classes"

    def __init__(self, *args):
        params = tuple(zip(self._ast_fields, args))
        assert any(k == 'value' for k, v in params)
        inputs = OrderedDict()
        for name, value in params:
            assert isinstance(name, str)
            assert isinstance(value, Funsor)
            inputs.update(value.inputs)
        inputs = OrderedDict(inputs)
        output = reals()
        super(Distribution, self).__init__(inputs, output)
        self.params = params

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               ', '.join('{}={}'.format(*kv) for kv in self.params))

    def eager_subs(self, subs):
        assert isinstance(subs, tuple)
        if not any(k in self.inputs for k, v in subs):
            return self
        params = OrderedDict((k, v.eager_subs(subs)) for k, v in self.params)
        return type(self)(**params)

    def eager_reduce(self, op, reduced_vars):
        if op is ops.logaddexp and isinstance(self.value, Variable) and self.value.name in reduced_vars:
            return Number(0.)  # distributions are normalized
        return super(Distribution, self).reduce(op, reduced_vars)

    @classmethod
    def eager_log_prob(cls, **params):
        inputs, tensors = align_tensors(*params.values())
        params = dict(zip(params, tensors))
        value = params.pop('value')
        data = cls.dist_class(**params).log_prob(value)
        return Tensor(data, inputs)


################################################################################
# Distribution Wrappers
################################################################################

class Categorical(Distribution):
    dist_class = dist.Categorical

    @staticmethod
    def _fill_defaults(probs, value=None):
        probs = to_funsor(probs)
        if value is None:
            size = probs.output.shape[0]
            value = Variable('value', bint(size))
        else:
            value = to_funsor(value)
        return probs, value

    def __init__(self, probs, value=None):
        super(Categorical, self).__init__(probs, value)


@eager.register(Categorical, Funsor, Tensor)
def eager_categorical(probs, value):
    return probs[value].log()


@eager.register(Categorical, Tensor, Tensor)
def eager_categorical(probs, value):
    return Categorical.eager_log_prob(probs=probs, value=value)


@eager.register(Categorical, Tensor, Variable)
def eager_categorical(probs, value):
    value = materialize(value)
    return Categorical.eager_log_prob(probs=probs, value=value)


class Delta(Distribution):
    dist_class = dist.Delta

    @staticmethod
    def _fill_defaults(v, log_density=0, value=None):
        v = to_funsor(v)
        log_density = to_funsor(log_density)
        if value is None:
            value = Variable('value', reals())
        else:
            value = to_funsor(value)
        return v, log_density, value

    def __init__(self, v, log_density=0, value=None):
        return super(Delta, self).__init__(v, log_density, value)


@eager.register(Delta, Tensor, Tensor, Tensor)
def eager_delta(v, log_density, value):
    # This handles event_dim specially, and hence cannot use the
    # generic Delta.eager_log_prob() method.
    assert v.output == value.output
    event_dim = len(v.output.shape)
    inputs, (v, log_density, value) = align_tensors(v, log_density, value)
    data = dist.Delta(v, log_density, event_dim).log_prob(value)
    return Tensor(data, inputs)


@eager.register(Delta, Funsor, Funsor, Variable)
@eager.register(Delta, Variable, Funsor, Variable)
def eager_delta(v, log_density, value):
    assert v.output == value.output
    return funsor.delta.Delta(value.name, v, log_density)


@eager.register(Delta, Variable, Funsor, Funsor)
def eager_delta(v, log_density, value):
    assert v.output == value.output
    return funsor.delta.Delta(v.name, value, log_density)


class Normal(Distribution):
    dist_class = dist.Normal

    @staticmethod
    def _fill_defaults(loc, scale, value=None):
        loc = to_funsor(loc)
        scale = to_funsor(scale)
        if value is None:
            value = Variable('value', reals())
        else:
            value = to_funsor(value)
        return loc, scale, value

    def __init__(self, loc, scale, value=None):
        super(Normal, self).__init__(loc, scale, value)


@eager.register(Normal, Tensor, Tensor, Tensor)
def eager_normal(loc, scale, value):
    return Normal.eager_log_prob(loc=loc, scale=scale, value=value)


# Create a Gaussian from a ground observation.
@eager.register(Normal, Variable, Tensor, Tensor)
def eager_normal(loc, scale, value):
    assert loc.output == reals()
    inputs, (scale, value) = align_tensors(scale, value)
    inputs.update(loc.inputs)

    log_density = -0.5 * math.log(2 * math.pi) - scale.log()
    loc = value.unsqueeze(-1)
    precision = scale.pow(-2).unsqueeze(-1).unsqueeze(-1)
    return Gaussian(log_density, loc, precision, inputs)


# Create a Gaussian from a ground observation.
@eager.register(Normal, Tensor, Tensor, Variable)
def eager_normal(loc, scale, value):
    assert value.output == reals()
    inputs, (loc, scale) = align_tensors(loc, scale)
    inputs.update(value.inputs)

    log_density = -0.5 * math.log(2 * math.pi) - scale.log()
    loc = loc.unsqueeze(-1)
    precision = scale.pow(-2).unsqueeze(-1).unsqueeze(-1)
    return Gaussian(log_density, loc, precision, inputs)


# Create a Gaussian from a noisy identity transform.
# This is extremely limited but suffices for examples/kalman_filter.py
@eager.register(Normal, Variable, Tensor, Variable)
def eager_normal(loc, scale, value):
    assert loc.output == reals()
    assert value.output == reals()
    assert loc.name != value.name
    inputs = loc.inputs.copy()
    inputs.update(scale.inputs)
    inputs.update(value.inputs)

    log_density = -0.5 * math.log(2 * math.pi) - scale.data.log()
    loc = scale.data.new_zeros(scale.data.shape + (2,))
    p = scale.data.pow(-2)
    precision = torch.stack([p, -p, -p, p], -1).reshape(p.shape + (2, 2))
    return Gaussian(log_density, loc, precision, inputs)


__all__ = [
    'Categorical',
    'Delta',
    'Distribution',
    'Normal',
]
