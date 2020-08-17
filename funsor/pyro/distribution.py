# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import torch
from torch.distributions import constraints
from pyro.distributions import TorchDistribution

from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import Bint
from funsor.pyro.convert import DIM_TO_NAME, funsor_to_tensor, tensor_to_funsor
from funsor.terms import Funsor, to_funsor


class FunsorDistribution(TorchDistribution):
    """
    :class:`~torch.distributions.Distribution` wrapper around a
    :class:`~funsor.terms.Funsor` for use in Pyro code. This is typically used
    as a base class for specific funsor inference algorithms wrapped in a
    distribution interface.

    :param funsor.terms.Funsor funsor_dist: A funsor with an input named
        "value" that is treated as a random variable. The distribution should
        be normalized over "value".
    :param torch.Size batch_shape: The distribution's batch shape. This must
        be in the same order as the input of the ``funsor_dist``, but may
        contain extra dims of size 1.
    :param event_shape: The distribution's event shape.
    """
    arg_constraints = {}

    def __init__(self, funsor_dist, batch_shape=torch.Size(), event_shape=torch.Size(),
                 dtype="real", validate_args=None):
        assert isinstance(funsor_dist, Funsor)
        assert isinstance(batch_shape, tuple)
        assert isinstance(event_shape, tuple)
        assert "value" in funsor_dist.inputs
        super(FunsorDistribution, self).__init__(batch_shape, event_shape, validate_args)
        self.funsor_dist = funsor_dist
        self.dtype = dtype

    @constraints.dependent_property
    def support(self):
        if self.dtype == "real":
            return constraints.real
        else:
            return constraints.integer_interval(0, self.dtype - 1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ndims = max(len(self.batch_shape), value.dim() - self.event_dim)
        value = tensor_to_funsor(value, event_output=self.event_dim, dtype=self.dtype)
        log_prob = self.funsor_dist(value=value)
        log_prob = funsor_to_tensor(log_prob, ndims=ndims)
        return log_prob

    def _sample_delta(self, sample_shape):
        sample_inputs = None
        if sample_shape:
            sample_inputs = OrderedDict()
            shape = sample_shape + self.batch_shape
            for dim in range(-len(shape), -len(self.batch_shape)):
                if shape[dim] > 1:
                    sample_inputs[DIM_TO_NAME[dim]] = Bint[shape[dim]]
        delta = self.funsor_dist.sample(frozenset({"value"}), sample_inputs)
        if isinstance(delta, Contraction):
            assert len([d for d in delta.terms if isinstance(d, Delta)]) == 1
            delta = delta.terms[0]
        assert isinstance(delta, Delta)
        return delta

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        delta = self._sample_delta(sample_shape)
        ndims = len(sample_shape) + len(self.batch_shape) + len(self.event_shape)
        value = funsor_to_tensor(delta.terms[0][1][0], ndims=ndims)
        return value.detach()

    def rsample(self, sample_shape=torch.Size()):
        delta = self._sample_delta(sample_shape)
        assert not delta.log_density.requires_grad, "distribution is not fully reparametrized"
        ndims = len(sample_shape) + len(self.batch_shape) + len(self.event_shape)
        value = funsor_to_tensor(delta.terms[0][1][0], ndims=ndims)
        return value

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        funsor_dist = self.funsor_dist + tensor_to_funsor(torch.zeros(batch_shape))
        super(type(self), new).__init__(
            funsor_dist, batch_shape, self.event_shape, self.dtype, validate_args=False)
        new.validate_args = self.__dict__.get('_validate_args')
        return new


@to_funsor.register(FunsorDistribution)
def funsordistribution_to_funsor(pyro_dist, output=None, dim_to_name=None):
    raise NotImplementedError("TODO implement conversion for FunsorDistribution")
