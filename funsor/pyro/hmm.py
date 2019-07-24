from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
import torch
from pyro.distributions.util import broadcast_shape

from funsor.domains import bint
from funsor.interpreter import interpretation
from funsor.pyro.convert import dist_to_funsor, tensor_to_funsor
from funsor.pyro.distribution import FunsorDistribution
from funsor.terms import lazy


class DiscreteDiscreteHMM(FunsorDistribution):
    def __init__(self, initial_logits, transition_logits, observation_logits):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(observation_logits, torch.Tensor)
        assert initial_logits.dim() >= 1
        assert transition_logits.dim() >= 2
        assert observation_logits.dim() >= 2
        time_shape = broadcast_shape((1,), transition_logits.shape[-3:-2],
                                     observation_logits.shape[-3:-2])
        time_domain = bint(time_shape[0])
        event_shape = time_shape + observation_logits.shape[-1:]
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                      transition_logits.shape[:-3],
                                      observation_logits.shape[:-3])

        # Convert tensors to funsors.
        initial_logits = tensor_to_funsor(initial_logits, event_dim=1)

        inputs = self.inputs.copy()
        if transition_logits.dim() >= 3 and transition_logits.size(-3) > 1:
            inputs["time"] = time_domain
        transition_logits = tensor_to_funsor(transition_logits, event_dim=2, inputs=inputs)

        inputs = self.inputs.copy()
        if observation_logits.dim() >= 3 and observation_logits.size(-3) > 1:
            inputs["time"] = time_domain
        observation_logits = tensor_to_funsor(observation_logits, event_dim=2, inputs=inputs)

        with interpretation(lazy):
            funsor_dist = initial_logits + transition_logits + observation_logits
        super(DiscreteDiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape)


class DiscreteHMM(FunsorDistribution):
    def __init__(self, initial_logits, transition_logits, observation_dist):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.Distribution)
        assert initial_logits.dim() >= 1
        assert transition_logits.dim() >= 2
        assert len(observation_dist.batch_shape) >= 1
        time_shape = broadcast_shape((1,), transition_logits.shape[-3:-2],
                                     observation_dist.batch_shape[-2:-1])
        time_domain = bint(time_shape[0])
        event_shape = time_shape + observation_dist.event_shape
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                      transition_logits.shape[:-3],
                                      observation_dist.batch_shape[:-2])
        self._has_rsample = observation_dist.has_rsample

        # Convert tensors and distributions to funsors.
        initial_logits = tensor_to_funsor(initial_logits, event_dim=1)

        inputs = self.inputs.copy()
        if transition_logits.dim() >= 3 and transition_logits.size(-3) > 1:
            inputs["time"] = time_domain
        transition_logits = tensor_to_funsor(transition_logits, event_dim=2, inputs=inputs)

        inputs = self.inputs.copy()
        if len(observation_dist.batch_shape) >= 2 and observation_dist.batch_shape[-2] > 1:
            inputs["time"] = time_domain
        observation_dist = dist_to_funsor(observation_dist, inputs=inputs)

        with interpretation(lazy):
            funsor_dist = initial_logits + transition_logits + observation_dist
        super(DiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape)

    @torch.distributions.constraints.dependent_property
    def has_rsample(self):
        return self._has_rsample


class GaussianGaussianHMM(FunsorDistribution):
    has_rsample = True

    def __init__(self, initial_dist, transition_matrix, transition_dist,
                 observation_matrix, observation_dist):
        assert isinstance(initial_dist, dist.MultivariateNormal)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_dist, dist.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_dist, dist.MultivariateNormal)
        # TODO assert shapes...
        raise NotImplementedError("TODO")
