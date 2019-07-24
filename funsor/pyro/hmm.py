from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
import torch
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
from funsor.interpreter import interpretation
from funsor.pyro.convert import dist_to_funsor, tensor_to_funsor
from funsor.pyro.distribution import FunsorDistribution
from funsor.terms import Independent, lazy


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
        # FIXME this non-scalar event_shape won't work for Categorical observation_dist.
        event_shape = time_shape + observation_dist.event_shape
        batch_shape = broadcast_shape(initial_logits.shape[:-1],
                                      transition_logits.shape[:-3],
                                      observation_dist.batch_shape[:-2])
        self._has_rsample = observation_dist.has_rsample

        # Convert tensors and distributions to funsors.
        init = tensor_to_funsor(initial_logits, event_dim=1)
        init = init["value"]

        if transition_logits.dim() == 2:
            trans = tensor_to_funsor(transition_logits, event_dim=2)
        elif transition_logits.size(-3) == 1:
            trans = tensor_to_funsor(transition_logits.squeeze(-3), event_dim=2)
        else:
            trans = tensor_to_funsor(transition_logits, event_dim=3)["time"]
        trans = trans["state", "state(time=1)"]

        if len(observation_dist.batch_shape) == 1:
            obs = dist_to_funsor(observation_dist, reinterpreted_batch_ndims=1)
        else:
            obs = dist_to_funsor(observation_dist, reinterpreted_batch_ndims=2)
            homogeneous = (observation_dist.batch_shape[-2] == 1)
            obs = obs[0 if homogeneous else "time"]
        obs = obs["state", "value"]
        dtype = obs.inputs["value"].dtype

        # Construct the joint, marginalizing over latent variables.
        with interpretation(lazy):
            # FIXME this is a bogus expression of the correct type. This should
            #   be replaced with markov_sum_product() once that is working.
            latent_vars = frozenset({"state", "state(time=1)"})
            funsor_dist = (init + trans + obs).reduce(ops.add, latent_vars)
            funsor_dist = Independent(funsor_dist, "value", "time")

        super(DiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape, dtype)

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
