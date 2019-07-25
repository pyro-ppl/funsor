import pyro.distributions as dist
import torch
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
from funsor.interpreter import interpretation
from funsor.pyro.convert import dist_to_funsor, funsor_to_tensor, tensor_to_funsor
from funsor.pyro.distribution import FunsorDistribution
from funsor.sum_product import sequential_sum_product
from funsor.terms import Variable, lazy


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
        init = tensor_to_funsor(initial_logits, ("state",))
        trans = tensor_to_funsor(transition_logits, ("time", "state", "state(time=1)"))
        obs = dist_to_funsor(observation_dist, ("time", "state"))
        dtype = obs.inputs["value"].dtype

        # Construct the joint funsor.
        with interpretation(lazy):
            # TODO perform math here once sequential_sum_product has been
            #   implemented as a first-class funsor.
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            # Until funsor_dist is defined, we save factors for hand-computation in .log_prob().
            self._init = init
            self._trans = trans
            self._obs = obs

        super(DiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape, dtype)

    @torch.distributions.constraints.dependent_property
    def has_rsample(self):
        return self._has_rsample

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        ndims = max(len(self.batch_shape), value.dim() - self.event_dim)
        value = tensor_to_funsor(value, ("time",), event_output=self.event_dim - 1,
                                 dtype=self.dtype)

        log_prob = self._obs(value=value)
        log_prob += self._trans
        log_prob = sequential_sum_product(ops.logaddexp, ops.add,
                                          log_prob, "time", "state", "state(time=1)")
        log_prob = log_prob.reduce(ops.logaddexp, "state(time=1)")
        log_prob += self._init
        log_prob = log_prob.reduce(ops.logaddexp, "state")

        log_prob = funsor_to_tensor(log_prob, ndims=ndims)
        return log_prob

    # TODO remove this once self.funsor_dist is defined.
    def _sample_delta(self, sample_shape):
        raise NotImplementedError("TODO")


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
