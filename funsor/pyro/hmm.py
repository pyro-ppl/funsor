import torch
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
from funsor.domains import reals
from funsor.gaussian import Gaussian
from funsor.interpreter import interpretation
from funsor.pyro.convert import dist_to_funsor, funsor_to_tensor, tensor_to_funsor
from funsor.pyro.distribution import FunsorDistribution
from funsor.sum_product import sequential_sum_product
from funsor.terms import Variable, lazy, Stack


class DiscreteHMM(FunsorDistribution):
    def __init__(self, initial_logits, transition_logits, observation_dist):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.Distribution)
        assert initial_logits.dim() >= 1
        assert transition_logits.dim() >= 2
        assert len(observation_dist.batch_shape) >= 1
        shape = broadcast_shape(initial_logits.shape[:-1] + (1,),
                                transition_logits.shape[:-2],
                                observation_dist.batch_shape[:-1])
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + observation_dist.event_shape
        self._has_rsample = observation_dist.has_rsample

        # Normalize.
        initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        transition_logits = transition_logits - transition_logits.logsumexp(-1, True)

        # Convert tensors and distributions to funsors.
        init = tensor_to_funsor(initial_logits, ("state",))
        trans = tensor_to_funsor(transition_logits, ("time", "state", "state(time=1)"))
        obs = dist_to_funsor(observation_dist, ("time", "state(time=1)"))
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

        # Compare with pyro.distributions.hmm.DiscreteHMM.log_prob().
        obs = self._obs(value=value)
        result = self._trans + obs
        result = sequential_sum_product(ops.logaddexp, ops.add,
                                        result, "time", "state", "state(time=1)")
        result = self._init + result.reduce(ops.logaddexp, "state(time=1)")
        result = result.reduce(ops.logaddexp, "state")

        result = funsor_to_tensor(result, ndims=ndims)
        return result

    # TODO remove this once self.funsor_dist is defined.
    def _sample_delta(self, sample_shape):
        raise NotImplementedError("TODO")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DiscreteHMM, _instance)
        batch_shape = torch.Size(batch_shape)
        new._has_rsample = self._has_rsample
        new._init = self._init
        new._trans = self._trans
        new._obs = self._obs
        super(DiscreteHMM, new).__init__(self.funsor_dist, batch_shape, self.event_shape)
        return new


class GaussianMRF(FunsorDistribution):
    has_rsample = True

    def __init__(self, initial_dist, transition_dist, observation_dist):
        assert isinstance(initial_dist, torch.distributions.MultivariateNormal)
        assert isinstance(transition_dist, torch.distributions.MultivariateNormal)
        assert isinstance(observation_dist, torch.distributions.MultivariateNormal)
        hidden_dim = initial_dist.event_shape[0]
        assert transition_dist.event_shape[0] == hidden_dim + hidden_dim
        obs_dim = observation_dist.event_shape[0] - hidden_dim
        shape = broadcast_shape(initial_dist.batch_shape + (1,),
                                transition_dist.batch_shape,
                                observation_dist.batch_shape)
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)

        # Convert distributions to funsors.
        init = dist_to_funsor(initial_dist)(value="state")
        trans_inputs = tensor_to_funsor(transition_dist.loc, ("time",), 1).inputs.copy()
        trans_inputs["state"] = reals(hidden_dim)
        trans_inputs["state(time=1)"] = reals(hidden_dim)
        trans = Gaussian(transition_dist.loc, transition_dist.precision_matrix, trans_inputs)
        obs_inputs = tensor_to_funsor(observation_dist.loc, ("time",), 1).inputs.copy()
        obs_inputs["state(time=1)"] = reals(hidden_dim)
        obs_inputs["value"] = reals(obs_dim)
        obs = Gaussian(observation_dist.loc, observation_dist.precision_matrix, obs_inputs)

        # Construct the joint funsor.
        with interpretation(lazy):
            # TODO perform math here once sequential_sum_product has been
            #   implemented as a first-class funsor.
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            # Until funsor_dist is defined, we save factors for hand-computation in .log_prob().
            self._init = init
            self._trans = trans
            self._obs = obs

        super(GaussianMRF, self).__init__(funsor_dist, batch_shape, event_shape)

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        ndims = max(len(self.batch_shape), value.dim() - 2)
        value = tensor_to_funsor(value, ("time",), 1)

        # Compare with pyro.distributions.hmm.GaussianMRF.log_prob().
        logp_oh = self._trans + self._obs(value=value)
        logp_h = self._trans + self._obs.reduce(ops.logaddexp, "value")
        logp = Stack((logp_oh, logp_h), "_stack")  # TODO add eager_stack_gaussian
        logp = sequential_sum_product(ops.logaddexp, ops.add,
                                      logp, "time", "state", "state(time=1)")
        logp += self._init
        logp = logp.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
        logp_oh, logp_h = logp(_stack=0), logp(_stack=1)
        result = logp_oh - logp_h

        result = funsor_to_tensor(result, ndims=ndims)
        return result

    # TODO remove this once self.funsor_dist is defined.
    def _sample_delta(self, sample_shape):
        raise NotImplementedError("TODO")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DiscreteHMM, _instance)
        batch_shape = torch.Size(batch_shape)
        new._has_rsample = self._has_rsample
        new._init = self._init
        new._trans = self._trans
        new._obs = self._obs
        super(GaussianMRF, new).__init__(self.funsor_dist, batch_shape, self.event_shape)
        return new
