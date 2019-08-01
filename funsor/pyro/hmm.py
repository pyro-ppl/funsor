from collections import OrderedDict

import torch
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
from funsor.domains import reals
from funsor.interpreter import interpretation
from funsor.pyro.convert import (
    dist_to_funsor,
    funsor_to_tensor,
    matrix_and_mvn_to_funsor,
    mvn_to_funsor,
    tensor_to_funsor
)
from funsor.pyro.distribution import FunsorDistribution
from funsor.sum_product import sequential_sum_product
from funsor.terms import Variable, lazy, moment_matching


class DiscreteHMM(FunsorDistribution):
    def __init__(self, initial_logits, transition_logits, observation_dist, validate_args=None):
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

        super(DiscreteHMM, self).__init__(funsor_dist, batch_shape, event_shape, dtype, validate_args)

    @torch.distributions.constraints.dependent_property
    def has_rsample(self):
        return self._has_rsample

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
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

    def __init__(self, initial_dist, transition_dist, observation_dist, validate_args=None):
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
        trans = mvn_to_funsor(transition_dist, ("time",),
                              OrderedDict([("state", reals(hidden_dim)),
                                           ("state(time=1)", reals(hidden_dim))]))
        obs = mvn_to_funsor(observation_dist, ("time",),
                            OrderedDict([("state(time=1)", reals(hidden_dim)),
                                         ("value", reals(obs_dim))]))

        # Construct the joint funsor.
        with interpretation(lazy):
            # TODO perform math here once sequential_sum_product has been
            #   implemented as a first-class funsor.
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            # Until funsor_dist is defined, we save factors for hand-computation in .log_prob().
            self._init = init
            self._trans = trans
            self._obs = obs

        dtype = "real"
        super(GaussianMRF, self).__init__(funsor_dist, batch_shape, event_shape, dtype, validate_args)

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ndims = max(len(self.batch_shape), value.dim() - 2)
        value = tensor_to_funsor(value, ("time",), 1)

        # Compare with pyro.distributions.hmm.GaussianMRF.log_prob().
        logp_oh = self._trans + self._obs(value=value)
        logp_oh = sequential_sum_product(ops.logaddexp, ops.add,
                                         logp_oh, "time", "state", "state(time=1)")
        logp_oh += self._init
        logp_oh = logp_oh.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
        logp_h = self._trans + self._obs.reduce(ops.logaddexp, "value")
        logp_h = sequential_sum_product(ops.logaddexp, ops.add,
                                        logp_h, "time", "state", "state(time=1)")
        logp_h += self._init
        logp_h = logp_h.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
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


class GaussianDiscreteMRF(FunsorDistribution):
    """
    Temporal Markov Random Field with Gaussian latent state and discrete
    observations, where observation factors are mixtures-of-gaussians.

    This distribution uses the :func:`~funsor.terms.moment_matching`
    approximation to perform inference in the following Pyro model, where
    ``state`` be the latent multivariate normal state, ``prediction`` is the
    predicted future state, and ``obs`` is the discrete observation::

        def model(data):
            state = pyro.sample("state_0", initial_dist)
            for t in range(len(data)):
                noise = pyro.sample("noise_{}".format(t + 1), transition_dist)
                state = state @ transition_matrix[t] + noise
                obs = pyro.sample("obs_{}".format(t + 1),
                                  dist.Categorical(observation_logits),
                                  obs=data[t])
                pyro.sample("state_{}".format(t + 1), observation_dist[t, obs],
                            obs=state)

    :param ~torch.distributions.MultivariateNormal initial_dist: Represents
        ``p(state[0])``.
    :param ~torch.Tensor transition_matrix: Transforms ``state[t]`` to
        ``prediction[t+1]``.
    :param ~torch.distributions.MultivariateNormal transition_dist: Represents
        ``p(state[t+1] | prediction[t+1]) = p(noise[t+1])`` where
        ``noise = state - prediction``.
    :param ~torch.Tensor observation_logits: Represents ``p(obs[t+1])``.
    :param ~torch.distributions.MultivariateNormal observation_dist: Represents
        ``p(state[t+1] | obs[t+1])``.
    """
    def __init__(self, initial_dist, transition_matrix, transition_dist,
                 observation_logits, observation_dist, validate_args=None):
        assert isinstance(initial_dist, torch.distributions.MultivariateNormal)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_dist, torch.distributions.MultivariateNormal)
        assert isinstance(observation_logits, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.MultivariateNormal)
        hidden_dim = initial_dist.event_shape[0]
        obs_dim = observation_logits.size(-1)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape[0] == hidden_dim
        assert observation_dist.shape()[-2:] == (obs_dim, hidden_dim)
        shape = broadcast_shape(initial_dist.batch_shape + (1,),
                                transition_matrix.shape[:-2],
                                transition_dist.batch_shape,
                                observation_logits.shape[:-1],
                                observation_dist.batch_shape[:-1])
        batch_shape, event_shape = shape[:-1], shape[-1:]

        # Convert distributions to funsors.
        init = dist_to_funsor(initial_dist)(value="state")
        trans = matrix_and_mvn_to_funsor(transition_matrix, transition_dist,
                                         ("time",), "state", "state(time=1)")
        obs_mvn = dist_to_funsor(observation_dist, ("time", "_value"))
        obs = (obs_mvn(value="state(time=1)")(_value="value") +
               tensor_to_funsor(observation_logits, ("time", "value")))
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

        super(GaussianDiscreteMRF, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ndims = max(len(self.batch_shape), value.dim() - 1)
        value = tensor_to_funsor(value, ("time",), dtype=self.dtype)

        with interpretation(moment_matching):
            logp_oh = self._trans + self._obs(value=value)
            logp_oh = sequential_sum_product(ops.logaddexp, ops.add,
                                             logp_oh, "time", "state", "state(time=1)")
            logp_oh += self._init
            logp_oh = logp_oh.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
            logp_h = self._trans + self._obs.reduce(ops.logaddexp, "value")
            logp_h = sequential_sum_product(ops.logaddexp, ops.add,
                                            logp_h, "time", "state", "state(time=1)")
            logp_h += self._init
            logp_h = logp_h.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
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
        super(GaussianDiscreteMRF, new).__init__(self.funsor_dist, batch_shape, self.event_shape)
        return new
