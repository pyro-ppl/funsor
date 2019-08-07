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


class GaussianHMM(FunsorDistribution):
    """
    Hidden Markov Model with Gaussians for initial, transition, and observation
    distributions.

    This corresponds to the generative model::

        z = initial_distribution.sample()
        x = []
        for t in range(num_steps):
            z = z @ transition_matrix + transition_dist.sample()
            x.append(z @ observation_matrix + observation_dist.sample())

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_dist`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        event_shape = (1, obs_dim)  # homogeneous + homogeneous case

    :param ~torch.distributions.MultivariateNormal initial_dist: A distribution
        over initial states. This should have batch_shape broadcastable to
        ``self.batch_shape``.  This should have event_shape ``(hidden_dim,)``.
    :param ~torch.Tensor transition_matrix: A linear transformation of hidden
        state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, hidden_dim)`` where the
        rightmost dims are ordered ``(old, new)``.
    :param ~torch.distributions.MultivariateNormal transition_dist: A process
        noise distribution. This should have batch_shape broadcastable to
        ``self.batch_shape + (num_steps,)``.  This should have event_shape
        ``(hidden_dim,)``.
    :param ~torch.Tensor transition_matrix: A linear transformation from hidden
        to observed state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, obs_dim)``.
    :param ~torch.distributions.MultivariateNormal observation_dist: An
        observation noise distribution. This should have batch_shape
        broadcastable to ``self.batch_shape + (num_steps,)``.  This should have
        event_shape ``(obs_dim,)``.
    """
    has_rsample = True
    arg_constraints = {}

    def __init__(self, initial_dist, transition_matrix, transition_dist,
                 observation_matrix, observation_dist, validate_args=None):
        assert isinstance(initial_dist, torch.distributions.MultivariateNormal)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_dist, torch.distributions.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_dist, torch.distributions.MultivariateNormal)
        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert initial_dist.event_shape == (hidden_dim,)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape == (hidden_dim,)
        assert observation_dist.event_shape == (obs_dim,)
        shape = broadcast_shape(initial_dist.batch_shape + (1,),
                                transition_matrix.shape[:-2],
                                transition_dist.batch_shape,
                                observation_matrix.shape[:-2],
                                observation_dist.batch_shape)
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)

        # Convert distributions to funsors.
        init = dist_to_funsor(initial_dist)(value="state")
        trans = matrix_and_mvn_to_funsor(transition_matrix, transition_dist,
                                         ("time",), "state", "state(time=1)")
        obs = matrix_and_mvn_to_funsor(observation_matrix, observation_dist,
                                       ("time",), "state(time=1)", "value")
        dtype = "real"

        # Construct the joint funsor.
        with interpretation(lazy):
            # TODO perform math here once sequential_sum_product has been
            #   implemented as a first-class funsor.
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            # Until funsor_dist is defined, we save factors for hand-computation in .log_prob().
            self._init = init
            self._trans = trans
            self._obs = obs

        super(GaussianHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianHMM, _instance)
        batch_shape = torch.Size(broadcast_shape(self.batch_shape, batch_shape))
        # We only need to expand one of the inputs, since batch_shape is determined
        # by broadcasting all three. To save computation in _sequential_gaussian_tensordot(),
        # we expand only _init, which is applied only after _sequential_gaussian_tensordot().
        new._init = self._init.expand(batch_shape)
        new._trans = self._trans
        new._obs = self._obs
        super(GaussianHMM, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new.validate_args = self.__dict__.get('_validate_args')
        return new

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ndims = max(len(self.batch_shape), value.dim() - self.event_dim)
        value = tensor_to_funsor(value, ("time",), event_output=self.event_dim - 1,
                                 dtype=self.dtype)

        # Compare with pyro.distributions.hmm.GaussianHMM.log_prob().
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
        new._init = self._init
        new._trans = self._trans
        new._obs = self._obs
        super(GaussianMRF, new).__init__(self.funsor_dist, batch_shape, self.event_shape)
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
        new._init = self._init
        new._trans = self._trans
        new._obs = self._obs
        super(GaussianMRF, new).__init__(self.funsor_dist, batch_shape, self.event_shape)
        return new


class SwitchingLinearHMM(FunsorDistribution):
    r"""
    Switching Linear Dynamical System represented as a Hidden Markov Model.

    This corresponds to the generative model::

        z = Categorical(logits=initial_logits).sample()
        y = initial_mvn[z].sample()
        x = []
        for t in range(num_steps):
            z = Categorical(logits=transition_logits[t, z]).sample()
            y = y @ transition_matrix[t, z] + transition_mvn[t, z].sample()
            x.append(y @ observation_matrix[t, z] + observation_mvn[t, z].sample())

    Viewed as a dynamic Bayesian network::

        z[t-1] ----> z[t] ---> z[t+1]         Discrete latent class
           |  \       |  \       |   \
           | y[t-1] ----> y[t] ----> y[t+1]   Gaussian latent state
           |   /      |   /      |   /
           V  /       V  /       V  /
        x[t-1]       x[t]      x[t+1]         Gaussian observation

    This uses the :func:`~funsor.terms.moment_matching` approximation.

    Let ``class`` be the latent class, ``state`` be the latent multivariate
    normal state, and ``value`` be the observed multivariate normal value.

    :param ~torch.Tensor initial_logits: Represents ``p(class[0])``.
    :param ~torch.distributions.MultivariateNormal initial_mvn: Represents
        ``p(state[0] | class[0])``.
    :param ~torch.Tensor transition_logits: Represents
        ``p(class[t+1] | class[t])``.
    :param ~torch.Tensor transition_matrix:
    :param ~torch.distributions.MultivariateNormal transition_mvn: Together
        with ``transition_matrix``, this represents
        ``p(state[t], state[t+1] | class[t])``.
    :param ~torch.Tensor observation_matrix:
    :param ~torch.distributions.MultivariateNormal observation_mvn: Together
        with ``observation_matrix``, this represents
        ``p(value[t+1], state[t+1] | class[t+1])``.
    """
    has_rsample = True
    arg_constraints = {}

    def __init__(self, initial_logits, initial_mvn,
                 transition_logits, transition_matrix, transition_mvn,
                 observation_matrix, observation_mvn, validate_args=None):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(initial_mvn, torch.distributions.MultivariateNormal)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_mvn, torch.distributions.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_mvn, torch.distributions.MultivariateNormal)
        hidden_cardinality = initial_logits.size(-1)
        hidden_dim = initial_mvn.event_shape[0]
        obs_dim = observation_mvn.event_shape[0]
        assert transition_logits.size(-1) == hidden_cardinality
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_mvn.event_shape[0] == hidden_dim
        init_shape = broadcast_shape(initial_logits.shape, initial_mvn.batch_shape)
        shape = broadcast_shape(init_shape[:-1] + (1, init_shape[-1]),
                                transition_logits.shape[:-1],
                                transition_matrix.shape[:-2],
                                transition_mvn.batch_shape,
                                observation_matrix.shape[:-2],
                                observation_mvn.batch_shape)
        batch_shape, time_shape = shape[:-2], shape[-2:-1]
        event_shape = time_shape + (obs_dim,)

        # Normalize.
        initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        transition_logits = transition_logits - transition_logits.logsumexp(-1, True)

        # Convert tensors and distributions to funsors.
        init = (tensor_to_funsor(initial_logits, ("class",)) +
                dist_to_funsor(initial_mvn, ("class",))(value="state"))
        trans = (tensor_to_funsor(transition_logits, ("time", "class", "class(time=1)")) +
                 matrix_and_mvn_to_funsor(transition_matrix, transition_mvn,
                                          ("time", "class(time=1)"), "state", "state(time=1)"))
        obs = matrix_and_mvn_to_funsor(observation_matrix, observation_mvn,
                                       ("time", "class(time=1)"), "state(time=1)", "value")
        dtype = "real"
        if "class" not in trans.inputs or "class(time=1)" not in set(trans.inputs).union(obs.inputs):
            raise ValueError("neither transition nor observation depend on discrete state")

        # Construct the joint funsor.
        with interpretation(lazy):
            # TODO perform math here once sequential_sum_product has been
            #   implemented as a first-class funsor.
            funsor_dist = Variable("value", obs.inputs["value"])  # a bogus value
            # Until funsor_dist is defined, we save factors for hand-computation in .log_prob().
            self._init = init
            self._trans = trans
            self._obs = obs

        super(SwitchingLinearHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        ndims = max(len(self.batch_shape), value.dim() - 2)
        value = tensor_to_funsor(value, ("time",), 1)

        with interpretation(moment_matching):
            result = self._trans + self._obs(value=value)
            result = sequential_sum_product(ops.logaddexp, ops.add, result, "time",
                                            ("class", "state"), ("class(time=1)", "state(time=1)"))
            result = result.reduce(ops.logaddexp, frozenset(["class(time=1)", "state(time=1)"]))
            result += self._init
            result = result.reduce(ops.logaddexp, frozenset(["class", "state"]))

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
        super(SwitchingLinearHMM, new).__init__(self.funsor_dist, batch_shape, self.event_shape)
        return new
