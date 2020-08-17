# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import torch

import funsor.ops as ops
from funsor.domains import Bint, Reals
from funsor.interpreter import interpretation
from funsor.pyro.convert import (
    dist_to_funsor,
    funsor_to_cat_and_mvn,
    funsor_to_tensor,
    matrix_and_mvn_to_funsor,
    mvn_to_funsor,
    tensor_to_funsor
)
from funsor.pyro.distribution import FunsorDistribution
from funsor.sum_product import MarkovProduct, naive_sequential_sum_product, sequential_sum_product
from funsor.terms import Variable, eager, lazy, moment_matching
from funsor.util import broadcast_shape


class DiscreteHMM(FunsorDistribution):
    r"""
    Hidden Markov Model with discrete latent state and arbitrary observation
    distribution. This uses [1] to parallelize over time, achieving
    O(log(time)) parallel complexity.

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_logits`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        # homogeneous + homogeneous case:
        event_shape = (1,) + observation_dist.event_shape

    This class should be interchangeable with
    :class:`pyro.distributions.hmm.DiscreteHMM` .

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :param ~torch.Tensor initial_logits: A logits tensor for an initial
        categorical distribution over latent states. Should have rightmost size
        ``state_dim`` and be broadcastable to ``batch_shape + (state_dim,)``.
    :param ~torch.Tensor transition_logits: A logits tensor for transition
        conditional distributions between latent states. Should have rightmost
        shape ``(state_dim, state_dim)`` (old, new), and be broadcastable to
        ``batch_shape + (num_steps, state_dim, state_dim)``.
    :param ~torch.distributions.Distribution observation_dist: A conditional
        distribution of observed data conditioned on latent state. The
        ``.batch_shape`` should have rightmost size ``state_dim`` and be
        broadcastable to ``batch_shape + (num_steps, state_dim)``. The
        ``.event_shape`` may be arbitrary.
    """
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
        time = Variable("time", Bint[self.event_shape[0]])
        value = tensor_to_funsor(value, ("time",), event_output=self.event_dim - 1,
                                 dtype=self.dtype)

        # Compare with pyro.distributions.hmm.DiscreteHMM.log_prob().
        obs = self._obs(value=value)
        result = self._trans + obs
        result = sequential_sum_product(ops.logaddexp, ops.add,
                                        result, time, {"state": "state(time=1)"})
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
        new._init = self._init + tensor_to_funsor(torch.zeros(batch_shape))
        new._trans = self._trans
        new._obs = self._obs
        super(DiscreteHMM, new).__init__(
            self.funsor_dist, batch_shape, self.event_shape, self.dtype, validate_args=False)
        new.validate_args = self.__dict__.get('_validate_args')
        return new


class GaussianHMM(FunsorDistribution):
    r"""
    Hidden Markov Model with Gaussians for initial, transition, and observation
    distributions. This adapts [1] to parallelize over time to achieve
    O(log(time)) parallel complexity, however it differs in that it tracks the
    log normalizer to ensure :meth:`log_prob` is differentiable.

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

    This class should be compatible with
    :class:`pyro.distributions.hmm.GaussianHMM` , but additionally supports
    funsor :mod:`~funsor.adjoint` algorithms.

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :ivar int hidden_dim: The dimension of the hidden state.
    :ivar int obs_dim: The dimension of the observed state.
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
    :param observation_dist: An observation noise distribution. This should
        have batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(obs_dim,)``.
    :type observation_dist: ~torch.distributions.MultivariateNormal or
        ~torch.distributions.Independent of ~torch.distributions.Normal
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
        assert obs_dim >= hidden_dim // 2, "obs_dim must be at least half of hidden_dim"
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
            value = Variable("value", Reals[time_shape[0], obs_dim])
            result = trans + obs(value=value["time"])
            result = MarkovProduct(ops.logaddexp, ops.add,
                                   result, "time", {"state": "state(time=1)"})
            result = init + result.reduce(ops.logaddexp, "state(time=1)")
            funsor_dist = result.reduce(ops.logaddexp, "state")

        super(GaussianHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim


class GaussianMRF(FunsorDistribution):
    r"""
    Temporal Markov Random Field with Gaussian factors for initial, transition,
    and observation distributions. This adapts [1] to parallelize over time to
    achieve O(log(time)) parallel complexity, however it differs in that it
    tracks the log normalizer to ensure :meth:`log_prob` is differentiable.

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_dist`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        event_shape = (1, obs_dim)  # homogeneous + homogeneous case

    This class should be compatible with
    :class:`pyro.distributions.hmm.GaussianMRF` , but additionally supports
    funsor :mod:`~funsor.adjoint` algorithms.

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :ivar int hidden_dim: The dimension of the hidden state.
    :ivar int obs_dim: The dimension of the observed state.
    :param ~torch.distributions.MultivariateNormal initial_dist: A distribution
        over initial states. This should have batch_shape broadcastable to
        ``self.batch_shape``.  This should have event_shape ``(hidden_dim,)``.
    :param ~torch.distributions.MultivariateNormal transition_dist: A joint
        distribution factor over a pair of successive time steps. This should
        have batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(hidden_dim + hidden_dim,)`` (old+new).
    :param ~torch.distributions.MultivariateNormal observation_dist: A joint
        distribution factor over a hidden and an observed state. This should
        have batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(hidden_dim + obs_dim,)``.
    """
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
                              OrderedDict([("state", Reals[hidden_dim]),
                                           ("state(time=1)", Reals[hidden_dim])]))
        obs = mvn_to_funsor(observation_dist, ("time",),
                            OrderedDict([("state(time=1)", Reals[hidden_dim]),
                                         ("value", Reals[obs_dim])]))

        # Construct the joint funsor.
        # Compare with pyro.distributions.hmm.GaussianMRF.log_prob().
        with interpretation(lazy):
            time = Variable("time", Bint[time_shape[0]])
            value = Variable("value", Reals[time_shape[0], obs_dim])
            logp_oh = trans + obs(value=value["time"])
            logp_oh = MarkovProduct(ops.logaddexp, ops.add,
                                    logp_oh, time, {"state": "state(time=1)"})
            logp_oh += init
            logp_oh = logp_oh.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
            logp_h = trans + obs.reduce(ops.logaddexp, "value")
            logp_h = MarkovProduct(ops.logaddexp, ops.add,
                                   logp_h, time, {"state": "state(time=1)"})
            logp_h += init
            logp_h = logp_h.reduce(ops.logaddexp, frozenset({"state", "state(time=1)"}))
            funsor_dist = logp_oh - logp_h

        dtype = "real"
        super(GaussianMRF, self).__init__(funsor_dist, batch_shape, event_shape, dtype, validate_args)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim


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
    :param bool exact: If True, perform exact inference at cost exponential in
        ``num_steps``. If False, use a :func:`~funsor.terms.moment_matching`
        approximation and use parallel scan algorithm to reduce parallel
        complexity to logarithmic in ``num_steps``. Defaults to False.
    """
    has_rsample = True
    arg_constraints = {}

    def __init__(self, initial_logits, initial_mvn,
                 transition_logits, transition_matrix, transition_mvn,
                 observation_matrix, observation_mvn, exact=False, validate_args=None):
        assert isinstance(initial_logits, torch.Tensor)
        assert isinstance(initial_mvn, torch.distributions.MultivariateNormal)
        assert isinstance(transition_logits, torch.Tensor)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_mvn, torch.distributions.MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_mvn, torch.distributions.MultivariateNormal)
        hidden_cardinality = initial_logits.size(-1)
        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert obs_dim >= hidden_dim // 2, "obs_dim must be at least half of hidden_dim"
        assert initial_mvn.event_shape[0] == hidden_dim
        assert transition_logits.size(-1) == hidden_cardinality
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_mvn.event_shape[0] == hidden_dim
        assert observation_mvn.event_shape[0] == obs_dim
        init_shape = broadcast_shape(initial_logits.shape, initial_mvn.batch_shape)
        shape = broadcast_shape(init_shape[:-1] + (1, init_shape[-1]),
                                transition_logits.shape[:-1],
                                transition_matrix.shape[:-2],
                                transition_mvn.batch_shape,
                                observation_matrix.shape[:-2],
                                observation_mvn.batch_shape)
        assert shape[-1] == hidden_cardinality
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
        if "class(time=1)" not in set(trans.inputs).union(obs.inputs):
            raise ValueError("neither transition nor observation depend on discrete state")
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

        super(SwitchingLinearHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)
        self.exact = exact

    # TODO remove this once self.funsor_dist is defined.
    def log_prob(self, value):
        ndims = max(len(self.batch_shape), value.dim() - 2)
        time = Variable("time", Bint[self.event_shape[0]])
        value = tensor_to_funsor(value, ("time",), 1)

        seq_sum_prod = naive_sequential_sum_product if self.exact else sequential_sum_product
        with interpretation(eager if self.exact else moment_matching):
            result = self._trans + self._obs(value=value)
            result = seq_sum_prod(ops.logaddexp, ops.add, result, time,
                                  {"class": "class(time=1)", "state": "state(time=1)"})
            result += self._init
            result = result.reduce(
                ops.logaddexp, frozenset(["class", "state", "class(time=1)", "state(time=1)"]))

            result = funsor_to_tensor(result, ndims=ndims)
            return result

    # TODO remove this once self.funsor_dist is defined.
    def _sample_delta(self, sample_shape):
        raise NotImplementedError("TODO")

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SwitchingLinearHMM, _instance)
        batch_shape = torch.Size(batch_shape)
        new._init = self._init + tensor_to_funsor(torch.zeros(batch_shape))
        new._trans = self._trans
        new._obs = self._obs
        new.exact = self.exact
        super(SwitchingLinearHMM, new).__init__(
            self.funsor_dist, batch_shape, self.event_shape, self.dtype, validate_args=False)
        new.validate_args = self.__dict__.get('_validate_args')
        return new

    def filter(self, value):
        """
        Compute posterior over final state given a sequence of observations.

        :param ~torch.Tensor value: A sequence of observations.
        :return: A posterior distribution over latent states at the final time
            step, represented as a pair ``(cat, mvn)``, where
            :class:`~pyro.distributions.Categorical` distribution over mixture
            components and ``mvn`` is a
            :class:`~pyro.distributions.MultivariateNormal` with rightmost
            batch dimension ranging over mixture components. This can then be
            used to initialize a sequential Pyro model for prediction.
        :rtype: tuple
        """
        ndims = max(len(self.batch_shape), value.dim() - 2)
        time = Variable("time", Bint[self.event_shape[0]])
        value = tensor_to_funsor(value, ("time",), 1)

        seq_sum_prod = naive_sequential_sum_product if self.exact else sequential_sum_product
        with interpretation(eager if self.exact else moment_matching):
            logp = self._trans + self._obs(value=value)
            logp = seq_sum_prod(ops.logaddexp, ops.add, logp, time,
                                {"class": "class(time=1)", "state": "state(time=1)"})
            logp += self._init
            logp = logp.reduce(ops.logaddexp, frozenset(["class", "state"]))

        cat, mvn = funsor_to_cat_and_mvn(logp, ndims, ("class(time=1)",))
        cat = cat.expand(self.batch_shape)
        mvn = mvn.expand(self.batch_shape + cat.logits.shape[-1:])
        return cat, mvn
