# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpyro.distributions as dist
from pyro.distributions.util import broadcast_shape

import funsor.ops as ops
from funsor.domains import bint, reals
from funsor.interpreter import interpretation
from funsor.numpyro.convert import (
    dist_to_funsor,
    funsor_to_cat_and_mvn,
    funsor_to_tensor,
    matrix_and_mvn_to_funsor,
    mvn_to_funsor,
    tensor_to_funsor
)
from funsor.numpy import array
from funsor.numpyro.distribution import FunsorDistribution
from funsor.sum_product import MarkovProduct, naive_sequential_sum_product, sequential_sum_product
from funsor.terms import Variable, eager, lazy, moment_matching


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
        assert isinstance(initial_dist, dist.MultivariateNormal)
        assert isinstance(transition_matrix, array)
        assert isinstance(transition_dist, dist.MultivariateNormal)
        assert isinstance(observation_matrix, array)
        assert isinstance(observation_dist, dist.MultivariateNormal)
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
            value = Variable("value", reals(time_shape[0], obs_dim))
            result = trans + obs(value=value["time"])
            result = MarkovProduct(ops.logaddexp, ops.add,
                                   result, "time", {"state": "state(time=1)"})
            result = init + result.reduce(ops.logaddexp, "state(time=1)")
            funsor_dist = result.reduce(ops.logaddexp, "state")

        super(GaussianHMM, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
