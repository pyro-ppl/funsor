# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpyro.distributions as dist
import pytest

import funsor.ops as ops
from funsor.numpyro.hmm import DiscreteHMM, GaussianHMM, GaussianMRF, SwitchingLinearHMM
from funsor.tensor import numeric_array
from funsor.testing import assert_close, empty, zeros, rand, randn, random_mvn
from funsor.util import broadcast_shape


def get_rng_key(seed=0):
    return np.array([0, seed], dtype=np.uint32)


def get_unexpanded_dist(d):
    if isinstance(d, dist.ExpandedDistribution):
        return get_unexpanded_dist(d.base_dist)
    return d


def check_expand(old_dist, old_data):
    new_batch_shape = (2,) + old_dist.batch_shape
    new_dist = old_dist.expand(new_batch_shape)
    assert new_dist.batch_shape == new_batch_shape

    old_log_prob = new_dist.log_prob(old_data)
    assert old_log_prob.shape == new_batch_shape

    new_data = ops.expand(old_data, new_batch_shape + new_dist.event_shape)
    new_log_prob = new_dist.log_prob(new_data)
    assert_close(old_log_prob, new_log_prob)
    assert new_dist.log_prob(new_data).shape == new_batch_shape


DISCRETE_HMM_SHAPES = [
    # init_shape, trans_shape, obs_shape
    ((), (1,), ()),
    ((), (), (1,)),
    ((), (2,), ()),
    ((), (7,), ()),
    ((), (), (7,)),
    ((), (7,), (1,)),
    ((), (1,), (7,)),
    ((), (7,), (5, 7)),
    ((), (5, 7), (7,)),
    ((), (5, 7), (5, 7)),
    ((5,), (7,), (7,)),
    ((5,), (7,), (5, 7)),
    ((5,), (5, 7), (7,)),
    ((5,), (5, 7), (5, 7)),
    ((4, 1, 1), (3, 1, 7), (2, 7)),
]


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_categorical_log_prob_smoke(init_shape, trans_shape, obs_shape, state_dim):
    obs_dim = 4
    init_logits = randn(init_shape + (state_dim,))
    trans_logits = randn(trans_shape + (state_dim, state_dim))
    obs_logits = randn(obs_shape + (state_dim, obs_dim))
    obs_dist = dist.Categorical(logits=obs_logits)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample(get_rng_key())
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_dist.log_prob(data)
    check_expand(actual_dist, data)


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_normal_log_prob_smoke(init_shape, trans_shape, obs_shape, state_dim):
    init_logits = randn(init_shape + (state_dim,))
    trans_logits = randn(trans_shape + (state_dim, state_dim))
    loc = randn(obs_shape + (state_dim,))
    scale = ops.exp(randn(obs_shape + (state_dim,)))
    obs_dist = dist.Normal(loc, scale)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample(get_rng_key())
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_dist.log_prob(data)
    check_expand(actual_dist, data)


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_mvn_log_prob_smoke(init_shape, trans_shape, obs_shape, state_dim):
    event_size = 4
    init_logits = randn(init_shape + (state_dim,))
    trans_logits = randn(trans_shape + (state_dim, state_dim))
    loc = randn(obs_shape + (state_dim, event_size))
    cov = randn(obs_shape + (state_dim, event_size, 2 * event_size))
    cov = ops.matmul(cov, ops.transpose(cov, -1, -2))
    scale_tril = ops.cholesky(cov)
    obs_dist = dist.MultivariateNormal(loc, scale_tril=scale_tril)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample(get_rng_key())
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_dist.log_prob(data)
    check_expand(actual_dist, data)


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_diag_normal_log_prob_smoke(init_shape, trans_shape, obs_shape, state_dim):
    event_size = 4
    init_logits = randn(init_shape + (state_dim,))
    trans_logits = randn(trans_shape + (state_dim, state_dim))
    loc = randn(obs_shape + (state_dim, event_size))
    scale = ops.exp(randn(obs_shape + (state_dim, event_size)))
    obs_dist = dist.Normal(loc, scale).to_event(1)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample(get_rng_key())
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_dist.log_prob(data)
    check_expand(actual_dist, data)


@pytest.mark.parametrize("obs_dim,hidden_dim",
                         [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("init_shape,trans_mat_shape,trans_mvn_shape,obs_mat_shape,obs_mvn_shape", [
    ((), (), (), (), ()),
    ((), (6,), (), (), ()),
    ((), (), (6,), (), ()),
    ((), (), (), (6,), ()),
    ((), (), (), (), (6,)),
    ((), (6,), (6,), (6,), (6,)),
    ((5,), (6,), (), (), ()),
    ((), (5, 1), (6,), (), ()),
    ((), (), (5, 1), (6,), ()),
    ((), (), (), (5, 1), (6,)),
    ((), (6,), (5, 1), (), ()),
    ((), (), (6,), (5, 1), ()),
    ((), (), (), (6,), (5, 1)),
    ((5,), (), (), (), (6,)),
    ((5,), (5, 6), (5, 6), (5, 6), (5, 6)),
], ids=str)
def test_gaussian_hmm_log_prob_smoke(init_shape, trans_mat_shape, trans_mvn_shape,
                                     obs_mat_shape, obs_mvn_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_mat = randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(trans_mvn_shape, hidden_dim)
    obs_mat = randn(obs_mat_shape + (hidden_dim, obs_dim))
    obs_dist = random_mvn(obs_mvn_shape, obs_dim)

    actual_dist = GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    shape = broadcast_shape(init_shape + (1,),
                            trans_mat_shape, trans_mvn_shape,
                            obs_mat_shape, obs_mvn_shape)
    data = obs_dist.expand(shape).sample(get_rng_key())
    assert data.shape == actual_dist.batch_shape + actual_dist.event_shape

    actual_dist.log_prob(data)
    check_expand(actual_dist, data)


@pytest.mark.parametrize("obs_dim", [1, 2, 3])
@pytest.mark.parametrize("hidden_dim", [1, 2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_gaussian_mrf_log_prob_smoke(init_shape, trans_shape, obs_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_dist = random_mvn(trans_shape, hidden_dim + hidden_dim)
    obs_dist = random_mvn(obs_shape, hidden_dim + obs_dim)

    actual_dist = GaussianMRF(init_dist, trans_dist, obs_dist)

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape).sample(get_rng_key())[..., hidden_dim:]
    actual_dist.log_prob(data)
    check_expand(actual_dist, data)


SLHMM_SCHEMA = ",".join([
    "init_cat_shape", "init_mvn_shape",
    "trans_cat_shape", "trans_mat_shape", "trans_mvn_shape",
    "obs_mat_shape", "obs_mvn_shape",
])
SLHMM_SHAPES = [
    ((2,), (), (1, 2,), (1, 3, 3), (1,), (1, 3, 4), (1,)),
    ((2,), (), (5, 1, 2,), (1, 3, 3), (1,), (1, 3, 4), (1,)),
    ((2,), (), (1, 2,), (5, 1, 3, 3), (1,), (1, 3, 4), (1,)),
    ((2,), (), (1, 2,), (1, 3, 3), (5, 1), (1, 3, 4), (1,)),
    ((2,), (), (1, 2,), (1, 3, 3), (1,), (5, 1, 3, 4), (1,)),
    ((2,), (), (1, 2,), (1, 3, 3), (1,), (1, 3, 4), (5, 1)),
    ((2,), (), (5, 1, 2,), (5, 1, 3, 3), (5, 1), (5, 1, 3, 4), (5, 1)),
    ((2,), (2,), (5, 2, 2,), (5, 2, 3, 3), (5, 2), (5, 2, 3, 4), (5, 2)),
    ((7, 2,), (), (7, 5, 1, 2,), (7, 5, 1, 3, 3), (7, 5, 1), (7, 5, 1, 3, 4), (7, 5, 1)),
    ((7, 2,), (7, 2), (7, 5, 2, 2,), (7, 5, 2, 3, 3), (7, 5, 2), (7, 5, 2, 3, 4), (7, 5, 2)),
]


@pytest.mark.parametrize(SLHMM_SCHEMA, SLHMM_SHAPES, ids=str)
def test_switching_linear_hmm_shape(init_cat_shape, init_mvn_shape,
                                    trans_cat_shape, trans_mat_shape, trans_mvn_shape,
                                    obs_mat_shape, obs_mvn_shape):
    hidden_dim, obs_dim = obs_mat_shape[-2:]
    assert trans_mat_shape[-2:] == (hidden_dim, hidden_dim)

    init_logits = randn(init_cat_shape)
    init_mvn = random_mvn(init_mvn_shape, hidden_dim)
    trans_logits = randn(trans_cat_shape)
    trans_matrix = randn(trans_mat_shape)
    trans_mvn = random_mvn(trans_mvn_shape, hidden_dim)
    obs_matrix = randn(obs_mat_shape)
    obs_mvn = random_mvn(obs_mvn_shape, obs_dim)

    init_shape = broadcast_shape(init_cat_shape, init_mvn_shape)
    shape = broadcast_shape(init_shape[:-1] + (1, init_shape[-1]),
                            trans_cat_shape[:-1],
                            trans_mat_shape[:-2],
                            trans_mvn_shape,
                            obs_mat_shape[:-2],
                            obs_mvn_shape)
    expected_batch_shape, time_shape = shape[:-2], shape[-2:-1]
    expected_event_shape = time_shape + (obs_dim,)

    actual_dist = SwitchingLinearHMM(init_logits, init_mvn,
                                     trans_logits, trans_matrix, trans_mvn,
                                     obs_matrix, obs_mvn)
    assert actual_dist.event_shape == expected_event_shape
    assert actual_dist.batch_shape == expected_batch_shape

    data = obs_mvn.expand(shape).sample(get_rng_key())[..., 0, :]
    actual_log_prob = actual_dist.log_prob(data)
    assert actual_log_prob.shape == expected_batch_shape
    check_expand(actual_dist, data)

    final_cat, final_mvn = actual_dist.filter(data)
    final_cat_base_dist = get_unexpanded_dist(final_cat)
    final_mvn_base_dist = get_unexpanded_dist(final_mvn)
    assert isinstance(final_cat_base_dist, dist.CategoricalLogits)
    assert isinstance(final_mvn_base_dist, dist.MultivariateNormal)
    assert final_cat.batch_shape == actual_dist.batch_shape
    assert final_mvn.batch_shape == actual_dist.batch_shape + final_cat_base_dist.logits.shape[-1:]


@pytest.mark.parametrize("num_components", [2, 3])
@pytest.mark.parametrize("obs_dim,hidden_dim",
                         [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "approx"])
def test_switching_linear_hmm_log_prob(exact, num_steps, hidden_dim, obs_dim, num_components):
    # This tests agreement between an SLDS and an HMM when all components
    # are identical, i.e. so latent can be marginalized out.
    init_logits = rand(num_components)
    init_mvn = random_mvn((), hidden_dim)
    trans_logits = rand(num_components)
    trans_matrix = randn(hidden_dim, hidden_dim)
    trans_mvn = random_mvn((), hidden_dim)
    obs_matrix = randn(hidden_dim, obs_dim)
    obs_mvn = random_mvn((), obs_dim)

    expected_dist = GaussianHMM(init_mvn,
                                ops.expand(trans_matrix, (num_steps, -1, -1)),
                                trans_mvn, obs_matrix, obs_mvn)
    actual_dist = SwitchingLinearHMM(init_logits, init_mvn, trans_logits,
                                     ops.expand(trans_matrix, (num_steps, num_components, -1, -1)),
                                     trans_mvn, obs_matrix, obs_mvn,
                                     exact=exact)
    assert actual_dist.batch_shape == expected_dist.batch_shape
    assert actual_dist.event_shape == expected_dist.event_shape

    data = obs_mvn.sample(get_rng_key(), expected_dist.batch_shape + (num_steps,))
    assert data.shape == expected_dist.batch_shape + expected_dist.event_shape
    expected_log_prob = expected_dist.log_prob(data)
    assert expected_log_prob.shape == expected_dist.batch_shape
    actual_log_prob = actual_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=None)


@pytest.mark.parametrize("num_steps", [2, 3])
@pytest.mark.parametrize("num_components", [2, 3])
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "approx"])
def test_switching_linear_hmm_log_prob_alternating(exact, num_steps, num_components):
    # This tests agreement between an SLDS and an HMM in the case that the two
    # SLDS discrete states alternate back and forth between 0 and 1 deterministically

    hidden_dim = 4
    obs_dim = 3
    extra_components = num_components - 2

    init_logits = numeric_array([float("-inf"), 0.0] + extra_components * [float("-inf")])
    init_mvn = random_mvn((num_components,), hidden_dim)

    left_logits = numeric_array([0.0, float("-inf")] + extra_components * [float("-inf")])
    right_logits = numeric_array([float("-inf"), 0.0] + extra_components * [float("-inf")])
    trans_logits = numeric_array([left_logits if t % 2 == 0 else right_logits for t in range(num_steps)])
    trans_logits = ops.unsqueeze(trans_logits, -2)

    hmm_trans_matrix = randn(num_steps, hidden_dim, hidden_dim)
    switching_trans_matrix = ops.expand(ops.unsqueeze(hmm_trans_matrix, -3), (-1, num_components, -1, -1))
    switching_trans_matrix = np.array(switching_trans_matrix)

    trans_mvn = random_mvn((num_steps, num_components,), hidden_dim)
    hmm_obs_matrix = randn(num_steps, hidden_dim, obs_dim)
    switching_obs_matrix = ops.expand(ops.unsqueeze(hmm_obs_matrix, -3), (-1, num_components, -1, -1))
    switching_obs_matrix = np.array(switching_obs_matrix)
    obs_mvn = random_mvn((num_steps, num_components), obs_dim)

    hmm_trans_mvn_loc = empty(num_steps, hidden_dim)
    hmm_trans_mvn_cov = empty(num_steps, hidden_dim, hidden_dim)
    hmm_obs_mvn_loc = empty(num_steps, obs_dim)
    hmm_obs_mvn_cov = empty(num_steps, obs_dim, obs_dim)

    for t in range(num_steps):
        # select relevant bits for hmm given deterministic dynamics in discrete space
        s = t % 2  # 0, 1, 0, 1, ...
        hmm_trans_mvn_loc[t] = trans_mvn.loc[t, s]
        hmm_trans_mvn_cov[t] = trans_mvn.covariance_matrix[t, s]
        hmm_obs_mvn_loc[t] = obs_mvn.loc[t, s]
        hmm_obs_mvn_cov[t] = obs_mvn.covariance_matrix[t, s]

        # scramble matrices in places that should never be accessed given deterministic dynamics in discrete space
        s = 1 - (t % 2)  # 1, 0, 1, 0, ...
        switching_trans_matrix[t, s, :, :] = rand(hidden_dim, hidden_dim)
        switching_obs_matrix[t, s, :, :] = rand(hidden_dim, obs_dim)

    expected_dist = GaussianHMM(dist.MultivariateNormal(init_mvn.loc[1], init_mvn.covariance_matrix[1]),
                                hmm_trans_matrix,
                                dist.MultivariateNormal(hmm_trans_mvn_loc, hmm_trans_mvn_cov),
                                hmm_obs_matrix,
                                dist.MultivariateNormal(hmm_obs_mvn_loc, hmm_obs_mvn_cov))

    actual_dist = SwitchingLinearHMM(init_logits, init_mvn, trans_logits,
                                     switching_trans_matrix,
                                     trans_mvn, switching_obs_matrix, obs_mvn,
                                     exact=exact)

    assert actual_dist.batch_shape == expected_dist.batch_shape
    assert actual_dist.event_shape == expected_dist.event_shape

    data = obs_mvn.sample(get_rng_key())[:, 0, :]
    assert data.shape == expected_dist.batch_shape + expected_dist.event_shape
    expected_log_prob = expected_dist.log_prob(data)
    assert expected_log_prob.shape == expected_dist.batch_shape
    actual_log_prob = actual_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-2, rtol=None)


@pytest.mark.parametrize("hidden_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_mat_shape,trans_mvn_shape,obs_mvn_shape", [
    ((), (), (), ()),
    ((), (6,), (), ()),
    ((), (), (6,), ()),
    ((), (), (), ()),
    ((), (), (), (6,)),
    ((5,), (6,), (), ()),
    ((), (5, 1), (6,), ()),
    ((), (), (), (6,)),
    ((), (6,), (5, 1), ()),
    ((5,), (), (), (6,)),
], ids=str)
def test_gaussian_hmm_log_prob_null_dynamics(init_shape, trans_mat_shape, trans_mvn_shape,
                                             obs_mvn_shape, hidden_dim):
    obs_dim = hidden_dim
    init_dist = random_mvn(init_shape, hidden_dim)

    # impose null dynamics
    trans_mat = zeros(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(trans_mvn_shape, hidden_dim, diag=True)

    # trivial observation matrix (hidden_dim = obs_dim)
    obs_mat = ops.new_eye(trans_mat, (hidden_dim,))
    obs_dist = random_mvn(obs_mvn_shape, obs_dim, diag=True)

    actual_dist = GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    shape = broadcast_shape(init_shape + (1,),
                            trans_mat_shape, trans_mvn_shape,
                            obs_mvn_shape)
    data = obs_dist.expand(shape).sample(get_rng_key())
    assert data.shape == actual_dist.batch_shape + actual_dist.event_shape

    actual_log_prob = actual_dist.log_prob(data)

    check_expand(actual_dist, data)

    obs_cov = ops.diagonal(obs_dist.covariance_matrix, -1, -2)
    trans_cov = ops.diagonal(trans_dist.covariance_matrix, -1, -2)
    sum_scale = ops.sqrt(obs_cov + trans_cov)
    sum_loc = trans_dist.loc + obs_dist.loc

    analytic_log_prob = dist.Normal(sum_loc, sum_scale).log_prob(data).sum(-1).sum(-1)
    assert_close(analytic_log_prob, actual_log_prob, atol=1.0e-5)
