import warnings

import pyro.distributions as dist
import pytest
import torch
from pyro.distributions.util import broadcast_shape

from funsor.pyro.hmm import DiscreteHMM, GaussianHMM, GaussianMRF, SwitchingLinearHMM
from funsor.testing import assert_close, random_mvn

DISCRETE_HMM_SHAPES = [
    # init_shape, trans_shape, obs_shape
    ((), (1,), ()),
    ((), (), (1,)),
    ((), (2,), ()),
    ((), (7,), ()),
    ((), (), (7,)),
    ((), (7,), (1,)),
    ((), (1,), (7,)),
    ((), (7,), (11, 7)),
    ((), (11, 7), (7,)),
    ((), (11, 7), (11, 7)),
    ((11,), (7,), (7,)),
    ((11,), (7,), (11, 7)),
    ((11,), (11, 7), (7,)),
    ((11,), (11, 7), (11, 7)),
    ((4, 1, 1), (3, 1, 7), (2, 7)),
]


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_categorical_log_prob(init_shape, trans_shape, obs_shape, state_dim):
    obs_dim = 4
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    obs_logits = torch.randn(obs_shape + (state_dim, obs_dim))
    obs_dist = dist.Categorical(logits=obs_logits)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)
    expected_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    assert actual_dist.event_shape == expected_dist.event_shape
    assert actual_dist.batch_shape == expected_dist.batch_shape

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample()
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_normal_log_prob(init_shape, trans_shape, obs_shape, state_dim):
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    loc = torch.randn(obs_shape + (state_dim,))
    scale = torch.randn(obs_shape + (state_dim,)).exp()
    obs_dist = dist.Normal(loc, scale)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)
    expected_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    assert actual_dist.event_shape == expected_dist.event_shape
    assert actual_dist.batch_shape == expected_dist.batch_shape

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample()
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_mvn_log_prob(init_shape, trans_shape, obs_shape, state_dim):
    event_size = 4
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    loc = torch.randn(obs_shape + (state_dim, event_size))
    cov = torch.randn(obs_shape + (state_dim, event_size, 2 * event_size))
    cov = cov.matmul(cov.transpose(-1, -2))
    scale_tril = torch.cholesky(cov)
    obs_dist = dist.MultivariateNormal(loc, scale_tril=scale_tril)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)
    expected_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    assert actual_dist.event_shape == expected_dist.event_shape
    assert actual_dist.batch_shape == expected_dist.batch_shape

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample()
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_discrete_diag_normal_log_prob(init_shape, trans_shape, obs_shape, state_dim):
    event_size = 4
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    loc = torch.randn(obs_shape + (state_dim, event_size))
    scale = torch.randn(obs_shape + (state_dim, event_size)).exp()
    obs_dist = dist.Normal(loc, scale).to_event(1)

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)
    expected_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    assert actual_dist.event_shape == expected_dist.event_shape
    assert actual_dist.batch_shape == expected_dist.batch_shape

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape + (state_dim,)).sample()
    data = data[(slice(None),) * len(batch_shape) + (0,)]
    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-5, rtol=1e-5)


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
def test_gaussian_hmm_log_prob(init_shape, trans_mat_shape, trans_mvn_shape,
                               obs_mat_shape, obs_mvn_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_mat = torch.randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(trans_mvn_shape, hidden_dim)
    obs_mat = torch.randn(obs_mat_shape + (hidden_dim, obs_dim))
    obs_dist = random_mvn(obs_mvn_shape, obs_dim)

    actual_dist = GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
    expected_dist = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
    assert actual_dist.batch_shape == expected_dist.batch_shape
    assert actual_dist.event_shape == expected_dist.event_shape

    shape = broadcast_shape(init_shape + (1,),
                            trans_mat_shape, trans_mvn_shape,
                            obs_mat_shape, obs_mvn_shape)
    data = obs_dist.expand(shape).sample()
    assert data.shape == actual_dist.shape()

    # https://github.com/pyro-ppl/funsor/issues/184
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    actual_log_prob = actual_dist.log_prob(data)

    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("obs_dim", [1, 2, 3])
@pytest.mark.parametrize("hidden_dim", [1, 2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", DISCRETE_HMM_SHAPES, ids=str)
def test_gaussian_mrf_log_prob(init_shape, trans_shape, obs_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_dist = random_mvn(trans_shape, hidden_dim + hidden_dim)
    obs_dist = random_mvn(obs_shape, hidden_dim + obs_dim)

    actual_dist = GaussianMRF(init_dist, trans_dist, obs_dist)
    expected_dist = dist.GaussianMRF(init_dist, trans_dist, obs_dist)
    assert actual_dist.event_shape == expected_dist.event_shape
    assert actual_dist.batch_shape == expected_dist.batch_shape

    batch_shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    data = obs_dist.expand(batch_shape).sample()[..., hidden_dim:]
    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=1e-4)


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

    init_logits = torch.randn(init_cat_shape)
    init_mvn = random_mvn(init_mvn_shape, hidden_dim)
    trans_logits = torch.randn(trans_cat_shape)
    trans_matrix = torch.randn(trans_mat_shape)
    trans_mvn = random_mvn(trans_mvn_shape, hidden_dim)
    obs_matrix = torch.randn(obs_mat_shape)
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

    data = obs_mvn.expand(shape).sample()[..., 0, :]
    actual_log_prob = actual_dist.log_prob(data)
    assert actual_log_prob.shape == expected_batch_shape


@pytest.mark.parametrize("num_components", [2, 3])
@pytest.mark.parametrize("obs_dim,hidden_dim",
                         [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)])
@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "approx"])
def test_switching_linear_hmm_log_prob(exact, num_steps, hidden_dim, obs_dim, num_components):
    # This tests agreement between an SLDS and an HMM when all components
    # are identical, i.e. so latent can be marginalized out.
    init_logits = torch.randn(num_components)
    init_mvn = random_mvn((), hidden_dim)
    trans_logits = torch.randn(num_components)
    trans_matrix = torch.randn(hidden_dim, hidden_dim)
    trans_mvn = random_mvn((), hidden_dim)
    obs_matrix = torch.randn(hidden_dim, obs_dim)
    obs_mvn = random_mvn((), obs_dim)

    expected_dist = GaussianHMM(init_mvn,
                                trans_matrix.expand(num_steps, -1, -1),
                                trans_mvn, obs_matrix, obs_mvn)
    actual_dist = SwitchingLinearHMM(init_logits, init_mvn, trans_logits,
                                     trans_matrix.expand(num_steps, num_components, -1, -1),
                                     trans_mvn, obs_matrix, obs_mvn,
                                     exact=exact)
    assert actual_dist.batch_shape == expected_dist.batch_shape
    assert actual_dist.event_shape == expected_dist.event_shape

    data = obs_mvn.sample(expected_dist.batch_shape + (num_steps,))
    assert data.shape == expected_dist.shape()
    expected_log_prob = expected_dist.log_prob(data)
    assert expected_log_prob.shape == expected_dist.batch_shape
    actual_log_prob = actual_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-5, rtol=1e-5)
