import warnings

import pyro.distributions as dist
import pytest
import torch
from pyro.distributions.util import broadcast_shape

from funsor.pyro.hmm import DiscreteHMM, GaussianHMM, GaussianMRF
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


@pytest.mark.parametrize("obs_dim", [1, 2, 3])
@pytest.mark.parametrize("hidden_dim", [1, 2, 3])
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
