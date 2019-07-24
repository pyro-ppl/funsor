import pyro.distributions as dist
import pytest
import torch

from funsor.pyro.hmm import DiscreteHMM
from funsor.testing import assert_close


@pytest.mark.xfail(reason="requires non-scalar shape of bint output", run=False)
@pytest.mark.parametrize("state_dim", [2, 3])
@pytest.mark.parametrize("init_shape,trans_shape,obs_shape", [
    ((), (1,), ()),
    ((), (), (1,)),
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

], ids=str)
def test_discrete_categorical_log_prob(init_shape, trans_shape, obs_shape, state_dim):
    obs_dim = 4
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    obs_logits = torch.randn(obs_shape + (state_dim, obs_dim))
    obs_dist = dist.Categorical(logits=obs_logits)
    data = obs_dist.sample()[(slice(None),) * len(obs_shape) + (0,)]

    actual_dist = DiscreteHMM(init_logits, trans_logits, obs_dist)
    expected_dist = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)

    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob)
