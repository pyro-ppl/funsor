# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from importlib import import_module

import pyro.distributions as dist
import pytest
import torch
from pyro.distributions.util import broadcast_shape

from funsor.numpyro.hmm import GaussianHMM
from funsor.distributions import BACKEND_TO_DISTRIBUTION_BACKEND
from funsor.testing import assert_close, random_mvn, randn, random_tensor


randn = partial(randn, backend="numpy")
random_tensor = partial(random_tensor, backend="numpy")
random_mvn = partial(random_mvn, backend="numpy")
dist = import_module(BACKEND_TO_DISTRIBUTION_BACKEND["numpy"])


def check_expand(old_dist, old_data):
    new_batch_shape = (2,) + old_dist.batch_shape
    new_dist = old_dist.expand(new_batch_shape)
    assert new_dist.batch_shape == new_batch_shape

    old_log_prob = new_dist.log_prob(old_data)
    assert old_log_prob.shape == new_batch_shape

    new_data = old_data.expand(new_batch_shape + new_dist.event_shape)
    new_log_prob = new_dist.log_prob(new_data)
    assert_close(old_log_prob, new_log_prob)
    assert new_dist.log_prob(new_data).shape == new_batch_shape


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
    trans_mat = randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(trans_mvn_shape, hidden_dim)
    obs_mat = randn(obs_mat_shape + (hidden_dim, obs_dim))
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

    actual_log_prob = actual_dist.log_prob(data)
    expected_log_prob = expected_dist.log_prob(data)
    assert_close(actual_log_prob, expected_log_prob, atol=1e-5, rtol=1e-5)
    check_expand(actual_dist, data)
