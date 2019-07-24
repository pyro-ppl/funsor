from __future__ import absolute_import, division, print_function

import pyro.distributions as dist
import pytest
import torch
from torch.distributions import constraints

from funsor.pyro.convert import tensor_to_funsor
from funsor.pyro.distribution import FunsorDistribution
from funsor.testing import assert_close

SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]


class Categorical(FunsorDistribution):
    def __init__(self, logits):
        batch_shape = logits.shape[:-1]
        event_shape = torch.Size()
        funsor_dist = tensor_to_funsor(logits, event_dim=1)["value"]
        dtype = int(logits.size(-1))
        super(Categorical, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype)


@pytest.mark.parametrize("cardinality", [2, 3])
@pytest.mark.parametrize("sample_shape", SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", SHAPES, ids=str)
def test_categorical_log_prob(sample_shape, batch_shape, cardinality):
    logits = torch.randn(batch_shape + (cardinality,))
    logits -= logits.logsumexp(dim=-1, keepdim=True)
    actual = Categorical(logits=logits)
    expected = dist.Categorical(logits=logits)
    assert actual.batch_shape == expected.batch_shape
    assert actual.event_shape == expected.event_shape

    value = expected.sample(sample_shape)
    actual_log_prob = actual.log_prob(value)
    expected_log_prob = expected.log_prob(value)
    assert_close(actual_log_prob, expected_log_prob)
