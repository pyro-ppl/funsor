import pyro.distributions as dist
import pytest
import torch
from torch.distributions import constraints

from funsor.pyro.convert import tensor_to_funsor
from funsor.pyro.distribution import FunsorDistribution
from funsor.testing import assert_close

SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]


class Categorical(FunsorDistribution):
    def __init__(self, logits, validate_args=None):
        batch_shape = logits.shape[:-1]
        event_shape = torch.Size()
        funsor_dist = tensor_to_funsor(logits, ("value",))
        dtype = int(logits.size(-1))
        super(Categorical, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self.dtype - 1)


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


@pytest.mark.parametrize("cardinality", [2, 3])
@pytest.mark.parametrize("sample_shape", SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", SHAPES, ids=str)
def test_categorical_sample(sample_shape, batch_shape, cardinality):
    logits = torch.randn(batch_shape + (cardinality,))
    logits -= logits.logsumexp(dim=-1, keepdim=True)
    actual = Categorical(logits=logits)
    expected = dist.Categorical(logits=logits)
    assert actual.batch_shape == expected.batch_shape
    assert actual.event_shape == expected.event_shape

    actual_sample = actual.sample(sample_shape)
    expected_sample = expected.sample(sample_shape)
    assert actual_sample.dtype == expected_sample.dtype
    assert actual_sample.shape == expected_sample.shape
    expected.log_prob(actual_sample)  # validates sample
