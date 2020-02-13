# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from importlib import import_module

import pytest
from jax import random

import funsor.ops as ops
from funsor.distributions import BACKEND_TO_DISTRIBUTION_BACKEND
from funsor.numpyro.convert import tensor_to_funsor
from funsor.numpyro.distribution import FunsorDistribution
from funsor.testing import assert_close, randn


randn = partial(randn, backend="numpy")
dist = import_module(BACKEND_TO_DISTRIBUTION_BACKEND["numpy"])
SHAPES = [(), (1,), (4,), (2, 3), (1, 2, 1, 3, 1)]


class Categorical(FunsorDistribution):
    def __init__(self, logits, validate_args=None):
        batch_shape = logits.shape[:-1]
        event_shape = ()
        funsor_dist = tensor_to_funsor(logits, ("value",))
        dtype = int(logits.shape[-1])
        super(Categorical, self).__init__(
            funsor_dist, batch_shape, event_shape, dtype, validate_args)


@pytest.mark.parametrize("cardinality", [2, 3])
@pytest.mark.parametrize("sample_shape", SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", SHAPES, ids=str)
def test_categorical_log_prob(sample_shape, batch_shape, cardinality):
    logits = randn(batch_shape + (cardinality,))
    logits -= ops.logsumexp(logits, -1)[..., None]
    actual = Categorical(logits=logits)
    expected = dist.Categorical(logits=logits)
    assert actual.batch_shape == expected.batch_shape
    assert actual.event_shape == expected.event_shape

    value = expected.sample(random.PRNGKey(0), sample_shape)
    actual_log_prob = actual.log_prob(value)
    expected_log_prob = expected.log_prob(value)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("cardinality", [2, 3])
@pytest.mark.parametrize("sample_shape", SHAPES, ids=str)
@pytest.mark.parametrize("batch_shape", SHAPES, ids=str)
def test_categorical_sample(sample_shape, batch_shape, cardinality):
    logits = randn(batch_shape + (cardinality,))
    logits -= ops.logsumexp(logits, -1)[..., None]
    actual = Categorical(logits=logits)
    expected = dist.Categorical(logits=logits)
    assert actual.batch_shape == expected.batch_shape
    assert actual.event_shape == expected.event_shape

    actual_sample = actual.sample(sample_shape)
    expected_sample = expected.sample(random.PRNGKey(0), sample_shape)
    assert actual_sample.dtype == expected_sample.dtype
    assert actual_sample.shape == expected_sample.shape
    expected.log_prob(actual_sample)  # validates sample
