# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.montecarlo import extract_samples
from funsor.recipes import forward_filter_backward_rsample
from funsor.testing import assert_close, random_gaussian


def get_moments(samples, sample_inputs):
    reduced_vars = frozenset(sample_inputs)
    moments = OrderedDict()

    # Compute first moments.
    diffs = OrderedDict()
    for name, value in samples.items():
        mean = value.reduce(ops.mean, reduced_vars)
        moments[name] = mean
        diffs[name] = value - mean

    # Compute centered second moments.
    for i, (name1, diff1) in enumerate(diffs.items()):
        diff1_ = diff1.reshape((diff1.output.num_elements, 1))
        for name2, diff2 in list(diffs.items())[:i]:
            diff_2 = diff2.reshape((1, diff2.output.num_elements))
            diff12 = diff1_ * diff_2
            moments[name1, name2] = diff12.reduce(ops.mean, reduced_vars)

    return moments


def test_ffbr_1():
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a, 1), obs=data)
    """
    num_samples = 1000

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"a": Real})),
    }
    eliminate = frozenset(["a", "b"])
    plates = frozenset()
    sample_inputs = {"particle": Bint[num_samples]}

    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples) == {"a"}
    assert set(actual_samples["a"].inputs) == {"particle"}

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    expected_moments = get_moments(expected_samples, sample_inputs)
    actual_moments = get_moments(actual_samples, sample_inputs)
    assert_close(actual_moments, expected_moments)


def test_ffbr_2():
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        c = pyro.sample("c", dist.Normal(a, b.exp()), obs=data)
    """
    num_samples = 1000

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"b": Real})),
        "c": random_gaussian(OrderedDict({"a": Real, "b": Real})),
    }
    eliminate = frozenset(["a", "b"])
    plates = frozenset()
    sample_inputs = {"particle": Bint[num_samples]}
    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples) == {"a", "b"}
    assert set(actual_samples["a"].inputs) == {"particle"}
    assert set(actual_samples["b"].inputs) == {"particle"}

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    expected_moments = get_moments(expected_samples, sample_inputs)
    actual_moments = get_moments(actual_samples, sample_inputs)
    assert_close(actual_moments, expected_moments)


def test_ffbr_3():
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("plate", 2):
            b = pyro.sample("b", dist.Normal(0, 1))
            c = pyro.sample("c", dist.Normal(a, b.exp()), obs=data)
    """
    num_samples = 1000

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"plate": Bint[2], "b": Real})),
        "c": random_gaussian(OrderedDict({"plate": Bint[2], "a": Real, "b": Real})),
    }
    eliminate = frozenset(["a", "b", "plate"])
    plates = frozenset(["plate"])
    sample_inputs = {"particle": Bint[num_samples]}
    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples) == {"a", "b"}
    assert set(actual_samples["a"].inputs) == {"particle"}
    assert set(actual_samples["b"].inputs) == {"plate", "particle"}

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    expected_moments = get_moments(expected_samples, sample_inputs)
    actual_moments = get_moments(actual_samples, sample_inputs)
    assert_close(actual_moments, expected_moments)


@pytest.mark.xfail(reason="TODO handle colliders via Lambda")
def test_ffbr_4():
    """
    def model(data):
        with pyro.plate("plate", 2):
            a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a.sum(), 1), obs=data)
    """
    num_samples = 1000

    factors = {
        "a": random_gaussian(OrderedDict({"plate": Bint[2], "a": Real})),
        "b": random_gaussian(OrderedDict({"a_plate": Reals[2]})),
    }
    eliminate = frozenset(["a", "plate"])
    plates = frozenset(["plate"])
    sample_inputs = {"particle": Bint[num_samples]}
    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples) == {"a"}
    assert set(actual_samples["a"].inputs) == {"plate", "particle"}

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    expected_moments = get_moments(expected_samples, sample_inputs)
    actual_moments = get_moments(actual_samples, sample_inputs)
    assert_close(actual_moments, expected_moments)
