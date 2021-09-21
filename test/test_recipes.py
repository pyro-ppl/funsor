# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor.ops as ops
from funsor.domains import Bint, Real
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
    sample_inputs = {"particles": Bint[num_samples]}
    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples.inputs) == {"a", "particles"}
    actual_moments = get_moments(actual_samples, sample_inputs)

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    assert set(expected_samples.inputs) == {"a", "particles"}
    expected_moments = get_moments(expected_samples, sample_inputs)
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
    sample_inputs = {"particles": Bint[num_samples]}
    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples.inputs) == {"a", "b", "particles"}
    actual_moments = get_moments(actual_samples, sample_inputs)

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    assert set(expected_samples.inputs) == {"a", "b", "particles"}
    expected_moments = get_moments(expected_samples, sample_inputs)
    assert_close(actual_moments, expected_moments)
