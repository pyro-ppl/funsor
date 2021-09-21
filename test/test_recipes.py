# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import funsor
from funsor import Bint, Real
from funsor.recipes import forward_filter_backward_rsample
from funsor.testing import random_gaussian
from funsor.montecarlo import extract_samples


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
    assert set(actual_samples.inputs) == {"a", "b", "particles"}

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    assert set(expected_samples.inputs) == {"a", "b", "particles"}


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

    # Check log_prob.
    guide = sum(factors.values())
    expected_log_prob = guide(**actual_samples)
    assert_close(actual_log_prob, expected_log_prob)

    # Check sample moments.
    joint = sum(factors.values())
    expected_samples = extract_samples(joint.sample(**sample_inputs))
    assert set(expected_samples.inputs) == {"a", "b", "particles"}
