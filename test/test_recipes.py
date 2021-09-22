# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Dict, Tuple

import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.montecarlo import extract_samples
from funsor.recipes import forward_filter_backward_rsample
from funsor.terms import Lambda, Variable
from funsor.testing import assert_close, random_gaussian


def get_moments(samples):
    reduced_vars = frozenset(["particle"])
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


def check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob):
    assert "particle" not in plates
    flat_vars: Dict[str, Variable] = {}
    plate_vars: Dict[str, Variable] = {}
    broken_plates: Dict[str, Tuple[Variable]] = {}
    for name, factor in factors.items():
        for k, d in factor.inputs.items():
            if k in plates:
                plate_vars[k] = Variable(k, d)
        if name in factor.inputs:  # i.e. if is latent
            broken_plates[name] = tuple(
                plate_vars[p] for p in sorted(plates.intersection(factor.inputs))
            )
            # I guess we could use Lambda here?
            broken_shape = tuple(p.output.size for p in broken_plates[name])
            domain = Reals[broken_shape + factor.inputs[name].shape]
            flat_vars[name] = Variable("flat_" + name, domain)[broken_plates[name]]

    flat_factors = []
    for factor in factors.values():
        f = factor(**flat_vars)
        f = f.reduce(ops.add, plates.intersection(f.inputs))
        flat_factors.append(f)

    # Check log prob.
    flat_joint = sum(flat_factors)
    log_Z = flat_joint.reduce(ops.logaddexp)
    flat_samples = {}
    for k, v in actual_samples.items():
        for p in reversed(broken_plates[k]):
            v = Lambda(p, v)
        flat_samples["flat_" + k] = v
    expected_log_prob = flat_joint(**flat_samples) - log_Z
    assert_close(actual_log_prob, expected_log_prob, atol=1e-5, rtol=None)

    # Check sample moments.
    sample_inputs = OrderedDict(particle=actual_log_prob.inputs["particle"])
    flat_deltas = flat_joint.sample({"flat_" + k for k in flat_vars}, sample_inputs)
    flat_samples = extract_samples(flat_deltas)
    expected_samples = {
        k: flat_samples["flat_" + k][broken_plates[k]] for k in flat_vars
    }
    expected_moments = get_moments(expected_samples)
    actual_moments = get_moments(actual_samples)
    assert_close(actual_moments, expected_moments, atol=0.02, rtol=None)


def test_ffbr_1():
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a, 1), obs=data)
    """
    num_samples = 10000

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"a": Real})),
    }
    eliminate = frozenset(["a"])
    plates = frozenset()
    sample_inputs = OrderedDict(particle=Bint[num_samples])

    actual_samples, actual_log_prob = forward_filter_backward_rsample(
        factors, eliminate, plates, sample_inputs
    )
    assert set(actual_samples) == {"a"}
    assert set(actual_samples["a"].inputs) == {"particle"}

    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


def test_ffbr_2():
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        c = pyro.sample("c", dist.Normal(a, b.exp()), obs=data)
    """
    num_samples = 10000

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

    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


def test_ffbr_3():
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("plate", 2):
            b = pyro.sample("b", dist.Normal(0, 1))
            c = pyro.sample("c", dist.Normal(a, b.exp()), obs=data)
    """
    num_samples = 10000

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

    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.xfail(reason="TODO handle colliders via Lambda")
def test_ffbr_4():
    """
    def model(data):
        with pyro.plate("plate", 2):
            a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a.sum(), 1), obs=data)
    """
    num_samples = 10000

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

    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)
