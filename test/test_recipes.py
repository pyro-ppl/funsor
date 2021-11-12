# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.interpretations import memoize
from funsor.montecarlo import extract_samples
from funsor.recipes import (
    forward_filter_backward_precondition,
    forward_filter_backward_rsample,
)
from funsor.terms import Lambda, Variable
from funsor.testing import Tensor, assert_close, randn, random_gaussian
from funsor.util import get_backend


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
    """
    This can be seen as performing naive tensor variable elimination by
    breaking all plates and creating a single flat joint distribution.
    """
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
    assert_close(actual_log_prob, expected_log_prob, atol=1e-4, rtol=None)

    # Check sample moments.
    sample_inputs = OrderedDict(particle=actual_log_prob.inputs["particle"])
    rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
    flat_deltas = flat_joint.sample(
        {"flat_" + k for k in flat_vars}, sample_inputs, rng_key
    )
    flat_samples = extract_samples(flat_deltas)
    expected_samples = {
        k: flat_samples["flat_" + k][broken_plates[k]] for k in flat_vars
    }
    expected_moments = get_moments(expected_samples)
    actual_moments = get_moments(actual_samples)
    assert_close(actual_moments, expected_moments, atol=0.02, rtol=None)


def substitute_aux(samples, log_prob, num_samples):
    assert all("aux" in v.inputs for v in samples.values())
    assert set(log_prob.inputs) == {"aux"}

    # Substitute noise for the aux value, as would happen each SVI step.
    aux_numel = log_prob.inputs["aux"].num_elements
    noise = Tensor(randn(num_samples, aux_numel))["particle"]
    with memoize():
        samples = {k: v(aux=noise) for k, v in samples.items()}
        log_prob = log_prob(aux=noise)

    return samples, log_prob


@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_1(backward):
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        pyro.sample("b", dist.Normal(a, 1), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"a": Real})),
    }
    eliminate = frozenset(["a"])
    plates = frozenset()

    if backward == "sample":
        sample_inputs = OrderedDict(particle=Bint[num_samples])
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a"}
    assert actual_samples["a"].output == Real
    assert set(actual_samples["a"].inputs) == {"particle"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_2(backward):
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        pyro.sample("c", dist.Normal(a, b.exp()), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"b": Real})),
        "c": random_gaussian(OrderedDict({"a": Real, "b": Real})),
    }
    eliminate = frozenset(["a", "b"])
    plates = frozenset()

    if backward == "sample":
        sample_inputs = {"particle": Bint[num_samples]}
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a", "b"}
    assert actual_samples["a"].output == Real
    assert actual_samples["b"].output == Real
    assert set(actual_samples["a"].inputs) == {"particle"}
    assert set(actual_samples["b"].inputs) == {"particle"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_3(backward):
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("i", 2):
            b = pyro.sample("b", dist.Normal(0, 1))
            pyro.sample("c", dist.Normal(a, b.exp()), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"i": Bint[2], "b": Real})),
        "c": random_gaussian(OrderedDict({"i": Bint[2], "a": Real, "b": Real})),
    }
    eliminate = frozenset(["a", "b", "i"])
    plates = frozenset(["i"])

    if backward == "sample":
        sample_inputs = {"particle": Bint[num_samples]}
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a", "b"}
    assert actual_samples["a"].output == Real
    assert actual_samples["b"].output == Real
    assert set(actual_samples["a"].inputs) == {"particle"}
    assert set(actual_samples["b"].inputs) == {"particle", "i"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_4(backward):
    """
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        with pyro.plate("i", 2):
            c = pyro.sample("c", dist.Normal(a, 1))
            d = pyro.sample("d", dist.Normal(b, 1))
            with pyro.plate("j", 3):
                pyro.sample("e", dist.Normal(c, d.exp()), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"a": Real})),
        "b": random_gaussian(OrderedDict({"b": Real})),
        "c": random_gaussian(OrderedDict({"i": Bint[2], "a": Real, "c": Real})),
        "d": random_gaussian(OrderedDict({"i": Bint[2], "b": Real, "d": Real})),
        "e": random_gaussian(
            OrderedDict({"i": Bint[2], "j": Bint[3], "c": Real, "d": Real})
        ),
    }
    eliminate = frozenset(["a", "b", "c", "d", "i", "j"])
    plates = frozenset(["i", "j"])

    if backward == "sample":
        sample_inputs = {"particle": Bint[num_samples]}
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a", "b", "c", "d"}
    assert actual_samples["a"].output == Real
    assert actual_samples["b"].output == Real
    assert actual_samples["c"].output == Real
    assert actual_samples["d"].output == Real
    assert set(actual_samples["a"].inputs) == {"particle"}
    assert set(actual_samples["b"].inputs) == {"particle"}
    assert set(actual_samples["c"].inputs) == {"particle", "i"}
    assert set(actual_samples["d"].inputs) == {"particle", "i"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_5(backward):
    """
    def model(data):
        a = pyro.sample("a", dist.MultivariateNormal(zeros(2), eye(2)))
        b = pyro.sample("b", dist.MultivariateNormal(a, eye(2)))
        c = pyro.sample("c", dist.MultivariateNormal(b, eye(2)))
        d = pyro.sample("d", dist.MultivariateNormal(c, eye(2)))
        pyro.sample("e", dist.MultivariateNormal(d, eye(2)), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"a": Reals[2]})),
        "b": random_gaussian(OrderedDict({"b": Reals[2], "a": Reals[2]})),
        "c": random_gaussian(OrderedDict({"c": Reals[2], "b": Reals[2]})),
        "d": random_gaussian(OrderedDict({"d": Reals[2], "c": Reals[2]})),
        "e": random_gaussian(OrderedDict({"d": Reals[2]})),
    }
    eliminate = frozenset(["a", "b", "c", "d"])
    plates = frozenset()

    if backward == "sample":
        sample_inputs = {"particle": Bint[num_samples]}
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a", "b", "c", "d"}
    assert actual_samples["a"].output == Reals[2]
    assert actual_samples["b"].output == Reals[2]
    assert actual_samples["c"].output == Reals[2]
    assert actual_samples["d"].output == Reals[2]
    assert set(actual_samples["a"].inputs) == {"particle"}
    assert set(actual_samples["b"].inputs) == {"particle"}
    assert set(actual_samples["c"].inputs) == {"particle"}
    assert set(actual_samples["d"].inputs) == {"particle"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.xfail(reason="TODO handle intractable case")
@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_intractable_1(backward):
    """
    def model(data):
        i_plate = pyro.plate("i", 2, dim=-2)
        j_plate = pyro.plate("j", 3, dim=-1)
        with i_plate:
            a = pyro.sample("a", dist.Normal(0, 1))
        with i_plate:
            b = pyro.sample("b", dist.Normal(0, 1))
        with i_plate, j_plate:
            pyro.sample("c", dist.Normal(a, b), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"i": Bint[2], "a": Real})),
        "b": random_gaussian(OrderedDict({"j": Bint[2], "b": Real})),
        "c": random_gaussian(
            OrderedDict({"i": Bint[2], "j": Bint[2], "a": Real, "b": Real})
        ),
    }
    eliminate = frozenset(["a", "b", "i", "j"])
    plates = frozenset(["i", "j"])

    if backward == "sample":
        sample_inputs = {"particle": Bint[num_samples]}
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a", "b"}
    assert actual_samples["a"].output == Real
    assert actual_samples["b"].output == Real
    assert set(actual_samples["a"].inputs) == {"particle", "i"}
    assert set(actual_samples["b"].inputs) == {"particle", "j"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)


@pytest.mark.xfail(reason="TODO handle colliders via Lambda")
@pytest.mark.parametrize("backward", ["sample", "precondition"])
def test_ffb_intractable_2(backward):
    """
    def model(data):
        with pyro.plate("i", 2):
            a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a.sum(), 1), obs=data)
    """
    num_samples = int(1e5)

    factors = {
        "a": random_gaussian(OrderedDict({"i": Bint[2], "a": Real})),
        "b": random_gaussian(OrderedDict({"a_i": Reals[2]})),
    }
    eliminate = frozenset(["a", "i"])
    plates = frozenset(["i"])

    if backward == "sample":
        sample_inputs = {"particle": Bint[num_samples]}
        rng_key = None if get_backend() != "jax" else np.array([0, 0], dtype=np.uint32)
        actual_samples, actual_log_prob = forward_filter_backward_rsample(
            factors, eliminate, plates, sample_inputs, rng_key
        )
    elif backward == "precondition":
        samples, log_prob = forward_filter_backward_precondition(
            factors, eliminate, plates
        )
        actual_samples, actual_log_prob = substitute_aux(samples, log_prob, num_samples)

    assert set(actual_samples) == {"a"}
    assert set(actual_samples["a"].inputs) == {"particle", "i"}
    check_ffbr(factors, eliminate, plates, actual_samples, actual_log_prob)
