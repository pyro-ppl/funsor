# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import numpy as np
import pytest

from funsor import ops
from funsor.approximations import (
    argmax_approximate,
    elbo_approximate,
    laplace_approximate,
    mean_approximate,
)
from funsor.domains import Bint, Real, Reals
from funsor.interpretations import eager, normalize
from funsor.interpreter import reinterpret
from funsor.montecarlo import MonteCarlo
from funsor.testing import (
    assert_close,
    random_gaussian,
    random_tensor,
    xfail_if_not_implemented,
    xfail_param,
)

monte_carlo = MonteCarlo(rng_key=np.array([0, 0], dtype=np.uint32))


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        elbo_approximate,
        monte_carlo,
    ],
)
def test_tensor_smoke(approximate):
    with normalize:
        model = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
        guide = random_tensor(OrderedDict(j=Bint[3]))
        p = model.approximate(ops.logaddexp, guide, "j")
    with approximate, xfail_if_not_implemented():
        q = reinterpret(p)
    assert q.output == p.output
    assert q.input_vars.issubset(p.input_vars)


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        elbo_approximate,
        laplace_approximate,
        xfail_param(mean_approximate, reason="alpha conversion bug"),
        monte_carlo,
    ],
)
def test_gaussian_smoke(approximate):
    with normalize:
        model = random_gaussian(OrderedDict(i=Bint[2], u=Real, v=Reals[3]))
        guide = random_gaussian(OrderedDict(u=Real, v=Reals[3]))
        p = model.approximate(ops.logaddexp, guide, {"u", "v"})
    with approximate, xfail_if_not_implemented():
        q = reinterpret(p)
    assert q.output == p.output
    assert q.input_vars.issubset(p.input_vars)


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        xfail_param(monte_carlo, reason="only true in expectation"),
    ],
)
def test_tensor_linear(approximate):
    m1 = random_tensor(OrderedDict(i=Bint[2], x=Bint[4]))
    m2 = random_tensor(OrderedDict(j=Bint[3], x=Bint[4]))
    s = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    guide = random_tensor(OrderedDict(i=Bint[2], j=Bint[3], x=Bint[4]))
    with approximate, xfail_if_not_implemented():
        expected = (m1 + s * m2).approximate(ops.logaddexp, guide, "x")
        q1 = m1.approximate(ops.logaddexp, guide, "x")
        q2 = m2.approximate(ops.logaddexp, guide, "x")
    actual = q1 + s * q2
    assert_close(actual, expected)


@pytest.mark.parametrize(
    "approximate",
    [
        eager,
        argmax_approximate,
        laplace_approximate,
        mean_approximate,
        xfail_param(monte_carlo, reason="only true in expectation"),
    ],
)
def test_gaussian_linear(approximate):
    m1 = random_gaussian(OrderedDict(i=Bint[2], x=Real))
    m2 = random_gaussian(OrderedDict(j=Bint[3], x=Real))
    s = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    guide = random_gaussian(OrderedDict(i=Bint[2], j=Bint[3], x=Real))
    with approximate, xfail_if_not_implemented():
        expected = (m1 + s * m2).approximate(ops.logaddexp, guide, "x")
        q1 = m1.approximate(ops.logaddexp, guide, "x")
        q2 = m2.approximate(ops.logaddexp, guide, "x")
    actual = q1 + s * q2
    assert_close(actual, expected)
