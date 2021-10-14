# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

from funsor import ops
from funsor.domains import Bint, Real
from funsor.elbo import Elbo
from funsor.integrate import Integrate
from funsor.interpretations import normalize
from funsor.interpreter import reinterpret
from funsor.montecarlo import MonteCarlo
from funsor.tensor import Tensor
from funsor.terms import Variable
from funsor.testing import assert_close, random_gaussian


def test_simple():
    model = random_gaussian(OrderedDict(x=Real))
    guide = random_gaussian(OrderedDict(x=Real))
    approx_vars = frozenset({Variable("x", Real)})

    expected = Integrate(guide, model - guide, approx_vars)

    with Elbo(guide, approx_vars):
        actual = model.reduce(ops.logaddexp, approx_vars)

    assert_close(actual, expected)


@pytest.mark.xfail(reason="interpreter stack overflow")
def test_monte_carlo():
    model = random_gaussian(OrderedDict(x=Real))
    guide = random_gaussian(OrderedDict(x=Real))
    approx_vars = frozenset({Variable("x", Real)})

    with Elbo(guide, approx_vars):
        expected = model.reduce(ops.logaddexp, approx_vars)
    assert isinstance(expected, Tensor)
    with MonteCarlo(particles=Bint[10000]), Elbo(guide, approx_vars):
        actual = model.reduce(ops.logaddexp, approx_vars)
    assert isinstance(actual, Tensor)

    assert_close(actual, expected, atol=0.1)


@pytest.mark.xfail(reason="missing pattern")
def test_complex():
    with normalize:
        xy = random_gaussian(OrderedDict(x=Real, y=Real))
        yz = random_gaussian(OrderedDict(y=Real, z=Real))
        model = xy + yz

        x = random_gaussian(OrderedDict(x=Real))
        y = random_gaussian(OrderedDict(y=Real))
        z = random_gaussian(OrderedDict(z=Real))
        guide = x + y + z

    approx_vars = frozenset(
        {Variable("x", Real), Variable("y", Real), Variable("z", Real)}
    )

    expected = Integrate(guide, model - guide, approx_vars)

    with Elbo(guide, approx_vars):
        actual = model.reduce(ops.logaddexp, approx_vars)

    # Reinterpret to ensure Integrate is evaluated.
    expected = reinterpret(expected)
    assert isinstance(expected, Tensor)
    actual = reinterpret(actual)
    assert isinstance(actual, Tensor)
    assert_close(actual, expected)
