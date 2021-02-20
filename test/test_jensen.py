# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from funsor import ops
from funsor.domains import Real
from funsor.integrate import Integrate
from funsor.interpretations import normalize
from funsor.interpreter import reinterpret
from funsor.jensen import JensenInterpretation
from funsor.terms import Variable
from funsor.testing import assert_close, random_gaussian


def test_simple():
    model = random_gaussian(OrderedDict(x=Real))
    guide = random_gaussian(OrderedDict(x=Real))

    approx_vars = frozenset({Variable("x", Real)})

    expected = Integrate(guide, model - guide, approx_vars)

    with JensenInterpretation(guide, approx_vars):
        actual = model.reduce(ops.logaddexp, approx_vars)

    assert_close(actual, expected)


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
        {
            Variable("x", Real),
            Variable("y", Real),
            Variable("z", Real),
        }
    )

    expected = Integrate(guide, model - guide, approx_vars)

    with JensenInterpretation(guide, approx_vars):
        actual = model.reduce(ops.logaddexp, approx_vars)

    # Reinterpret to ensure Integrate is evaluated.
    actual = reinterpret(actual)
    expected = reinterpret(expected)
    assert_close(actual, expected)
