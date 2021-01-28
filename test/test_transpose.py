# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from funsor.domains import Real
from funsor.terms import Variable
from funsor.testing import assert_close
from funsor.transpose import transpose

import pytest


@pytest.mark.parametrize("f", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
@pytest.mark.parametrize("g", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
def test_product_rule(f, g):
    x = Variable("x", Real)
    y = Variable("y", Real)
    f = eval(f)
    g = eval(g)

    for arg in [x, y]:
        actual = transpose(f * g)[arg]
        expected = transpose(f)[arg] * g + f * transpose(g)[arg]
        print(f"actual:\n{str(actual)}")
        print(f"expected:\n{str(expected)}")
        assert actual.input_vars <= expected.input_vars
        assert actual.output == expected.output
        if actual is not expected:
            for x_, y_ in [(0., 0.1), (-1.3, 0.2), (0.3, 1.5)]:
                assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))


@pytest.mark.parametrize("f", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
@pytest.mark.parametrize("g", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
def test_sum_rule(f, g):
    x = Variable("x", Real)
    y = Variable("y", Real)
    f = eval(f)
    g = eval(g)

    for arg in [x, y]:
        actual = transpose(f + g)[arg]
        expected = transpose(f)[arg] + transpose(g)[arg]
        print(f"actual:\n{str(actual)}")
        print(f"expected:\n{str(expected)}")
        assert actual.input_vars <= expected.input_vars
        assert actual.output == expected.output
        if actual is not expected:
            for x_, y_ in [(0., 0.1), (-1.3, 0.2), (0.3, 1.5)]:
                assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))
