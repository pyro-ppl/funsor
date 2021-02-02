# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import Bint, Real, Reals
from funsor.interpreter import interpretation
from funsor.montecarlo import MonteCarlo
from funsor.terms import Variable, lazy
from funsor.testing import assert_close, random_tensor
from funsor.transpose import transpose


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
            for x_, y_ in [(0.0, 0.1), (-1.3, 0.2), (0.3, 1.5)]:
                assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))


@pytest.mark.parametrize("f", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
@pytest.mark.parametrize("g", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
def test_binary_sum_rule(f, g):
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
            for x_, y_ in [(0.0, 0.1), (-1.3, 0.2), (0.3, 1.5)]:
                assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))


@pytest.mark.xfail(reason="probably an alpha conversion issue")
def test_reduce_sum_rule():
    i = Variable("i", Bint[4])
    f = random_tensor(OrderedDict(i=Bint[4]))
    with interpretation(lazy):
        f_ = f.reduce(ops.add, i)

    actual = transpose(f_)[f]
    expected = transpose(f)[f]
    print(f"actual:\n{str(actual)}")
    print(f"expected:\n{str(expected)}")
    assert actual.input_vars <= expected.input_vars
    assert actual.output == expected.output
    if actual is not expected:
        for x_, y_ in [(0.0, 0.1), (-1.3, 0.2), (0.3, 1.5)]:
            assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))


@pytest.mark.xfail(reason="TODO reverse alpha renaming")
def test_reduce_sum_getitem():
    x = Variable("x", Reals[4])
    i = Variable("i", Bint[4])
    f = x[i]

    actual = transpose(f.reduce(ops.add, i))[x]
    expected = transpose(f)[x]
    print(f"actual:\n{str(actual)}")
    print(f"expected:\n{str(expected)}")
    assert actual.input_vars <= expected.input_vars
    assert actual.output == expected.output
    if actual is not expected:
        for x_, y_ in [(0.0, 0.1), (-1.3, 0.2), (0.3, 1.5)]:
            assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))


@pytest.mark.xfail(reason="missing behavior")
def test_backward_marginal():
    """
    This tests the following ideas:

      i:Bint[3] |- x:Real
    -----------------------
    |- x.reduce(op, i):Real

               i:Bint[3] |- x:Real
    ----------------------------------------------------
    i2:Bint[3] |- transpose(x.reduce(op, i, i2))[x]:Real

       G1 |- x1:t1   ...   Gn |- xn:tn
          G0 |- f(x1, ..., xn):Real
            f structurally linear
    --------------------------------------
    Gi |- transpose(F(x1, ..., xn))[xi]:ti

    G1 |- x:t1
    G2 |- y = f(x) : t2
    f structurally linear
    --------------------------------------
    G2 |- transpose(transpose(y)) = y : t2
    """
    i = Variable("i", Bint[3])
    j = Variable("j", Bint[4])
    x = random_tensor(OrderedDict(i=i.output, j=j.output))

    # A forward computation.
    with interpretation(lazy):
        y = x.reduce(ops.logaddexp, j)

    # Its transpose: backward marginal.
    x2 = transpose(y)[x]
    assert x2.inputs == x.inputs
    assert x2.output == x.output

    # Now in a different interpretation.
    # FIXME can we sample eagerly but substitute lazily?
    with interpretation(lazy), MonteCarlo():
        y2 = x.reduce(ops.logaddexp, j)

    # monte carlo sampling
    x3 = transpose(y2)[x]
    assert x3.inputs == x.inputs
    assert x3.output == x.output

    # monte carlo integration
    y3 = x3.reduce(ops.logaddexp, j)
    assert_close(y2, y3)
