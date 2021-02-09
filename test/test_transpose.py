# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.adjoint import adjoint
from funsor.domains import Bint, Real, Reals
from funsor.interpreter import interpretation
from funsor.terms import Number, Variable, lazy
from funsor.testing import assert_close, random_tensor

# from funsor.transpose import transpose


def transpose(expr):
    return adjoint(ops.add, ops.mul, expr)


def test_identity():
    x = random_tensor(OrderedDict(i=Bint[2]))
    assert transpose(x)[x] is Number(1.0)


def test_two():
    x = random_tensor(OrderedDict(i=Bint[2]))
    with interpretation(lazy):
        y = x + x
    assert transpose(y)[x] is Number(2.0)
    assert transpose(y)[y] is Number(1.0)


def test_zero():
    x = random_tensor(OrderedDict(i=Bint[2]))
    with interpretation(lazy):
        y = x - x
    assert transpose(y)[x] is Number(0.0)
    assert transpose(y)[y] is Number(1.0)


def test_four():
    x = random_tensor(OrderedDict(i=Bint[2]))
    with interpretation(lazy):
        y = x + x
        z = y + y
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_four_variable():
    x = Variable("x", Real)
    with interpretation(lazy):
        y = x + x
        z = y + y
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_eight_tensor():
    w = random_tensor(OrderedDict(i=Bint[2]))
    with interpretation(lazy):
        x = w + w
        y = x + x
        z = y + y
    assert transpose(z)[w] is Number(8.0)
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_eight_variable():
    w = Variable("w", Real)
    with interpretation(lazy):
        x = w + w
        y = x + x
        z = y + y
    assert transpose(z)[w] is Number(8.0)
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_mul():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    with interpretation(lazy):
        z = x * y
    assert_close(transpose(z)[x], y.reduce(ops.add, "k"))
    assert_close(transpose(z)[y], x.reduce(ops.add, "i"))


def test_sum():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4]))
    with interpretation(lazy):
        z = x.reduce(ops.add, "j")
    assert_close(transpose(z)[x], Number(1.0))


def test_matmul_tensor():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, "j")
    assert xy in transpose(z)
    assert_close(transpose(z)[xy], Number(1.0))
    assert_close(transpose(z)[x], y.reduce(ops.add, "k"))
    assert_close(transpose(z)[y], x.reduce(ops.add, "i"))


@pytest.mark.xfail(reason="scaling reductions not executing in lazy?")
def test_matmul_variable():
    x = Variable("x", Real)
    y = Variable("y", Real)
    j = Variable("j", Bint[4])
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, j)
    assert xy in transpose(z)
    assert_close(transpose(z)[xy], Number(1.0))
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)


def test_batched_matmul():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4], b=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5], b=Bint[2]))
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, "j")
    assert_close(transpose(z)[x], y.reduce(ops.add, {"k"}))
    assert_close(transpose(z)[y], x.reduce(ops.add, {"i"}))
    assert_close(transpose(z)[xy], Number(1.0))


def test_expand_reduce():
    i = Variable("i", Bint[3])
    x = Variable("x", Real)
    y = Variable("y", Real)
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, i)
    assert_close(transpose(z)[x], y * 3)
    assert_close(transpose(z)[y], x * 3)
    assert_close(transpose(z)[xy], Number(1.0))


def test_tensor_contract():
    x = random_tensor(OrderedDict(i1=Bint[2], i2=Bint[3], j1=Bint[4], j2=Bint[5]))
    y = random_tensor(OrderedDict(j1=Bint[4], j2=Bint[5], k1=Bint[6], k2=Bint[7]))
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, {"j1", "j2"})
    assert_close(transpose(z)[x], y.reduce(ops.add, {"k1", "k2"}))
    assert_close(transpose(z)[y], x.reduce(ops.add, {"i1", "i2"}))
    assert_close(transpose(z)[xy], Number(1.0))


@pytest.mark.parametrize("height", [1, 2, 3, 10])  # , 100, 1000])
def test_tower_sum(height):
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    with interpretation(lazy):
        top = x
        for _ in range(height):
            top = top + top
    assert transpose(top)[x] is Number(2.0 ** height)


@pytest.mark.skip(reason="exponential growth? infinite loop?")
@pytest.mark.parametrize("height", [1, 2, 3, 10, 100, 1000])
def test_tower_prod(height):
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    with interpretation(lazy):
        top = x
        expected = x
        for _ in range(height):
            top = top * top
            expected = expected + expected
        # This might be too much to ask:
        transpose(top)[x] is expected

    # This should definitely hold:
    assert_close(transpose(top)[x], x * 2 ** height)


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
    expected = f
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


"""
TODO tests the following ideas:

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
