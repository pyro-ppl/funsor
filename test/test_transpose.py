# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.adjoint import adjoint
from funsor.domains import Bint, Real, Reals
from funsor.interpreter import interpretation, reinterpret
from funsor.tensor import Tensor
from funsor.terms import Number, Scatter, Variable, lazy, reflect
from funsor.testing import assert_close, random_tensor

# from funsor.transpose import transpose

try:
    import torch
except ImportError:
    pytest.skip()


def transpose(expr):
    result = adjoint(ops.add, ops.mul, expr)
    for key, value in result.items():
        assert value.input_vars <= key.input_vars | expr.input_vars
    return result


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
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)


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
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)


def test_matmul_variable():
    x = Variable("x", Real)
    y = Variable("y", Real)
    j = Variable("j", Bint[4])
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, j)
    assert xy in transpose(z)
    assert_close(transpose(z)[xy], Number(4))
    assert_close(transpose(z)[x], 4 * y)
    assert_close(transpose(z)[y], 4 * x)


def test_batched_matmul():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4], b=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5], b=Bint[2]))
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, "j")
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)
    assert_close(transpose(z)[xy], Number(1.0))


def test_expand_reduce():
    i = Variable("i", Bint[3])
    x = Variable("x", Real)
    y = Variable("y", Real)
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, i)
    assert_close(transpose(z)[x], 3 * y)
    assert_close(transpose(z)[y], 3 * x)
    assert_close(transpose(z)[xy], Number(1.0))


def test_tensor_contract():
    x = random_tensor(OrderedDict(i1=Bint[2], i2=Bint[3], j1=Bint[4], j2=Bint[5]))
    y = random_tensor(OrderedDict(j1=Bint[4], j2=Bint[5], k1=Bint[6], k2=Bint[7]))
    with interpretation(lazy):
        xy = x * y
        z = xy.reduce(ops.add, {"j1", "j2"})
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)
    assert_close(transpose(z)[xy], Number(1.0))


@pytest.mark.parametrize("height", [1, 2, 3, 10])  # , 100, 1000])
def test_tower_sum(height):
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    with interpretation(lazy):
        top = x
        for _ in range(height):
            top = top + top
    assert transpose(top)[x] is Number(2.0 ** height)


@pytest.mark.parametrize("height", [0, 1, 2, 3, 10])  # , 100, 1000])
def test_tower_prod(height):
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    with interpretation(lazy):
        top = x
        expected = Number(1.0)
        for _ in range(height):
            expected = top * expected + expected * top
            top = top * top
        # This might be too much to ask:
        # assert transpose(top)[x] is expected

    # This should definitely hold:
    assert_close(transpose(top)[x], reinterpret(expected))


@pytest.mark.parametrize("f", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
@pytest.mark.parametrize("g", ["x", "y", "x + y", "1 + x * y", "2 * x - y"])
def test_binary_product_rule(f, g):
    x = Variable("x", Real)
    y = Variable("y", Real)
    with interpretation(lazy):
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
    with interpretation(lazy):
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


@pytest.mark.xfail(reason="getitem adjoint not yet implemented")
def test_reduce_sum_getitem_variable():
    x = Variable("x", Reals[4])
    i = Variable("i", Bint[4])
    with interpretation(lazy):
        f = x[i]
        y = f.reduce(ops.add, i)
    actual = transpose(y)[x]
    expected = transpose(f)[x]
    print(f"actual:\n{str(actual)}")
    print(f"expected:\n{str(expected)}")
    assert actual.input_vars <= expected.input_vars
    assert actual.output == expected.output
    if actual is not expected:
        for x_, y_ in [(0.0, 0.1), (-1.3, 0.2), (0.3, 1.5)]:
            assert_close(actual(x=x_, y=y_), expected(x=x_, y=y_))


def test_adjoint_subs_variable():

    w = Variable("w", Real)
    x = Variable("x", Real)
    y = Variable("y", Real)
    with interpretation(reflect):
        xy = x + y
        z = xy(x=w)

    # x = Scatter(dest, subs, src) <==> transpose(x)[src] == dest(**subs)
    # zero(**subs) = out_adj <== notation ==> Scatter(zero, subs, out_adj)
    # in tensor-land: out = in[indices] <==> transpose(out)[in] == zero[indices] = out_adj

    with interpretation(lazy):
        assert transpose(z)[y] is Number(1.0)
        # assert transpose(z)[xy] is Scatter(Number(1.), ((x, w),), Number(0.))
        assert transpose(z)[x] is Number(0.0)  # XXX 0 or 1 or w?
        assert transpose(z)[w] is Number(1.0)


def test_adjoint_subs_tensor():

    i = Variable("i", Bint[2])
    j = Variable("j", Bint[2])
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[2]))
    with interpretation(reflect):
        y = x(i=0)

    # concretely
    expected = Tensor(torch.tensor([1.0, 0.0]))["i"]
    assert_close(transpose(y)[x], expected)

    # conceptually
    expected = Scatter(ops.add, (("i", Number(0, 2)),), Number(1.0))
    assert_close(transpose(y)[x], expected)


def test_adjoint_subs_tensor_rename():

    i = Variable("i", Bint[2])
    j = Variable("j", Bint[2])
    k = Variable("k", Bint[2])
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[2]))
    with interpretation(reflect):
        y = x(i=k)

    # concretely
    expected = Tensor(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))["i", "k"]
    assert_close(transpose(y)[x], expected)

    # conceptually
    expected = Scatter(ops.add, (("i", k),), Number(1.0))
    assert_close(transpose(y)[x], expected)


@pytest.mark.xfail(reason="requires ops.scatter_add")
def test_adjoint_subs_tensor_expand():

    k = Tensor(torch.tensor([0, 0, 1, 1]), OrderedDict(k=Bint[4]), 2)
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[2]))
    with interpretation(reflect):
        y = x(i=k)

    # conceptually
    expected = Scatter(ops.add, (("i", k),), Number(1.0))
    assert_close(transpose(y)[x], expected)

    # concretely
    expected = Tensor(torch.tensor([2.0, 2.0]))["i"]
    # or expected = Number(2)?
    assert_close(transpose(y)[x], expected)


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
