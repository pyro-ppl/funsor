# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.adjoint import adjoint
from funsor.domains import Bint, Real, Reals
from funsor.einsum import einsum
from funsor.interpretations import lazy, reflect
from funsor.interpreter import reinterpret
from funsor.tensor import Tensor, numeric_array
from funsor.terms import Number, Scatter, Slice, Variable
from funsor.testing import assert_close, make_einsum_example, random_tensor
from funsor.util import get_backend


def assert_close_extensional(actual, expected):
    zero = actual - expected
    probe = {k: random_tensor(OrderedDict(), d) for k, d in zero.inputs.items()}
    zero = zero(**probe)
    assert_close(zero, zero * 0)


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
    with lazy:
        y = x + x
    assert transpose(y)[x] is Number(2.0)
    assert transpose(y)[y] is Number(1.0)


@pytest.mark.xfail(reason="missing pattern for subtraction")
def test_zero_minus():
    x = random_tensor(OrderedDict(i=Bint[2]))
    with lazy:
        y = x - x
    assert transpose(y)[x] is Number(0.0)
    assert transpose(y)[y] is Number(1.0)


def test_zero_mul():
    x = random_tensor(OrderedDict(i=Bint[2]))
    with lazy:
        y = x * 0
    assert transpose(y)[x] is Number(0.0)
    assert transpose(y)[y] is Number(1.0)


def test_four():
    x = random_tensor(OrderedDict(i=Bint[2]))
    with lazy:
        y = x + x
        z = y + y
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_four_variable():
    x = Variable("x", Real)
    with lazy:
        y = x + x
        z = y + y
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_eight_tensor():
    w = random_tensor(OrderedDict(i=Bint[2]))
    with lazy:
        x = w + w
        y = x + x
        z = y + y
    assert transpose(z)[w] is Number(8.0)
    assert transpose(z)[x] is Number(4.0)
    assert transpose(z)[y] is Number(2.0)
    assert transpose(z)[z] is Number(1.0)


def test_eight_variable():
    w = Variable("w", Real)
    with lazy:
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
    with lazy:
        z = x * y
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)


def test_mul_variable_2():
    x = Variable("x", Real)
    y = Variable("y", Real)
    with lazy:
        z = x * y
    assert transpose(z)[x] is y
    assert transpose(z)[y] is x


def test_mul_variable_3():
    w = Variable("w", Real)
    x = Variable("x", Real)
    y = Variable("y", Real)
    with lazy:
        z = w * x * y
    assert_close_extensional(transpose(z)[w], x * y)
    assert_close_extensional(transpose(z)[x], w * y)
    assert_close_extensional(transpose(z)[y], w * x)


def test_sum():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4]))
    with lazy:
        z = x.reduce(ops.add, "j")
    assert_close(transpose(z)[x], Number(1.0))


def test_matmul_tensor():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5]))
    with lazy:
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
    with lazy:
        xy = x * y
        z = xy.reduce(ops.add, j)
    assert xy in transpose(z)
    assert_close(transpose(z)[xy], Number(4))
    assert_close_extensional(transpose(z)[x], 4 * y)
    assert_close_extensional(transpose(z)[y], 4 * x)


def test_batched_matmul():
    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[4], b=Bint[2]))
    y = random_tensor(OrderedDict(j=Bint[4], k=Bint[5], b=Bint[2]))
    with lazy:
        xy = x * y
        z = xy.reduce(ops.add, "j")
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)
    assert_close(transpose(z)[xy], Number(1.0))


def test_expand_reduce():
    i = Variable("i", Bint[3])
    x = Variable("x", Real)
    y = Variable("y", Real)
    with lazy:
        xy = x * y
        z = xy.reduce(ops.add, i)  # note the implicit expand before the .reduce
    assert_close_extensional(transpose(z)[x], 3 * y)
    assert_close_extensional(transpose(z)[y], 3 * x)
    assert_close(transpose(z)[xy], Number(3.0))


def test_tensor_contract():
    x = random_tensor(OrderedDict(i1=Bint[2], i2=Bint[3], j1=Bint[4], j2=Bint[5]))
    y = random_tensor(OrderedDict(j1=Bint[4], j2=Bint[5], k1=Bint[6], k2=Bint[7]))
    with lazy:
        xy = x * y
        z = xy.reduce(ops.add, {"j1", "j2"})
    assert_close(transpose(z)[x], y)
    assert_close(transpose(z)[y], x)
    assert_close(transpose(z)[xy], Number(1.0))


@pytest.mark.parametrize("height", [1, 2, 3, 10])  # , 100, 1000])
def test_tower_sum(height):
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    with lazy:
        top = x
        for _ in range(height):
            top = top + top
    assert transpose(top)[x] is Number(2.0**height)


# Note: we get overflow issue for height=10.
@pytest.mark.parametrize("height", [0, 1, 2, 3, 9])  # , 10, 100, 1000])
def test_tower_prod(height):
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    with lazy:
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

    if "-" in f or "-" in g:
        pytest.xfail(reason="missing pattern for subtraction")

    x = Variable("x", Real)
    y = Variable("y", Real)
    with lazy:
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

    if "-" in f or "-" in g:
        pytest.xfail(reason="missing pattern for subtraction")

    x = Variable("x", Real)
    y = Variable("y", Real)
    with lazy:
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
    with lazy:
        f_ = f.reduce(ops.add, i)

    actual = transpose(f_)[f]
    expected = Number(1.0)
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
    with lazy:
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
    with reflect:
        xy = x + y
        z = xy(x=w)

    assert transpose(z)[y] is Number(1.0)
    assert transpose(z)[xy] is Scatter(ops.add, (("x", w),), Number(1.0), frozenset())
    assert transpose(z)[x] is Number(1.0)  # FIXME is this right?
    assert transpose(z)[w] is Number(0.0)  # FIXME is this right?


def test_adjoint_subs_tensor():

    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[2]))
    with reflect:
        y = x(i=0)

    # concretely
    expected = Tensor(numeric_array([1.0, 0.0]))["i"]
    assert_close(transpose(y)[x], expected)

    # conceptually
    expected = Scatter(ops.add, (("i", Number(0, 2)),), Number(1.0), frozenset())
    assert_close(transpose(y)[x], expected)


def test_adjoint_subs_tensor_rename():

    k = Variable("k", Bint[2])
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[2]))
    with reflect:
        y = x(i=k)

    # concretely
    expected = Number(1.0)
    assert_close(transpose(y)[x], expected)

    # conceptually
    expected = Scatter(ops.add, (("i", k),), Number(1.0), frozenset())
    assert_close(transpose(y)[x], expected)


@pytest.mark.xfail(reason="not meaningful without a final reduce?")
def test_adjoint_subs_binary():

    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    y = random_tensor(OrderedDict(i=Bint[4], j=Bint[2]))
    k = Variable("k", Bint[2])

    with reflect:
        xk = x(i=k)
        yk = y(j=k)
        z = xk * yk

    assert_close(transpose(z)[xk], y(j=k))
    assert_close(transpose(z)[yk], x(i=k))

    expected_x = y(j=k).reduce(ops.add, "i")(k="i")
    actual_x = transpose(z)[x]
    assert_close(actual_x, expected_x)

    expected_y = x(i=k).reduce(ops.add, "j")(k="j")
    actual_y = transpose(z)[y]
    assert_close(actual_y, expected_y)


def test_adjoint_subs_binary_reduce():

    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[3]))
    y = random_tensor(OrderedDict(i=Bint[4], j=Bint[2]))
    k = Variable("k", Bint[2])

    with reflect:
        xk = x(i=k)
        yk = y(j=k)
        zk = xk * yk
        z = zk.reduce(ops.add, {"i", "j", "k"})

    expected_x = y(j=k).reduce(ops.add, "i")(k="i")
    actual_x = transpose(z)[x]
    assert_close(actual_x, expected_x)

    expected_y = x(i=k).reduce(ops.add, "j")(k="j")
    actual_y = transpose(z)[y]
    assert_close(actual_y, expected_y)


def test_adjoint_subs_binary_reduce_simple_1():

    x = random_tensor(OrderedDict(i=Bint[2]))
    y = random_tensor(OrderedDict(i=Bint[2]))
    k = Variable("k", Bint[2])

    with reflect:
        xk = x(i=k)
        yk = y(i=k)
        zk = xk * yk
        z = zk.reduce(ops.add)

    expected_x = y(i=k)(k="i")
    actual_x = transpose(z)[x]
    assert_close(actual_x, expected_x)


def test_adjoint_subs_binary_reduce_simple_2():

    x = random_tensor(OrderedDict(i=Bint[3], j=Bint[2]))
    y = random_tensor(OrderedDict(i=Bint[2]))
    k = Variable("k", Bint[2])

    with reflect:
        xk = x(j=k)
        yk = y(i=k)
        zk = xk * yk
        z = zk.reduce(ops.add)

    expected_x = y(i=k)(k="j")
    actual_x = transpose(z)[x]
    assert_close(actual_x, expected_x)

    expected_y = x(j=k).reduce(ops.add, "i")(k="i")
    actual_y = transpose(z)[y]
    assert_close(actual_y, expected_y)


def test_adjoint_binary_reduce_simple_2():

    x = random_tensor(OrderedDict(i=Bint[3], k=Bint[2]))
    y = random_tensor(OrderedDict(k=Bint[2]))

    with reflect:
        zk = x * y
        z = zk.reduce(ops.add)

    expected_x = y
    actual_x = transpose(z)[x]
    assert_close(actual_x, expected_x)

    expected_y = x.reduce(ops.add, "i")
    actual_y = transpose(z)[y]
    assert_close(actual_y, expected_y)


def test_adjoint_subs_binary_reduce_slices():

    x = random_tensor(OrderedDict(t=Bint[2], i=Bint[2], j=Bint[2]))
    k = Variable("k", Bint[2])

    with reflect:
        xik = x(i=k)
        xjk = x(j=k)
        xk = xik(t=Slice("t", 0, 2, 2, 2))
        yk = xjk(t=Slice("t", 1, 2, 2, 2))
        zk = xk * yk
        z = zk.reduce(ops.add, {"k"}).reduce(ops.mul, "t")
        z = z(i="i", j="j")
        z = z.reduce(ops.add, {"i", "j"})

    expected_x = transpose(z)[xik](k="i") + transpose(z)[xjk](k="j")
    actual_x = transpose(z)[x]
    assert_close(actual_x, expected_x.align(tuple(actual_x.inputs)))


@pytest.mark.xfail(reason="requires ops.scatter_add")
def test_adjoint_subs_tensor_expand():

    k = Tensor(numeric_array([0, 0, 1, 1]), OrderedDict(k=Bint[4]), 2)
    x = random_tensor(OrderedDict(i=Bint[2], j=Bint[2]))
    with reflect:
        y = x(i=k)

    # conceptually
    expected = Scatter(ops.add, (("i", k),), Number(1.0), frozenset())
    assert_close(transpose(y)[x], expected)

    # concretely
    expected = Tensor(numeric_array([2.0, 2.0]))["i"]
    # or expected = Number(2)?
    assert_close(transpose(y)[x], expected)


@pytest.mark.parametrize(
    "equation,plates",
    [
        (",->", ""),
        ("a,b->", ""),
        ("ab,a->", ""),
        ("a,b,c->", ""),
        ("a,a->", ""),
        ("a,a,a,ab->", ""),
        ("abc,bcd,cde->", ""),
        ("abc,cbd,edc->", ""),
        ("ab,bc,cd->", ""),
        ("ab,b,bc,c,cd,d->", ""),
        (",i->", "i"),
        (",ai,abij->", "ij"),
        ("a,ai,bij->", "ij"),
        ("a,ai,abi,bci,cdi->", "i"),
        ("a,aij,abij,bcij->", "ij"),
        ("a,abi,bcij,cdij->", "ij"),
    ],
)
def test_einsum_adjoint_vs_forward(equation, plates):
    backend = "jax.numpy" if get_backend() == "jax" else get_backend()
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with reflect:
        fwd_expr = einsum(equation, *funsor_operands, plates=plates, backend=backend)
    actuals = transpose(fwd_expr)

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        if set(plates) & set(inp):
            continue  # skip this term - can't write its adjoint as an einsum
        actual = actuals[fv]
        eqn_expected = ",".join(inputs[:i] + inputs[i + 1 :]) + "->" + inp
        operands_expected = funsor_operands[:i] + funsor_operands[i + 1 :]
        expected = einsum(
            eqn_expected, *operands_expected, plates=plates, backend=backend
        )
        assert_close(actual, expected.align(tuple(actual.inputs)), atol=1e-4, rtol=None)


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
