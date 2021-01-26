# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import Array, Bint, Real, Reals
from funsor.factory import Bound, Fresh, make_funsor, to_funsor
from funsor.tensor import Tensor
from funsor.terms import Cat, Funsor, Lambda, Number, eager
from funsor.testing import check_funsor, random_tensor


def test_lambda_lambda():
    @make_funsor
    def LambdaLambda(
        i: Bound,
        j: Bound,
        x: Funsor,
    ) -> Fresh[lambda i, j, x: Array[x.dtype, (i.size, j.size) + x.shape]]:
        assert i in x.inputs
        assert j in x.inputs
        return Lambda(i, Lambda(j, x))


@pytest.mark.parametrize("num_inputs", [0, 1, 2])
def test_getitem_getitem(num_inputs):
    @make_funsor
    def GetitemGetitem(
        x: Funsor,
        i: Fresh[lambda x: Bint[x.shape[0]]],
        j: Fresh[lambda x: Bint[x.shape[1]]],
    ) -> Fresh[lambda x: Array[x.dtype, x.shape[2:]]]:
        return x[i][j]

    inputs = OrderedDict()
    for name in "abc"[:num_inputs]:
        inputs[name] = Bint[2]
    x = random_tensor(inputs, Reals[3, 4])

    expected = x["i"]["j"]
    actual = GetitemGetitem(x, "i", "j")
    check_funsor(actual, expected.inputs, expected.output, expected.data)


@pytest.mark.xfail(reason="missing pattern Variable //,% Number")
def test_flatten():
    @make_funsor
    def Flatten21(
        x: Funsor,
        i: Bound,
        j: Bound,
        ij: Fresh[lambda i, j: Bint[i.size * j.size]],
    ) -> Fresh[lambda x: x.dtype]:
        m = to_funsor(i, x.inputs.get(i, None)).output.size
        n = to_funsor(j, x.inputs.get(j, None)).output.size
        ij = to_funsor(ij, Bint[m * n])
        ij = x.materialize(ij)
        return x(**{i.name: ij // Number(n, m * n),
                    j.name: ij % Number(n, m * n)})

    inputs = OrderedDict()
    inputs["a"] = Bint[3]
    inputs["b"] = Bint[4]
    data = random_tensor(inputs, Real)
    x = Flatten21(data, "a", "b", "ab")

    check_funsor(x, {"ab": Bint[12]}, Real, data.data.reshape(-1))


def test_unflatten():
    @make_funsor
    def Unflatten(
        x: Funsor,
        i: Bound,
        i_over_2: Fresh[lambda i: Bint[i.size // 2]],
        i_mod_2: Fresh[lambda: Bint[2]],
    ) -> Fresh[lambda x: x]:
        assert i.output.size % 2 == 0
        return x(**{i.name: i_over_2 * Number(2, 3) + i_mod_2})

    inputs = OrderedDict()
    inputs["a"] = Bint[5]
    inputs["b"] = Bint[6]
    data = random_tensor(inputs, Real)
    x = Unflatten(data, "b", "c", "d")

    check_funsor(x, {"a": Bint[5], "c": Bint[3], "d": Bint[2]}, Real,
                 data.data.reshape(5, 3, 2))


def test_cat2():
    @make_funsor
    def Cat2(
        x: Funsor,
        y: Funsor,
        i: Bound,
        j: Bound,
        ij: Fresh[lambda i, j: Bint[i.size + j.size]],
    ) -> Fresh[lambda x: x]:
        y = y(**{j.name: i})
        result = Cat(i.name, (x, y))
        result = result(**{i.name: ij})
        return result

    inputs = OrderedDict()
    inputs["a"] = Bint[3]
    inputs["b"] = Bint[4]
    x = random_tensor(inputs, Real)
    y = random_tensor(inputs, Real)
    xy = Cat2(x, y, "a", "a", "aa")

    check_funsor(xy, {"aa": Bint[6], "b": Bint[4]}, Real)


def test_normal():
    @make_funsor
    def Normal(
        loc: Funsor,
        scale: Funsor,
        value: Fresh[lambda loc: loc],
    ) -> Fresh[Real]:
        return None

    @eager.register(Normal, Tensor, Tensor, Tensor)
    def _(loc, scale, value):
        z = (value - loc) / scale
        log_prob = (-0.5) * z ** 2 - 0.5 * math.log(2 * math.pi)
        return log_prob.sum()

    inputs = OrderedDict(i=Bint[3])
    loc = random_tensor(inputs)
    scale = random_tensor(inputs).exp()
    value = random_tensor(inputs)
    d = Normal(loc, scale, "value")
    assert isinstance(d, Normal)
    check_funsor(d, {"value": Real, "i": Bint[3]}, Real)

    actual = d(value=value)
    assert isinstance(actual, Tensor)
    check_funsor(actual, {"i": Bint[3]}, Real)


def test_matmul():
    @make_funsor
    def MatMul(
        x: Funsor,
        y: Funsor,
        i: Bound,
    ) -> Fresh[lambda x: x]:
        return (x * y).reduce(ops.add, i)

    x = random_tensor(OrderedDict(a=Bint[3], b=Bint[4]))
    y = random_tensor(OrderedDict(c=Bint[4], d=Bint[3]))
    xy = MatMul(x, y, "b")
    check_funsor(xy, {"a": Bint[3], "c": Bint[4], "d": Bint[3]}, Real)


@pytest.mark.xfail(reason="alpha conversion incorrectly changes key")
def test_scatter1():
    @make_funsor
    def Scatter1(
        destin: Funsor,
        key: Bound,
        key_: Fresh[lambda key: key],
        value: Funsor,
        source: Funsor,
    ) -> Fresh[lambda destin: destin]:
        return None

    destin = random_tensor(OrderedDict(a=Bint[9]))
    source = random_tensor(OrderedDict(b=Bint[3]))
    value = Number(4, 9)
    x = Scatter1(destin, "a", "a", value, source)
    check_funsor(x, {"a": Bint[9], "b": Bint[3]}, Real)
