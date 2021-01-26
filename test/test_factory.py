# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.domains import Array, Bint, Real, Reals
from funsor.factory import Bound, Fresh, make_funsor, to_funsor
from funsor.tensor import Tensor
from funsor.terms import Cat, Funsor, Lambda, eager
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


@pytest.mark.xfail(reason="missing pattern Variable // Number")
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
        return x(**{i: ij // n, j: ij % n})

    inputs = OrderedDict()
    inputs["a"] = Bint[3]
    inputs["b"] = Bint[4]
    data = random_tensor(inputs, Real)
    x = Flatten21(data, "a", "b", "ab")

    check_funsor(x, {"ab": Bint[12]}, Real, data.data.reshape(-1))


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


def test_Normal():
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
