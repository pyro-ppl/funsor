# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

import funsor.ops as ops
from funsor.constant import Constant
from funsor.delta import Delta
from funsor.domains import Bint, Real
from funsor.tensor import Tensor
from funsor.terms import Number, Variable, to_data, to_funsor
from funsor.testing import assert_close, randn
from funsor.util import get_backend


def test_eager_subs_variable():
    v = Variable("v", Real)
    data = Tensor(randn(3))
    c = Constant(OrderedDict(b=Real), data)
    assert c(b=v) is Constant(OrderedDict(v=Real), data)


def test_eager_subs_ground():
    data = Tensor(randn(3))
    c = Constant(OrderedDict(x=Bint[3], y=Real), data)
    t = Tensor(randn(2))["a"]
    assert c(x=0) is Constant(OrderedDict(y=Real), data)
    assert c(y=t) is Constant(OrderedDict(x=Bint[3], a=Bint[2]), data)
    assert c(x=0, y=1) is data


def test_unary_exp():
    z = Tensor(randn(2))["z"]
    c = Constant(OrderedDict(x=Bint[3], y=Real), z)

    result = c.exp()
    assert result.const_inputs == c.const_inputs
    assert_close(result.arg, c.arg.exp())


def test_reduce():
    c = Constant(OrderedDict(x=Bint[3], y=Bint[2]), Number(1))
    assert c.reduce(ops.add, {"x", "y"}) is Number(6)


def test_add_constant_funsor():
    x = Tensor(randn(3))["x"]
    z = Tensor(randn(2))["z"]
    c = Constant(OrderedDict(x=Bint[3], y=Real), z)

    result = c + x  # Constant(OrderedDict(y=Real), z + x)
    assert result.const_inputs == OrderedDict(y=Real)
    assert_close(result.arg, c.arg + x)

    result = x + c  # Constant(OrderedDict(y=Real), x + z)
    assert result.const_inputs == OrderedDict(y=Real)
    assert_close(result.arg, x + c.arg)


def test_add_constant_delta():
    z = Tensor(randn(2))["z"]
    c = Constant(OrderedDict(x=Bint[3], y=Real), z)

    point1 = Number(5.0)
    d1 = Delta("y", point1)
    assert c + d1 is c(y=point1) + d1
    assert d1 + c is d1 + c(y=point1)

    point2 = Tensor(randn(2))["z"]
    d2 = Delta("y", point2)
    assert c + d2 is c(y=point2) + d2
    assert d2 + c is d2 + c(y=point2)


@pytest.mark.skipif(
    get_backend() != "torch",
    reason="numpy/jax backend has not implemented ProvenanceTensor",
)
def test_to_funsor():
    import torch

    from funsor.torch.provenance import ProvenanceTensor

    data = torch.zeros(3, 3)
    pt = ProvenanceTensor(data, frozenset({("x", Real)}))
    c = to_funsor(pt)
    assert c is Constant(OrderedDict(x=Real), Tensor(data))


@pytest.mark.skipif(
    get_backend() != "torch",
    reason="numpy/jax backend has not implemented ProvenanceTensor",
)
def test_to_data():
    import torch

    from funsor.torch.provenance import ProvenanceTensor

    data = torch.zeros(3, 3)
    c = Constant(OrderedDict(x=Real), Tensor(data))
    pt = to_data(c)
    assert isinstance(pt, ProvenanceTensor)
    assert pt._t is data
    assert pt._provenance == frozenset({("x", Real)})
