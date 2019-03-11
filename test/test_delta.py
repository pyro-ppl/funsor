from __future__ import absolute_import, division, print_function

import torch

import funsor.ops as ops
from funsor.delta import Delta
from funsor.domains import reals
from funsor.terms import Number, Variable
from funsor.testing import check_funsor
from funsor.torch import Tensor


def test_eager_subs_variable():
    v = Variable('v', reals(3))
    point = Tensor(torch.randn(3))
    d = Delta('foo', v)
    assert d(v=point) is Delta('foo', point)


def test_eager_subs_ground():
    point1 = Tensor(torch.randn(3))
    point2 = Tensor(torch.randn(3))
    d = Delta('foo', point1)
    check_funsor(d(foo=point1), {}, reals(), torch.tensor(0.))
    check_funsor(d(foo=point2), {}, reals(), torch.tensor(-float('inf')))


def test_add_delta_funsor():
    x = Variable('x', reals(3))
    y = Variable('y', reals(3))
    d = Delta('x', y)

    expr = -(1 + x ** 2).log()
    assert d + expr is d + expr(x=y)
    assert expr + d is expr(x=y) + d


def test_reduce():
    point = Tensor(torch.randn(3))
    d = Delta('foo', point)
    assert d.reduce(ops.logaddexp, frozenset(['foo'])) is Number(0)
