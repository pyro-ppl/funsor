from __future__ import absolute_import, division, print_function

import torch

from funsor.delta import Delta
from funsor.domains import reals
from funsor.terms import Variable
from funsor.torch import Tensor


def test_eager_subs():
    v = Variable('v', reals(3))
    point = Tensor(torch.randn(3))
    d = Delta('foo', v)
    assert d(v=point) is Delta('foo', point)


def test_add_delta_funsor():
    x = Variable('x', reals(3))
    y = Variable('y', reals(3))
    d = Delta('x', y)

    expr = -(1 + x ** 2).log()
    assert d + expr is d + expr(x=y)
    assert expr + d is expr(x=y) + d

