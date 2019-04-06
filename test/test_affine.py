from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from funsor.affine import Affine
from funsor.domains import bint, reals
from funsor.terms import Number, Variable
from funsor.torch import Tensor


SMOKE_TESTS = [
    ('t+x', Affine),
    ('x+t', Affine),
    ('n+x', Affine),
    ('n*x', Affine),
    ('t*x', Affine),
    ('x*t', Affine),
    ('-x', Affine),
    ('t-x', Affine),
]


@pytest.mark.parametrize('expr,expected_type', SMOKE_TESTS)
def test_smoke(expr, expected_type):

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    n = Number(2.)
    assert isinstance(n, Number)

    x = Variable('x', reals())
    assert isinstance(x, Variable)

    y = Variable('y', reals())
    assert isinstance(y, Variable)

    result = eval(expr)
    assert isinstance(result, expected_type)
