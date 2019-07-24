from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.terms import Number, Variable
from funsor.testing import check_funsor
from funsor.torch import Tensor

SMOKE_TESTS = [
    ('t+x', Contraction),
    ('x+t', Contraction),
    ('n+x', Contraction),
    ('n*x', Contraction),
    ('t*x', Contraction),
    ('x*t', Contraction),
    ('-x', Contraction),
    ('t-x', Contraction),
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
    assert result.is_affine()


SUBS_TESTS = [
    ("(t * x)(i=1)", Contraction, {"j": bint(3), "x": reals()}),
    ("(t * x)(i=1, x=y)", Contraction, {"j": bint(3), "y": reals()}),
    ("(t * x + n)(x=y)", Contraction, {"y": reals(), "i": bint(2), "j": bint(3)}),
    ("(x + y)(y=z)", Contraction, {"x": reals(), "z": reals()}),
    ("(-x)(x=y+z)", Contraction, {"y": reals(), "z": reals()}),
    ("(t * x + t * y)(x=z)", Contraction, {"y": reals(), "z": reals(), "i": bint(2), "j": bint(3)}),
]


@pytest.mark.parametrize("expr,expected_type,expected_inputs", SUBS_TESTS)
def test_affine_subs(expr, expected_type, expected_inputs):

    expected_output = reals()

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    n = Number(2.)
    assert isinstance(n, Number)

    x = Variable('x', reals())
    assert isinstance(x, Variable)

    y = Variable('y', reals())
    assert isinstance(y, Variable)

    z = Variable('z', reals())
    assert isinstance(z, Variable)

    result = eval(expr)
    assert isinstance(result, expected_type)
    check_funsor(result, expected_inputs, expected_output)
    assert result.is_affine()
