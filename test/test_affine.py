# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest
import torch

from funsor.affine import extract_affine, is_affine
from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.terms import Number, Unary, Variable
from funsor.testing import assert_close, check_funsor, random_gaussian, random_tensor
from funsor.tensor import Einsum, Tensor

assert random_gaussian  # flake8

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
    assert is_affine(result)


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
    assert is_affine(result)


@pytest.mark.parametrize('expr', [
    "-Variable('x', reals())",
    "Variable('x', reals(2)).sum()",
    "Variable('x', reals()) + 0.5",
    "Variable('x', reals(2, 3)) + Variable('y', reals(2, 3))",
    "Variable('x', reals(2)) + Variable('y', reals(2))",
    "Variable('x', reals(2)) + torch.ones(2)",
    "Variable('x', reals(2)) * torch.randn(2)",
    "Variable('x', reals(2)) * torch.randn(2) + torch.ones(2)",
    "Variable('x', reals(2)) + Tensor(torch.randn(3, 2), OrderedDict(i=bint(3)))",
    "Einsum('abcd,ac->bd',"
    " (Tensor(torch.randn(2, 3, 4, 5)), Variable('x', reals(2, 4))))",
    "Tensor(torch.randn(3, 5)) + Einsum('abcd,ac->bd',"
    " (Tensor(torch.randn(2, 3, 4, 5)), Variable('x', reals(2, 4))))",
    "Variable('x', reals(2, 8))[0] + torch.randn(8)",
    "Variable('x', reals(2, 8))[Variable('i', bint(2))] / 4 - 3.5",
])
def test_extract_affine(expr):
    x = eval(expr)
    assert is_affine(x)
    assert isinstance(x, (Unary, Contraction, Einsum))
    real_inputs = OrderedDict((k, d) for k, d in x.inputs.items()
                              if d.dtype == 'real')

    const, coeffs = extract_affine(x)
    assert isinstance(const, Tensor)
    assert const.shape == x.shape
    assert list(coeffs) == list(real_inputs)
    for name, (coeff, eqn) in coeffs.items():
        assert isinstance(name, str)
        assert isinstance(coeff, Tensor)
        assert isinstance(eqn, str)

    subs = {k: random_tensor(OrderedDict(), d) for k, d in real_inputs.items()}
    expected = x(**subs)
    assert isinstance(expected, Tensor)

    actual = const + sum(Einsum(eqn, (coeff, subs[k]))
                         for k, (coeff, eqn) in coeffs.items())
    assert isinstance(actual, Tensor)
    assert_close(actual, expected)


@pytest.mark.parametrize("expr", [
    "Variable('x', reals()).log()",
    "Variable('x', reals()).exp()",
    "Variable('x', reals()).sigmoid()",
    "Variable('x', reals(2)).prod()",
    "Variable('x', reals()) ** 2",
    "Variable('x', reals()) ** 2",
    "2 ** Variable('x', reals())",
    "Variable('x', reals()) * Variable('x', reals())",
    "Variable('x', reals()) * Variable('y', reals())",
    "Variable('x', reals()) / Variable('y', reals())",
    "Variable('x', reals()) / Variable('y', reals())",
    "Variable('x', reals(2,3)) @ Variable('y', reals(3,4))",
    "random_gaussian(OrderedDict(x=reals()))",
    "Einsum('abcd,ac->bd',"
    " (Variable('y', reals(2, 3, 4, 5)), Variable('x', reals(2, 4))))",
])
def test_not_is_affine(expr):
    x = eval(expr)
    assert not is_affine(x)
