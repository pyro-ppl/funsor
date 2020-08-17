# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest

from funsor.affine import extract_affine, is_affine
from funsor.cnf import Contraction
from funsor.domains import Bint, Real, Reals  # noqa: F401
from funsor.terms import Number, Unary, Variable
from funsor.testing import assert_close, check_funsor, ones, randn, random_gaussian, random_tensor  # noqa: F401
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

    t = Tensor(randn(2, 3), OrderedDict([('i', Bint[2]), ('j', Bint[3])]))
    assert isinstance(t, Tensor)

    n = Number(2.)
    assert isinstance(n, Number)

    x = Variable('x', Real)
    assert isinstance(x, Variable)

    y = Variable('y', Real)
    assert isinstance(y, Variable)

    result = eval(expr)
    assert isinstance(result, expected_type)
    assert is_affine(result)


SUBS_TESTS = [
    ("(t * x)(i=1)", Contraction, {"j": Bint[3], "x": Real}),
    ("(t * x)(i=1, x=y)", Contraction, {"j": Bint[3], "y": Real}),
    ("(t * x + n)(x=y)", Contraction, {"y": Real, "i": Bint[2], "j": Bint[3]}),
    ("(x + y)(y=z)", Contraction, {"x": Real, "z": Real}),
    ("(-x)(x=y+z)", Contraction, {"y": Real, "z": Real}),
    ("(t * x + t * y)(x=z)", Contraction, {"y": Real, "z": Real, "i": Bint[2], "j": Bint[3]}),
]


@pytest.mark.parametrize("expr,expected_type,expected_inputs", SUBS_TESTS)
def test_affine_subs(expr, expected_type, expected_inputs):

    expected_output = Real

    t = Tensor(randn(2, 3), OrderedDict([('i', Bint[2]), ('j', Bint[3])]))
    assert isinstance(t, Tensor)

    n = Number(2.)
    assert isinstance(n, Number)

    x = Variable('x', Real)
    assert isinstance(x, Variable)

    y = Variable('y', Real)
    assert isinstance(y, Variable)

    z = Variable('z', Real)
    assert isinstance(z, Variable)

    result = eval(expr)
    assert isinstance(result, expected_type)
    check_funsor(result, expected_inputs, expected_output)
    assert is_affine(result)


@pytest.mark.parametrize('expr', [
    "-Variable('x', Real)",
    "Variable('x', Reals[2]).sum()",
    "Variable('x', Real) + 0.5",
    "Variable('x', Reals[2, 3]) + Variable('y', Reals[2, 3])",
    "Variable('x', Reals[2]) + Variable('y', Reals[2])",
    "Variable('x', Reals[2]) + ones(2)",
    "Variable('x', Reals[2]) * randn(2)",
    "Variable('x', Reals[2]) * randn(2) + ones(2)",
    "Variable('x', Reals[2]) + Tensor(randn(3, 2), OrderedDict(i=Bint[3]))",
    "Einsum('abcd,ac->bd',"
    " (Tensor(randn(2, 3, 4, 5)), Variable('x', Reals[2, 4])))",
    "Tensor(randn(3, 5)) + Einsum('abcd,ac->bd',"
    " (Tensor(randn(2, 3, 4, 5)), Variable('x', Reals[2, 4])))",
    "Variable('x', Reals[2, 8])[0] + randn(8)",
    "Variable('x', Reals[2, 8])[Variable('i', Bint[2])] / 4 - 3.5",
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
    "Variable('x', Real).log()",
    "Variable('x', Real).exp()",
    "Variable('x', Real).sigmoid()",
    "Variable('x', Reals[2]).prod()",
    "Variable('x', Real) ** 2",
    "Variable('x', Real) ** 2",
    "2 ** Variable('x', Real)",
    "Variable('x', Real) * Variable('x', Real)",
    "Variable('x', Real) * Variable('y', Real)",
    "Variable('x', Real) / Variable('y', Real)",
    "Variable('x', Real) / Variable('y', Real)",
    "Variable('x', Reals[2,3]) @ Variable('y', Reals[3,4])",
    "random_gaussian(OrderedDict(x=Real))",
    "Einsum('abcd,ac->bd',"
    " (Variable('y', Reals[2, 3, 4, 5]), Variable('x', Reals[2, 4])))",
])
def test_not_is_affine(expr):
    x = eval(expr)
    assert not is_affine(x)
