from collections import OrderedDict

import pytest
import torch

from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.einsum import einsum, naive_plated_einsum
from funsor.interpreter import interpretation, reinterpret
from funsor.terms import Number, Variable, eager, normalize, reflect
from funsor.testing import assert_close, check_funsor, make_einsum_example, random_tensor
from funsor.torch import Einsum, Tensor
from funsor.util import quote

assert Variable  # flake8
assert bint  # flake8
assert reals  # flake8
assert torch  # flake8

EINSUM_EXAMPLES = [
    ("a,b->", ''),
    ("ab,a->", ''),
    ("a,a->", ''),
    ("a,a->a", ''),
    ("ab,bc,cd->da", ''),
    ("ab,cd,bc->da", ''),
    ("a,a,a,ab->ab", ''),
    ('i->', 'i'),
    (',i->', 'i'),
    ('ai->', 'i'),
    (',ai,abij->', 'ij'),
    ('a,ai,bij->', 'ij'),
    ('ai,abi,bci,cdi->', 'i'),
    ('aij,abij,bcij->', 'ij'),
    ('a,abi,bcij,cdij->', 'ij'),
]


@pytest.mark.parametrize('equation,plates', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize('einsum_impl', [einsum, naive_plated_einsum])
def test_normalize_einsum(equation, plates, backend, einsum_impl):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        expr = einsum_impl(equation, *funsor_operands, backend=backend, plates=plates)

    with interpretation(normalize):
        transformed_expr = reinterpret(expr)

    assert isinstance(transformed_expr, Contraction)
    check_funsor(transformed_expr, expr.inputs, expr.output)

    assert all(isinstance(v, (Number, Tensor, Contraction)) for v in transformed_expr.terms)

    with interpretation(normalize):
        transformed_expr2 = reinterpret(transformed_expr)

    assert transformed_expr2 is transformed_expr  # check normalization

    with interpretation(eager):
        actual = reinterpret(transformed_expr)
        expected = reinterpret(expr)

    assert_close(actual, expected, rtol=1e-4)

    actual = eval(quote(expected))  # requires torch, bint
    assert_close(actual, expected)


@pytest.mark.parametrize('expr', [
    "Variable('x', reals()) + Number(0.5)",
    "Variable('x', reals(2)) + Variable('y', reals(2))",
    "Variable('x', reals(2)) + Tensor(torch.ones(2))",
])
def test_extract_affine(expr):
    x = eval(expr)
    assert isinstance(x, Contraction)
    assert x.is_affine

    const, coeffs = x.extract_affine()
    assert isinstance(const, Tensor)
    assert const.shape == x.shape
    assert list(coeffs) == list(x.inputs)
    for name, (coeff, eqn) in coeffs.items():
        assert isinstance(name, str)
        assert isinstance(coeff, Tensor)
        assert isinstance(eqn, str)

    real_inputs = OrderedDict((k, d) for k, d in x.inputs.items()
                              if d.dtype == 'real')
    int_inputs = OrderedDict((k, d) for k, d in x.inputs.items()
                             if d.dtype != 'real')
    subs = {k: random_tensor(int_inputs, d) for k, d in real_inputs.items()}
    expected = x(**subs)
    assert isinstance(expected, Tensor)

    actual = const + sum(Einsum(eqn, (coeff, subs[k]))
                         for k, (coeff, eqn) in coeffs.items())
    assert isinstance(actual, Tensor)
    assert_close(actual, expected)
