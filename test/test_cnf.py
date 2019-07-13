from __future__ import absolute_import, division, print_function

import pytest
import torch

from collections import OrderedDict

from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.einsum import einsum, naive_einsum
from funsor.interpreter import interpretation, reinterpret
from funsor.terms import Number, Variable, eager, normalize, reflect
from funsor.testing import assert_close, check_funsor, make_einsum_example
from funsor.torch import Tensor


EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    "a,a->",
    "a,a->a",
    "ab,bc,cd->da",
    "ab,cd,bc->da",
    "a,a,a,ab->ab",
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize('einsum_impl', [einsum, naive_einsum])
def test_normalize_einsum(equation, backend, einsum_impl):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        expr = einsum_impl(equation, *funsor_operands, backend=backend)

    with interpretation(normalize):
        transformed_expr = reinterpret(expr)

    assert isinstance(transformed_expr, Contraction)
    assert all(isinstance(v, (Number, Tensor)) for v in transformed_expr.terms)

    with interpretation(normalize):
        transformed_expr2 = reinterpret(transformed_expr)

    assert transformed_expr2 is transformed_expr  # check normalization

    with interpretation(eager):
        actual = reinterpret(transformed_expr)
        expected = reinterpret(expr)

    assert_close(actual, expected, rtol=1e-4)


AFFINE_SMOKE_TESTS = [
    ('t+x', Contraction, {"i": bint(2), "j": bint(3), "x": reals()}),
    ('x+t', Contraction, {"x": reals(), "i": bint(2), "j": bint(3)}),
    ('n+x', Contraction, {"x": reals()}),
    ('n*x', Contraction, {"x": reals()}),
    ('t*x', Contraction, {"i": bint(2), "j": bint(3), "x": reals()}),
    ('x*t', Contraction, {"x": reals(), "i": bint(2), "j": bint(3)}),
    # # ('-x', Contraction, {"x": reals()}),
    ('t-x', Contraction, {"i": bint(2), "j": bint(3), "x": reals()}),
    ("(t * x)(i=1)", Contraction, {"j": bint(3), "x": reals()}),
    ("(t * x)(i=1, x=y)", Contraction, {"j": bint(3), "y": reals()}),
    ("(t * x + n)(x=y)", Contraction, {"y": reals(), "i": bint(2), "j": bint(3)}),
    ("(x + y)(y=z)", Contraction, {"x": reals(), "z": reals()}),
    # ("(-x)(x=y+z)", Contraction, {"y": reals(), "z": reals()}),
    ("(t * x + t * y)(x=z)", Contraction, {"y": reals(), "z": reals(), "i": bint(2), "j": bint(3)}),
]


@pytest.mark.parametrize("expr,expected_type,expected_inputs", AFFINE_SMOKE_TESTS)
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

    with interpretation(normalize):
        result = eval(expr)

    assert isinstance(result, expected_type)
    check_funsor(result, expected_inputs, expected_output)
