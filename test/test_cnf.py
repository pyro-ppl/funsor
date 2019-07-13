from __future__ import absolute_import, division, print_function

import pytest
import torch

from collections import OrderedDict

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.einsum import einsum, naive_einsum
from funsor.gaussian import Gaussian
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
    ('-x', Contraction, {"x": reals()}),
    ('t-x', Contraction, {"i": bint(2), "j": bint(3), "x": reals()}),
    ("(t * x)(i=1)", Contraction, {"j": bint(3), "x": reals()}),
    ("(t * x)(i=1, x=y)", Contraction, {"j": bint(3), "y": reals()}),
    ("(t * x + n)(x=y)", Contraction, {"y": reals(), "i": bint(2), "j": bint(3)}),
    ("(x + y)(y=z)", Contraction, {"x": reals(), "z": reals()}),
    ("(-x)(x=y+z)", Contraction, {"y": reals(), "z": reals()}),
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


JOINT_SMOKE_TESTS = [
    ('dx + dy', Contraction),
    ('dx + g', Contraction),
    ('dy + g', Contraction),
    ('g + dx', Contraction),
    ('g + dy', Contraction),
    ('dx + t', Contraction),
    ('dy + t', Contraction),
    ('dx - t', Contraction),
    ('dy - t', Contraction),
    ('t + dx', Contraction),
    ('t + dy', Contraction),
    ('g + 1', Contraction),
    ('g - 1', Contraction),
    ('1 + g', Contraction),
    ('g + t', Contraction),
    ('g - t', Contraction),
    ('t + g', Contraction),
    ('t - g', Contraction),
    ('g + g', Contraction),
    ('-(g + g)', Contraction),
    ('(dx + dy)(i=i0)', Contraction),
    ('(dx + g)(i=i0)', Contraction),
    ('(dy + g)(i=i0)', Contraction),
    ('(g + dx)(i=i0)', Contraction),
    ('(g + dy)(i=i0)', Contraction),
    ('(dx + t)(i=i0)', Contraction),
    ('(dy + t)(i=i0)', Contraction),
    ('(dx - t)(i=i0)', Contraction),
    ('(dy - t)(i=i0)', Contraction),
    ('(t + dx)(i=i0)', Contraction),
    ('(t + dy)(i=i0)', Contraction),
    ('(g + 1)(i=i0)', Contraction),
    ('(g - 1)(i=i0)', Contraction),
    ('(1 + g)(i=i0)', Contraction),
    ('(g + t)(i=i0)', Contraction),
    ('(g - t)(i=i0)', Contraction),
    ('(t + g)(i=i0)', Contraction),
    ('(g + g)(i=i0)', Contraction),
    ('(dx + dy)(x=x0)', Contraction),
    ('(dx + g)(x=x0)', Tensor),
    ('(dy + g)(x=x0)', Contraction),
    ('(g + dx)(x=x0)', Tensor),
    ('(g + dy)(x=x0)', Contraction),
    ('(dx + t)(x=x0)', Tensor),
    ('(dy + t)(x=x0)', Contraction),
    ('(dx - t)(x=x0)', Tensor),
    ('(dy - t)(x=x0)', Contraction),
    ('(t + dx)(x=x0)', Tensor),
    ('(t + dy)(x=x0)', Contraction),
    ('(g + 1)(x=x0)', Tensor),
    ('(g - 1)(x=x0)', Tensor),
    ('(1 + g)(x=x0)', Tensor),
    ('(g + t)(x=x0)', Tensor),
    ('(g - t)(x=x0)', Tensor),
    ('(t + g)(x=x0)', Tensor),
    ('(g + g)(x=x0)', Tensor),
    ('(g + dy).reduce(ops.logaddexp, "x")', Contraction),
    ('(g + dy).reduce(ops.logaddexp, "y")', Gaussian),
    ('(t + g + dy).reduce(ops.logaddexp, "x")', Contraction),
    ('(t + g + dy).reduce(ops.logaddexp, "y")', Contraction),
    ('(t + g).reduce(ops.logaddexp, "x")', Tensor),
]


@pytest.mark.parametrize('expr,expected_type', JOINT_SMOKE_TESTS)
def test_joint_smoke(expr, expected_type):
    dx = Delta('x', Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2))])))
    assert isinstance(dx, Delta)

    dy = Delta('y', Tensor(torch.randn(3, 4), OrderedDict([('j', bint(3))])))
    assert isinstance(dy, Delta)

    t = Tensor(torch.randn(2, 3), OrderedDict([('i', bint(2)), ('j', bint(3))]))
    assert isinstance(t, Tensor)

    g = Gaussian(
        loc=torch.tensor([[0.0, 0.1, 0.2],
                          [2.0, 3.0, 4.0]]),
        precision=torch.tensor([[[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]],
                                [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g, Gaussian)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    with interpretation(normalize):
        result = eval(expr)

    if expected_type is not Contraction:
        assert isinstance(result, Contraction)
        result = reinterpret(result)

    assert isinstance(result, expected_type)
