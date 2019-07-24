from __future__ import absolute_import, division, print_function

import pytest
import torch

from collections import OrderedDict

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.domains import bint, reals
from funsor.einsum import einsum, naive_plated_einsum
from funsor.gaussian import Gaussian
from funsor.interpreter import interpretation, reinterpret
from funsor.terms import Number, Variable, eager, moment_matching, normalize, reflect
from funsor.testing import assert_close, check_funsor, make_einsum_example  # , xfail_param
from funsor.torch import Tensor


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
    if plates:
        assert all(isinstance(v, (Number, Tensor, Contraction)) for v in transformed_expr.terms)
    else:
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
    ("-(y+z)", Contraction, {"y": reals(), "z": reals()}),
    # xfail_param(('-x', Contraction, {"x": reals()}), reason="not a contraction"),
    ('t-x', Contraction, {"i": bint(2), "j": bint(3), "x": reals()}),
    ("(t * x)(i=1)", Contraction, {"j": bint(3), "x": reals()}),
    ("(t * x)(i=1, x=y)", Contraction, {"j": bint(3), "y": reals()}),
    ("(t * x + n)(x=y)", Contraction, {"y": reals(), "i": bint(2), "j": bint(3)}),
    ("(x + y)(y=z)", Contraction, {"x": reals(), "z": reals()}),
    # xfail_param(("(-x)(x=y+z)", Contraction, {"y": reals(), "z": reals()}), reason="not a contraction"),
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


@pytest.mark.xfail(reason="pattern collisions with Joint")
def test_reduce_moment_matching_univariate():
    int_inputs = [('i', bint(2))]
    real_inputs = [('x', reals())]
    inputs = OrderedDict(int_inputs + real_inputs)
    int_inputs = OrderedDict(int_inputs)
    real_inputs = OrderedDict(real_inputs)

    p = 0.8
    t = 1.234
    s1, s2, s3 = 2.0, 3.0, 4.0
    loc = torch.tensor([[-s1], [s1]])
    precision = torch.tensor([[[s2 ** -2]], [[s3 ** -2]]])
    discrete = Tensor(torch.tensor([1 - p, p]).log() + t, int_inputs)
    gaussian = Gaussian(loc, precision, inputs)

    with interpretation(normalize):
        joint = discrete + gaussian

    with interpretation(moment_matching):
        actual = joint.reduce(ops.logaddexp, 'i')

    expected_loc = torch.tensor([(2 * p - 1) * s1])
    expected_variance = (4 * p * (1 - p) * s1 ** 2
                         + (1 - p) * s2 ** 2
                         + p * s3 ** 2)
    expected_precision = torch.tensor([[1 / expected_variance]])
    expected_gaussian = Gaussian(expected_loc, expected_precision, real_inputs)
    expected_discrete = Tensor(torch.tensor(t))
    expected = expected_discrete + expected_gaussian
    assert_close(actual, expected, atol=1e-5, rtol=None)
