import pytest
import torch

from collections import OrderedDict

import funsor.ops as ops
from funsor.cnf import Contraction
from funsor.delta import Delta
from funsor.domains import bint, reals
from funsor.einsum import einsum, naive_plated_einsum
from funsor.gaussian import Gaussian
from funsor.interpreter import interpretation, reinterpret
from funsor.terms import Number, eager, moment_matching, normalize, reflect
from funsor.testing import assert_close, make_einsum_example  # , xfail_param
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
    # ('(g + dy).reduce(ops.logaddexp, "y")', Gaussian),  # TODO xfail when FUNSOR_NORMALIZE == 0
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
        result = reinterpret(result)

    assert isinstance(result, expected_type)


GAUSSIAN_SMOKE_TESTS = [
    ('g1 + 1', Contraction),
    ('g1 - 1', Contraction),
    ('1 + g1', Contraction),
    ('g1 + shift', Contraction),
    ('g1 + shift', Contraction),
    ('shift + g1', Contraction),
    ('shift - g1', Contraction),
    ('g1 + g1', Contraction),
    ('(g1 + g2 + g2) - g2', Contraction),
    ('g1(i=i0)', Gaussian),
    ('g2(i=i0)', Gaussian),
    ('g1(i=i0) + g2(i=i0)', Contraction),
    ('g1(i=i0) + g2', Contraction),
    ('g1(x=x0)', Tensor),
    ('g2(y=y0)', Tensor),
    ('(g1 + g2)(i=i0)', Contraction),
    ('(g1 + g2)(x=x0, y=y0)', Tensor),
    ('(g2 + g1)(x=x0, y=y0)', Tensor),
    ('g1.reduce(ops.logaddexp, "x")', Tensor),
    ('(g1 + g2).reduce(ops.logaddexp, "x")', Contraction),
    ('(g1 + g2).reduce(ops.logaddexp, "y")', Contraction),
    ('(g1 + g2).reduce(ops.logaddexp, frozenset(["x", "y"]))', Tensor),
]


@pytest.mark.parametrize('expr,expected_type', GAUSSIAN_SMOKE_TESTS)
def test_gaussian_joint_smoke(expr, expected_type):
    g1 = Gaussian(
        loc=torch.tensor([[0.0, 0.1, 0.2],
                          [2.0, 3.0, 4.0]]),
        precision=torch.tensor([[[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]],
                                [[1.0, 0.1, 0.2],
                                 [0.1, 1.0, 0.3],
                                 [0.2, 0.3, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('x', reals(3))]))
    assert isinstance(g1, Gaussian)

    g2 = Gaussian(
        loc=torch.tensor([[0.0, 0.1],
                          [2.0, 3.0]]),
        precision=torch.tensor([[[1.0, 0.2],
                                 [0.2, 1.0]],
                                [[1.0, 0.2],
                                 [0.2, 1.0]]]),
        inputs=OrderedDict([('i', bint(2)), ('y', reals(2))]))
    assert isinstance(g2, Gaussian)

    shift = Tensor(torch.tensor([-1., 1.]), OrderedDict([('i', bint(2))]))
    assert isinstance(shift, Tensor)

    i0 = Number(1, 2)
    assert isinstance(i0, Number)

    x0 = Tensor(torch.tensor([0.5, 0.6, 0.7]))
    assert isinstance(x0, Tensor)

    y0 = Tensor(torch.tensor([[0.2, 0.3],
                              [0.8, 0.9]]),
                inputs=OrderedDict([('i', bint(2))]))
    assert isinstance(y0, Tensor)

    with interpretation(normalize):
        result = eval(expr)

    if expected_type is not Contraction:
        result = reinterpret(result)

    assert isinstance(result, expected_type)


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
