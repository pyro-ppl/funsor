from __future__ import absolute_import, division, print_function

import pytest

import opt_einsum
import torch
from pyro.ops.contract import einsum as pyro_einsum
from pyro.ops.einsum.adjoint import require_backward as pyro_require_backward

import funsor

from funsor.domains import bint
from funsor.terms import reflect, Variable
from funsor.interpreter import interpretation, reinterpret
from funsor.testing import make_einsum_example  # , xfail_param

from funsor.einsum import naive_einsum, naive_plated_einsum, einsum
from funsor.adjoint import adjoint


EINSUM_EXAMPLES = [
    "a->",
    "ab->",
    ",->",
    ",,->",
    "a,a->a",
    "a,a,a->a",
    "a,b->",
    "ab,a->",
    "a,b,c->",
    "a,a->",
    "a,a,a,ab->",
    "abc,bcd,cde->",
    "ab,bc,cd->",
    "ab,b,bc,c,cd,d->",
]


@pytest.mark.parametrize('einsum_impl', [naive_einsum, einsum])
@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_marginal'])
def test_einsum_adjoint(einsum_impl, equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        fwd_expr = einsum_impl(equation, *funsor_operands, backend=backend)
        actuals = adjoint(fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands,
                               modulo_total=True,
                               backend=backend)[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = reinterpret(actuals[fv])
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


@pytest.mark.parametrize('einsum_impl', [naive_einsum, einsum])
@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_marginal'])
def test_einsum_adjoint_unary_marginals(einsum_impl, equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    equation = ",".join(inputs) + "->"

    targets = [Variable(k, bint(sizes[k])) for k in set(sizes)]
    with interpretation(reflect):
        fwd_expr = einsum_impl(equation, *funsor_operands, backend=backend)
        actuals = adjoint(fwd_expr, targets)

    for target in targets:
        actual = reinterpret(actuals[target])

        expected = opt_einsum.contract(equation + target.name, *operands,
                                       backend=backend)
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


PLATED_EINSUM_EXAMPLES = [
    ('i->', 'i'),
    (',i->', 'i'),
    ('ai->', 'i'),
    (',ai,abij->', 'ij'),
    ('a,ai,bij->', 'ij'),
    ('ai,abi,bci,cdi->', 'i'),
    ('aij,abij,bcij->', 'ij'),
    ('a,abi,bcij,cdij->', 'ij'),
]


@pytest.mark.parametrize('einsum_impl', [naive_plated_einsum, einsum])
@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_marginal'])
def test_plated_einsum_adjoint(einsum_impl, equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        fwd_expr = einsum_impl(equation, *funsor_operands, plates=plates, backend=backend)
        actuals = adjoint(fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands,
                               modulo_total=False,
                               plates=plates,
                               backend=backend)[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        print(actuals[fv])
        actual = reinterpret(actuals[fv])
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)
