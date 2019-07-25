import opt_einsum
import pytest
import torch
from pyro.ops.contract import einsum as pyro_einsum
from pyro.ops.einsum.adjoint import require_backward as pyro_require_backward

import funsor
from funsor.adjoint import adjoint
from funsor.domains import bint
from funsor.einsum import einsum, naive_einsum, naive_plated_einsum
from funsor.interpreter import interpretation
from funsor.terms import Variable, reflect
from funsor.testing import make_einsum_example, make_plated_hmm_einsum

# FIXME rewrite adjoint for compatibility with substitution changes
xfail_with_new_subs = pytest.mark.xfail(True, reason="fails w/ new subs")


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


@xfail_with_new_subs
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
        actual = actuals[fv]
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


@xfail_with_new_subs
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
        actual = actuals[target]

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


@xfail_with_new_subs
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
        actual = actuals[fv]
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)


OPTIMIZED_PLATED_EINSUM_EXAMPLES = [
    make_plated_hmm_einsum(num_steps, num_obs_plates=b, num_hidden_plates=a)
    for num_steps in range(20, 50, 6)
    for (a, b) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
]


@xfail_with_new_subs
@pytest.mark.parametrize('equation,plates', OPTIMIZED_PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_marginal'])
def test_optimized_plated_einsum_adjoint(equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)

    with interpretation(reflect):
        fwd_expr = einsum(equation, *funsor_operands, plates=plates, backend=backend)
    actuals = adjoint(fwd_expr, funsor_operands)

    for operand in operands:
        pyro_require_backward(operand)
    expected_out = pyro_einsum(equation, *operands,
                               modulo_total=False,
                               plates=plates,
                               backend=backend)[0]
    expected_out._pyro_backward()

    for i, (inp, tv, fv) in enumerate(zip(inputs, operands, funsor_operands)):
        actual = actuals[fv]
        expected = tv._pyro_backward_result
        if inp:
            actual = actual.align(tuple(inp))
        assert isinstance(actual, funsor.Tensor)
        assert expected.shape == actual.data.shape
        assert torch.allclose(expected, actual.data, atol=1e-7)
