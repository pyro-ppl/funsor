from __future__ import absolute_import, division, print_function

import pytest

import opt_einsum
import torch
from pyro.ops.contract import naive_ubersum

import funsor
import funsor.ops as ops

from funsor.terms import reflect, Binary
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer

from funsor.testing import assert_close, make_einsum_example, naive_einsum


def naive_plated_einsum(eqn, *terms, **kwargs):
    assert isinstance(eqn, str)
    assert all(isinstance(term, funsor.Funsor) for term in terms)
    # ...
    raise NotImplementedError("TODO implement naive plated einsum")


EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    "a,a->",
    "a,a->a",
    "a,a,a,ab->ab",
    "ab->ba",
    "ab,bc,cd->da",
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
def test_einsum(equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = opt_einsum.contract(equation, *operands, backend=backend)

    with interpretation(reflect):
        naive_ast = naive_einsum(equation, *funsor_operands, backend=backend)
        optimized_ast = apply_optimizer(naive_ast)
    print("Naive expression: {}".format(naive_ast))
    print("Optimized expression: {}".format(optimized_ast))
    actual_optimized = reinterpret(optimized_ast)  # eager by default
    actual = naive_einsum(equation, *funsor_operands, backend=backend)

    assert_close(actual, actual_optimized, atol=1e-4)

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))

    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


PLATED_EINSUM_EXAMPLES = [(ex, '') for ex in EINSUM_EXAMPLES] + [
    ('i->', 'i'),
    ('i->i', 'i'),
    (',i->', 'i'),
    (',i->i', 'i'),
    ('ai->', 'i'),
    ('ai->i', 'i'),
    ('ai->ai', 'i'),
    (',ai,abij->aij', 'ij'),
    ('a,ai,bij->bij', 'ij'),
]


@pytest.mark.xfail(reason="naive plated einsum not implemented")
@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
def test_plated_einsum(equation, plates):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = naive_ubersum(equation, *operands, plates=plates, backend='torch', modulo_total=False)[0]
    actual = naive_plated_einsum(equation, *funsor_operands, plates=plates)
    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]
