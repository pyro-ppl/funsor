from __future__ import absolute_import, division, print_function

import pytest
from collections import OrderedDict

import opt_einsum
import torch
from pyro.ops.contract import naive_ubersum

import funsor

from funsor.distributions import Categorical
from funsor.domains import bint
from funsor.terms import reflect, Variable
from funsor.torch import Tensor
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.testing import assert_close, make_einsum_example

from funsor.einsum import naive_einsum, naive_plated_einsum


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

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_optimized = actual_optimized.align(tuple(outputs[0]))

    assert_close(actual, actual_optimized, atol=1e-4)
    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
def test_einsum_categorical(equation):
    inputs, outputs, sizes, operands, _ = make_einsum_example(equation)
    operands = [operand.abs() / operand.abs().sum(-1, keepdim=True)
                for operand in operands]

    expected = opt_einsum.contract(equation, *operands, backend='torch')

    with interpretation(reflect):
        funsor_operands = [
            Categorical(probs=Tensor(
                operand,
                inputs=OrderedDict([(d, bint(sizes[d])) for d in inp[:-1]])
            ))(value=Variable(inp[-1], bint(sizes[inp[-1]]))).exp()
            for inp, operand in zip(inputs, operands)
        ]

        naive_ast = naive_einsum(equation, *funsor_operands)
        optimized_ast = apply_optimizer(naive_ast)

    print("Naive expression: {}".format(naive_ast))
    print("Optimized expression: {}".format(optimized_ast))
    actual_optimized = reinterpret(optimized_ast)  # eager by default
    actual = naive_einsum(equation, *map(reinterpret, funsor_operands))

    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_optimized = actual_optimized.align(tuple(outputs[0]))

    assert_close(actual, actual_optimized, atol=1e-4)

    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


PLATED_EINSUM_EXAMPLES = [
    ('i->', 'i'),
    (',i->', 'i'),
    ('ai->', 'i'),
    (',ai,abij->', 'ij'),
    ('a,ai,bij->', 'ij'),
    ('ai,abi,bci,cdi->', 'i'),
    ('aij,abij,bcij,cdij->', 'ij'),
    ('a,abi,bcij,cdij->', 'ij'),
]


@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
def test_plated_einsum(equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = naive_ubersum(equation, *operands, plates=plates, backend=backend, modulo_total=False)[0]
    with interpretation(reflect):
        naive_ast = naive_plated_einsum(equation, *funsor_operands, plates=plates, backend=backend)
        optimized_ast = apply_optimizer(naive_ast)
    actual_optimized = reinterpret(optimized_ast)  # eager by default
    actual = naive_plated_einsum(equation, *funsor_operands, plates=plates, backend=backend)

    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))
        actual_optimized = actual_optimized.align(tuple(outputs[0]))

    assert_close(actual, actual_optimized, atol=1e-4)

    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]
