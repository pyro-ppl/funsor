from collections import OrderedDict

import opt_einsum
import pyro.ops.contract as pyro_einsum
import pytest
import torch

import funsor
from funsor.distributions import Categorical
from funsor.domains import bint
from funsor.einsum import einsum, naive_contract_einsum, naive_einsum, naive_plated_einsum
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer, optimize
from funsor.terms import Variable, lazy, normalize, reflect
from funsor.testing import assert_close, make_chain_einsum, make_einsum_example, make_hmm_einsum, make_plated_hmm_einsum
from funsor.torch import Tensor

OPTIMIZED_EINSUM_EXAMPLES = [
    make_chain_einsum(t) for t in range(2, 50, 10)
] + [
    make_hmm_einsum(t) for t in range(2, 50, 10)
]


@pytest.mark.parametrize('equation', OPTIMIZED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize("einsum_impl", [
    naive_einsum,
    # naive_contract_einsum,  # XXX not working, probably issue with canonicalization
])
def test_optimized_einsum(equation, backend, einsum_impl):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = opt_einsum.contract(equation, *operands, backend=backend)
    with interpretation(normalize):
        naive_ast = einsum_impl(equation, *funsor_operands, backend=backend)
    optimized_ast = apply_optimizer(naive_ast)
    actual = reinterpret(optimized_ast)  # eager by default

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))

    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]


@pytest.mark.parametrize("eqn1,eqn2", [
    ("ab,bc,cd->d", "de,ef,fg->"),
])
@pytest.mark.parametrize("optimize1", [False, True])
@pytest.mark.parametrize("optimize2", [False, True])
@pytest.mark.parametrize("backend1", ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize("backend2", ['torch', 'pyro.ops.einsum.torch_log'])
@pytest.mark.parametrize("einsum_impl", [naive_einsum, naive_contract_einsum])
def test_nested_einsum(eqn1, eqn2, optimize1, optimize2, backend1, backend2, einsum_impl):
    inputs1, outputs1, sizes1, operands1, _ = make_einsum_example(eqn1, sizes=(3,))
    inputs2, outputs2, sizes2, operands2, funsor_operands2 = make_einsum_example(eqn2, sizes=(3,))

    # normalize the probs for ground-truth comparison
    operands1 = [operand.abs() / operand.abs().sum(-1, keepdim=True)
                 for operand in operands1]

    expected1 = opt_einsum.contract(eqn1, *operands1, backend=backend1)
    expected2 = opt_einsum.contract(outputs1[0] + "," + eqn2, *([expected1] + operands2), backend=backend2)

    with interpretation(normalize):
        funsor_operands1 = [
            Categorical(probs=Tensor(
                operand,
                inputs=OrderedDict([(d, bint(sizes1[d])) for d in inp[:-1]])
            ))(value=Variable(inp[-1], bint(sizes1[inp[-1]]))).exp()
            for inp, operand in zip(inputs1, operands1)
        ]

        output1_naive = einsum_impl(eqn1, *funsor_operands1, backend=backend1)
        with interpretation(reflect):
            output1 = apply_optimizer(output1_naive) if optimize1 else output1_naive
        output2_naive = einsum_impl(outputs1[0] + "," + eqn2, *([output1] + funsor_operands2), backend=backend2)
        with interpretation(reflect):
            output2 = apply_optimizer(output2_naive) if optimize2 else output2_naive

    actual1 = reinterpret(output1)
    actual2 = reinterpret(output2)

    assert torch.allclose(expected1, actual1.data)
    assert torch.allclose(expected2, actual2.data)


PLATED_EINSUM_EXAMPLES = [
    make_plated_hmm_einsum(num_steps, num_obs_plates=b, num_hidden_plates=a)
    for num_steps in range(3, 50, 6)
    for (a, b) in [(0, 1), (0, 2), (0, 0), (1, 1), (1, 2)]
]


@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_log'])
def test_optimized_plated_einsum(equation, plates, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = pyro_einsum.einsum(equation, *operands, plates=plates, backend=backend)[0]
    actual = einsum(equation, *funsor_operands, plates=plates, backend=backend)

    if len(equation) < 10:
        actual_naive = naive_plated_einsum(equation, *funsor_operands, plates=plates, backend=backend)
        assert_close(actual, actual_naive)

    assert isinstance(actual, funsor.Tensor) and len(outputs) == 1
    if len(outputs[0]) > 0:
        actual = actual.align(tuple(outputs[0]))

    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]
