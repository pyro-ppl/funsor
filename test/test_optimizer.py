from __future__ import absolute_import, division, print_function

import pytest
from collections import OrderedDict

import opt_einsum
import torch
from pyro.ops.contract import naive_ubersum

import funsor
import funsor.ops as ops

from funsor.distributions import Categorical
from funsor.domains import bint
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer
from funsor.terms import reflect, Binary, Variable
from funsor.torch import Tensor

from funsor.testing import assert_close, make_einsum_example, naive_einsum


def make_chain_einsum(num_steps):
    inputs = [str(opt_einsum.get_symbol(0))]
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t+1)))
    equation = ",".join(inputs) + "->"
    return equation


def make_hmm_einsum(num_steps):
    inputs = [str(opt_einsum.get_symbol(0))]
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t+1)))
        inputs.append(str(opt_einsum.get_symbol(t+1)))
    equation = ",".join(inputs) + "->"
    return equation


OPTIMIZED_EINSUM_EXAMPLES = [
    make_chain_einsum(t) for t in range(2, 50, 10)
] + [
    make_hmm_einsum(t) for t in range(2, 50, 10)
]


@pytest.mark.parametrize('equation', OPTIMIZED_EINSUM_EXAMPLES)
@pytest.mark.parametrize('backend', ['pyro.ops.einsum.torch_log'])
def test_optimized_einsum(equation, backend):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = opt_einsum.contract(equation, *operands, backend=backend)
    with interpretation(reflect):
        naive_ast = naive_einsum(equation, *funsor_operands, backend=backend)
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


@pytest.mark.xfail(reason="recursion errors?")
@pytest.mark.parametrize("eqn1,eqn2", [
    ("ab,bc,cd->d", "de,ef,fg->"),
])
@pytest.mark.parametrize("optimize1", [False, True])
@pytest.mark.parametrize("optimize2", [False, True])
@pytest.mark.parametrize("backend1", ['torch'])
@pytest.mark.parametrize("backend2", ['torch', 'pyro.ops.einsum.torch_log'])
def test_nested_einsum(eqn1, eqn2, optimize1, optimize2, backend1, backend2):
    inputs1, outputs1, sizes1, operands1, _ = make_einsum_example(eqn1, sizes=(3,))
    inputs2, outputs2, sizes2, operands2, funsor_operands2 = make_einsum_example(eqn2, sizes=(3,))

    operands1 = [torch.distributions.Categorical(probs=operand).probs
                 for operand in operands1]

    expected1 = opt_einsum.contract(eqn1, *operands1, backend=backend1)
    expected2 = opt_einsum.contract(outputs1[0] + "," + eqn2, *([expected1] + operands2), backend=backend2)

    with interpretation(reflect):
        funsor_operands1 = [
            Categorical(probs=Tensor(
                operand,  # XXX .abs()?
                inputs=OrderedDict([(d, bint(sizes1[d])) for d in inp[:-1]])
            ))(value=Variable(inp[-1], bint(sizes1[inp[-1]])))
            for inp, operand in zip(inputs1, operands1)
        ]

    with interpretation(reflect):
        output1 = naive_einsum(eqn1, *tuple(funsor_operands1), backend=backend1)
        output1 = apply_optimizer(output1) if optimize1 else output1
        output2 = naive_einsum(outputs1[0] + "," + eqn2, *tuple([output1] + funsor_operands2), backend=backend2)
        output2 = apply_optimizer(output2) if optimize2 else output2

    actual2 = reinterpret(output2)
    assert torch.allclose(expected2, actual2.data)
