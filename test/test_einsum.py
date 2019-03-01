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

from funsor.testing import assert_close, make_einsum_example


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


def naive_einsum(eqn, *terms, **kwargs):
    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend == 'pyro.ops.einsum.torch_log':
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, funsor.Funsor) for term in terms)
    inputs, output = eqn.split('->')
    assert len(output.split(',')) == 1
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(d for d in output)
    reduce_dims = tuple(d for d in input_dims - output_dims)
    prod = terms[0]
    for term in terms[1:]:
        prod = Binary(prod_op, prod, term)
    for reduce_dim in reduce_dims:
        prod = prod.reduce(sum_op, reduce_dim)
    return prod


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
    make_chain_einsum(5),
    make_hmm_einsum(6),
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('backend', ['torch', 'pyro.ops.einsum.torch_log'])
def test_einsum(equation, optimized, backend):
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


OPTIMIZED_EINSUM_EXAMPLES = [
    make_chain_einsum(t) for t in range(2, 100, 10)
] + [
    make_hmm_einsum(t) for t in range(2, 100, 10)
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
