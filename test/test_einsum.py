from __future__ import absolute_import, division, print_function

import pytest

import opt_einsum
import torch
from pyro.ops.contract import naive_ubersum

import funsor

from funsor.terms import reflect
from funsor.interpreter import interpretation, reinterpret
from funsor.optimizer import apply_optimizer

from funsor.testing import xfail_param, make_einsum_example


def make_hmm_einsum(num_steps):
    inputs = []
    for t in range(num_steps):
        inputs.append(str(opt_einsum.get_symbol(t)) + str(opt_einsum.get_symbol(t+1)))
        inputs.append(str(opt_einsum.get_symbol(t)))
    equation = ",".join(inputs) + "->"
    return equation


def naive_einsum(eqn, *terms):
    assert isinstance(eqn, str)
    assert all(isinstance(term, funsor.Funsor) for term in terms)
    inputs, output = eqn.split('->')
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(d for d in output)
    reduce_dims = tuple(d for d in input_dims - output_dims)
    prod = terms[0]
    for term in terms[1:]:
        prod = prod * term
    for reduce_dim in reduce_dims:
        prod = prod.sum(reduce_dim)
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
    "a,ab,bc,cd->",
    make_hmm_einsum(10),
    # make_hmm_einsum(15),  # slows down tests
    # make_hmm_einsum(20),  # slows down tests
]

XFAIL_EINSUM_EXAMPLES = [
    xfail_param("ab->ba", reason="align not implemented"),  # see pyro-ppl/funsor#26
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES + XFAIL_EINSUM_EXAMPLES)
@pytest.mark.parametrize('optimized', [False, True])
def test_einsum(equation, optimized):
    inputs, outputs, sizes, operands, funsor_operands = make_einsum_example(equation)
    expected = opt_einsum.contract(equation, *operands, backend='torch')
    if optimized:
        with interpretation(reflect):
            naive_ast = naive_einsum(equation, *funsor_operands)
            optimized_ast = apply_optimizer(naive_ast)
        print("Naive expression: {}".format(naive_ast))
        print("Optimized expression: {}".format(optimized_ast))
        actual = reinterpret(optimized_ast)  # eager by default
    else:
        actual = naive_einsum(equation, *funsor_operands)

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
