from __future__ import absolute_import, division, print_function

import itertools
import pytest
from collections import OrderedDict

import torch

import funsor


def make_example(equation, fill=None, sizes=(2, 3)):
    symbols = sorted(set(equation) - set(',->'))
    sizes = {dim: size for dim, size in zip(symbols, itertools.cycle(sizes))}
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    outputs = outputs.split(',')
    operands = []
    for dims in inputs:
        shape = tuple(sizes[dim] for dim in dims)
        operands.append(torch.randn(shape) if fill is None else torch.full(shape, fill))
    return inputs, outputs, operands, sizes


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


EINSUM_EXAMPLES = [
    "a,b->",
    "ab,a->",
    "a,a->",
    "a,a->a",
    "a,a,a,ab->ab",
    "a,ab,bc,cd->",
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES)
def test_einsum(equation):
    inputs, outputs, operands, sizes = make_example(equation)
    funsor_operands = [
        funsor.Tensor(operand, OrderedDict([(d, funsor.ints(sizes[d])) for d in inp]))
        for inp, operand in zip(inputs, operands)
    ]
    expected = torch.einsum(equation, operands)
    actual = naive_einsum(equation, *funsor_operands)
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]
