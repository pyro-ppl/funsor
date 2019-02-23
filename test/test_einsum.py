from __future__ import absolute_import, division, print_function

import itertools
import pytest
from collections import OrderedDict
from six.moves import reduce

import torch
from pyro.ops.contract import naive_ubersum

import funsor
import funsor.ops as ops


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


def naive_plated_einsum(eqn, *terms, plates='', modulo_total=False):
    assert isinstance(eqn, str)
    assert all(isinstance(term, funsor.Funsor) for term in terms)
    inputs, output = eqn.split('->')
    plate_dims = frozenset(plates)
    input_dims = frozenset(d for inp in inputs.split(',') for d in inp)
    output_dims = frozenset(d for d in output)
    reduce_dims = tuple(d for d in input_dims - output_dims - plate_dims)

    tensor_tree = OrderedDict()
    for inp, term in zip(inputs, terms):
        ordinal = frozenset(inp) & plate_dims
        tensor_tree.setdefault(ordinal, []).append(term)

    while any(t.input.keys()  for t in tensor_tree.values()):
        ordinal = max(tensor_tree, key=len)
        leaf_terms = tensor_tree.pop(ordinal)
        new_term = reduce(ops.mul, leaf_terms[1:], leaf_terms[0])
        for plate_dim in (ordinal - output_dims):
            new_term = new_term.prod(plate_dim)
        new_ordinal = frozenset(new_term.inputs.keys()) & plate_dims
        tensor_tree.setdefault(new_ordinal, []).append(new_term)

    ordinal, output_terms = tensor_tree.popitem()
    output_term = reduce(ops.mul, output_terms[1:], output_terms[0])

    for plate_dim in (ordinal - output_dims):
        output_term = output_term.prod(plate_dim)

    for reduce_dim in reduce_dims:
        output_term = output_term.sum(reduce_dim)

    # raise NotImplementedError("TODO implement naive plated einsum")
    return output_term


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


# @pytest.mark.xfail(reason="naive plated einsum not implemented")
@pytest.mark.parametrize('equation,plates', PLATED_EINSUM_EXAMPLES)
def test_plated_einsum(equation, plates):
    inputs, outputs, operands, sizes = make_example(equation)
    funsor_operands = [
        funsor.Tensor(operand, OrderedDict([(d, funsor.ints(sizes[d])) for d in inp]))
        for inp, operand in zip(inputs, operands)
    ]
    expected = naive_ubersum(equation, *operands, plates=plates, backend='torch', modulo_total=False)[0]
    actual = naive_plated_einsum(equation, *funsor_operands, plates=plates)
    assert expected.shape == actual.data.shape
    assert torch.allclose(expected, actual.data)
    for output in outputs:
        for i, output_dim in enumerate(output):
            assert output_dim in actual.inputs
            assert actual.inputs[output_dim].dtype == sizes[output_dim]
