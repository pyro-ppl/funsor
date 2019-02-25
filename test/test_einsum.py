from __future__ import absolute_import, division, print_function

import os
import itertools
import pytest
from collections import OrderedDict

import torch
from pyro.ops.contract import naive_ubersum

import funsor


if torch.cuda.is_available():
    torch.set_default_tensor_type(os.environ.get('TORCH_TENSOR_TYPE', 'torch.DoubleTensor'))


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


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
]

XFAIL_EINSUM_EXAMPLES = [
    xfail_param("ab->ba", reason="align not implemented"),  # see pyro-ppl/funsor#26
]


@pytest.mark.parametrize('equation', EINSUM_EXAMPLES + XFAIL_EINSUM_EXAMPLES)
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


@pytest.mark.xfail(reason="naive plated einsum not implemented")
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
