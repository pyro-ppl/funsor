from __future__ import absolute_import, division, print_function

import itertools
import operator
from collections import OrderedDict

import pytest
import torch
from six.moves import reduce

import funsor.ops as ops
from funsor.domains import Domain, bint
from funsor.gaussian import Gaussian
from funsor.terms import Binary, Funsor
from funsor.torch import Tensor


def assert_close(actual, expected, atol=1e-6, rtol=1e-6):
    assert isinstance(actual, Funsor)
    assert isinstance(expected, Funsor)
    assert actual.inputs == expected.inputs, (actual.inputs, expected.inputs)
    assert actual.output == expected.output
    if isinstance(actual, Tensor):
        if actual.data.dtype in (torch.long, torch.uint8):
            assert (actual.data == expected.data).all()
        else:
            diff = (actual.data.detach() - expected.data.detach()).abs()
            assert diff.max() < atol
            assert (diff / (atol + expected.data.detach().abs())).max() < rtol
    else:
        raise ValueError('cannot compare objects of type {}'.format(type(actual)))


def check_funsor(x, inputs, output, data=None):
    """
    Check dims and shape modulo reordering.
    """
    assert isinstance(x, Funsor)
    assert dict(x.inputs) == dict(inputs)
    if output is not None:
        assert x.output == output
    if data is not None:
        if x.inputs == inputs:
            x_data = x.data
        else:
            x_data = x.align(tuple(inputs)).data
        if inputs or output.shape:
            assert (x_data == data).all()
        else:
            assert x_data == data


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


def make_einsum_example(equation, fill=None, sizes=(2, 3)):
    symbols = sorted(set(equation) - set(',->'))
    sizes = {dim: size for dim, size in zip(symbols, itertools.cycle(sizes))}
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')
    outputs = outputs.split(',')
    operands = []
    for dims in inputs:
        shape = tuple(sizes[dim] for dim in dims)
        operands.append(torch.randn(shape) if fill is None else torch.full(shape, fill))
    funsor_operands = [
        Tensor(operand, OrderedDict([(d, bint(sizes[d])) for d in inp]))
        for inp, operand in zip(inputs, operands)
    ]

    assert equation == \
        ",".join(["".join(operand.inputs.keys()) for operand in funsor_operands]) + "->" + ",".join(outputs)
    return inputs, outputs, sizes, operands, funsor_operands


def assert_equiv(x, y):
    """
    Check that two funsors are equivalent up to permutation of inputs.
    """
    check_funsor(x, y.inputs, y.output, y.data)


def random_tensor(inputs, output):
    """
    Creates a random :class:`funsor.torch.Tensor` with given inputs and output.
    """
    assert isinstance(inputs, OrderedDict)
    assert isinstance(output, Domain)
    shape = tuple(d.dtype for d in inputs.values()) + output.shape
    if output.dtype == 'real':
        data = torch.randn(shape)
    else:
        num_elements = reduce(operator.mul, shape, 1)
        data = torch.multinomial(torch.ones(output.dtype),
                                 num_elements,
                                 replacement=True).reshape(shape)
    return Tensor(data, inputs, output.dtype)


def random_gaussian(inputs):
    """
    Creates a random :class:`funsor.gaussian.Gaussian` with given inputs.
    """
    assert isinstance(inputs, OrderedDict)
    batch_shape = tuple(d.dtype for d in inputs.values() if d.dtype != 'real')
    event_shape = (sum(d.num_elements for d in inputs.values() if d.dtype == 'real'),)
    log_density = torch.randn(batch_shape)
    loc = torch.randn(batch_shape + event_shape)
    prec_sqrt = torch.randn(batch_shape + event_shape + event_shape)
    precision = torch.matmul(prec_sqrt, prec_sqrt.transpose(-1, -2))
    return Gaussian(log_density, loc, precision, inputs)


def naive_einsum(eqn, *terms, **kwargs):
    backend = kwargs.pop('backend', 'torch')
    if backend == 'torch':
        sum_op, prod_op = ops.add, ops.mul
    elif backend == 'pyro.ops.einsum.torch_log':
        sum_op, prod_op = ops.logaddexp, ops.add
    else:
        raise ValueError("{} backend not implemented".format(backend))

    assert isinstance(eqn, str)
    assert all(isinstance(term, Funsor) for term in terms)
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
